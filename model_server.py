import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn
import os
import time

# ==========================================
# 1. CẤU HÌNH & MODEL
# ==========================================
class Config:
    BERT_NAME = "local_phobert" 
    HIDDEN_SIZE = 768
    NUM_CLASSES = 4 
    GRU_HIDDEN = 128
    CNN_FILTERS = 64
    CNN_KERNEL_SIZES = [2, 3, 4]

# --- Model 1: ViTHSD ---
class PhoBertHybridModel(nn.Module):
    def __init__(self, config):
        super(PhoBertHybridModel, self).__init__()
        self.bert = AutoModel.from_pretrained(config.BERT_NAME, local_files_only=True)
        self.gru = nn.GRU(input_size=config.HIDDEN_SIZE, hidden_size=config.GRU_HIDDEN, bidirectional=True, batch_first=True)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=config.HIDDEN_SIZE, out_channels=config.CNN_FILTERS, kernel_size=k)
            for k in config.CNN_KERNEL_SIZES
        ])
        concat_dim = (config.GRU_HIDDEN * 2) + (config.CNN_FILTERS * len(config.CNN_KERNEL_SIZES))
        self.dense = nn.Linear(concat_dim, 256)
        self.batch_norm = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.3)
        self.fc_individual = nn.Linear(256, config.NUM_CLASSES)
        self.fc_group = nn.Linear(256, config.NUM_CLASSES)
        self.fc_societal = nn.Linear(256, config.NUM_CLASSES)

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        cls_embedding = bert_out[:, 0, :] 
        
        gru_out, _ = self.gru(bert_out)
        # Pooling: Logic đơn giản hóa cho JIT ổn định
        gru_pool = torch.max(gru_out, dim=1)[0]
        
        cnn_in = bert_out.permute(0, 2, 1)
        cnn_outs = []
        for conv in self.convs:
            x = torch.relu(conv(cnn_in))
            x = torch.max_pool1d(x, kernel_size=x.shape[2]).squeeze(2)
            cnn_outs.append(x)
        cnn_pool = torch.cat(cnn_outs, dim=1)
        
        combined = torch.cat([gru_pool, cnn_pool], dim=1)
        x = self.dropout(torch.relu(self.batch_norm(self.dense(combined))))
        
        return self.fc_individual(x), self.fc_group(x), self.fc_societal(x), cls_embedding

# --- Model 2: Head ---
class TypeAttackHead(nn.Module):
    def __init__(self, input_dim=768, num_classes=19):
        super(TypeAttackHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 2. SERVER SETUP (TỐI ƯU CPU)
# ==========================================
app = FastAPI()
device = torch.device("cpu") 

# Tối ưu luồng: Tận dụng tối đa core vật lý (thường là 8-16) cho 1 process duy nhất
torch.set_num_threads(8) 

resources = {}

@app.on_event("startup")
def load_resources():
    print(f">>> Server running on: {device} (Optimized with TorchScript)")
    try:
        # 1. Tokenizer
        if os.path.exists("local_phobert"):
            resources["tokenizer"] = AutoTokenizer.from_pretrained("local_phobert", local_files_only=True)
        else:
            resources["tokenizer"] = AutoTokenizer.from_pretrained("vinai/phobert-base")

        # 2. Load Model 1 & JIT Trace
        model1 = PhoBertHybridModel(Config())
        if os.path.exists("model_1.pth"):
            state = torch.load("model_1.pth", map_location=device)
            new_state = {k.replace("module.", ""): v for k, v in state.items()}
            model1.load_state_dict(new_state, strict=False)
            model1.eval()
            
            # --- JIT OPTIMIZATION ---
            print("Optimizing Model 1 with TorchScript...")
            try:
                # Tạo dummy input để JIT học luồng dữ liệu
                dummy_ids = torch.randint(0, 100, (1, 64), dtype=torch.long)
                dummy_mask = torch.ones((1, 64), dtype=torch.long)
                
                # Trace & Freeze
                traced_model1 = torch.jit.trace(model1, (dummy_ids, dummy_mask))
                traced_model1 = torch.jit.optimize_for_inference(traced_model1)
                resources["model_1"] = traced_model1
                print("Model 1 Optimized successfully")
            except Exception as e:
                print(f"JIT Failed ({e}), using standard model")
                resources["model_1"] = model1
        else:
            print("Error: model_1.pth not found!")

        # 3. Load Model 2 & JIT Trace
        if os.path.exists("model2_head.pth"):
            model2 = TypeAttackHead()
            model2.load_state_dict(torch.load("model2_head.pth", map_location=device))
            model2.eval()
            
            # JIT cho Model 2
            print("Optimizing Model 2 with TorchScript...")
            dummy_embed = torch.randn(1, 768)
            traced_model2 = torch.jit.trace(model2, dummy_embed)
            traced_model2 = torch.jit.optimize_for_inference(traced_model2)
            
            resources["model_2"] = traced_model2
            print("Model 2 Optimized successfully")
        else:
            print("Error: model2_head.pth not found!")

    except Exception as e:
        print(f"Init Error: {e}")

class CommentRequest(BaseModel):
    id: str
    text: str

class BatchRequest(BaseModel):
    batch: List[CommentRequest]

@app.post("/predict_batch")
async def predict_batch(req: BatchRequest):
    t0 = time.time()
    items = req.batch
    texts = [item.text for item in items]
    ids = [item.id for item in items]
    batch_size = len(items)
    
    final_targets = [[0,0,0]] * batch_size
    final_attacks = [""] * batch_size

    if "tokenizer" in resources and "model_1" in resources:
        tokenizer = resources["tokenizer"]
        model1 = resources["model_1"]
        
        # Tokenize: Max length 60 là đủ cho cmt, tăng tốc đáng kể
        inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=60)
        input_ids = inputs["input_ids"].to(device)
        attn_mask = inputs["attention_mask"].to(device)
        
        t1 = time.time()
        
        with torch.no_grad():
            # --- MODEL 1 (JIT) ---
            # Output JIT luôn là Tuple
            o1, o2, o3, cls_embeddings = model1(input_ids, attn_mask)
            
            pred_ind = torch.argmax(o1, dim=1)
            pred_grp = torch.argmax(o2, dim=1)
            pred_soc = torch.argmax(o3, dim=1)
            
            targets_tensor = torch.stack([pred_ind, pred_grp, pred_soc], dim=1)
            mask_hate = (targets_tensor >= 2).any(dim=1) 
            
            final_targets = targets_tensor.tolist()
            t2 = time.time()

            # --- MODEL 2 (JIT) ---
            if "model_2" in resources and mask_hate.any():
                model2 = resources["model_2"]
                
                hate_embeddings = cls_embeddings[mask_hate]
                head_outputs = model2(hate_embeddings) 
                probs = torch.sigmoid(head_outputs)
                
                binary_matrix = (probs > 0.3).int()
                
                # Xử lý chuỗi nhị phân (CPU thuần)
                hate_indices = torch.nonzero(mask_hate).squeeze()
                if hate_indices.ndim == 0: hate_indices = hate_indices.unsqueeze(0)
                
                hate_indices_list = hate_indices.tolist()
                binary_list = binary_matrix.tolist()
                
                for idx, row_bin in zip(hate_indices_list, binary_list):
                    bin_str = "".join(map(str, row_bin))
                    if len(bin_str) > 19: bin_str = bin_str[:19]
                    elif len(bin_str) < 19: bin_str = bin_str.ljust(19, '0')
                    final_attacks[idx] = bin_str
            
            t3 = time.time()

    results = [
        {"id": ids[i], "text": texts[i], "targets": final_targets[i], "type_attack_binary": final_attacks[i]}
        for i in range(batch_size)
    ]
    
    # Log gọn nhẹ để debug hiệu năng
    print(f"Batch {batch_size} | M1: {t2-t1:.3f}s | M2: {t3-t2:.3f}s | Total: {t3-t0:.3f}s")
    
    return {"results": results}

if __name__ == "__main__":
    # Workers=1 để tránh OOM và tận dụng đa luồng (threads=8) hiệu quả hơn trên CPU
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)