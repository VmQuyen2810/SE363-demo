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

# ==============================================================================
# 1. CẤU HÌNH
# ==============================================================================
class Config:
    BERT_NAME = "local_phobert" 
    HIDDEN_SIZE = 768
    NUM_CLASSES = 4 
    GRU_HIDDEN = 128
    CNN_FILTERS = 64
    CNN_KERNEL_SIZES = [2, 3, 4]

# ==============================================================================
# 2. KIẾN TRÚC MODEL (GIỮ NGUYÊN)
# ==============================================================================
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
        try:
            gru_pool = torch.max(gru_out, dim=1)[0]
        except:
            gru_pool = torch.max_pool1d(gru_out.permute(0, 2, 1), kernel_size=gru_out.shape[1]).squeeze(2)
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

# ==============================================================================
# 3. SERVER SETUP & LOGIC TỐI ƯU
# ==============================================================================
app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(4) 

resources = {}

@app.on_event("startup")
def load_resources():
    print(f">>> Server running on: {device}")
    try:
        # 1. Tokenizer
        if os.path.exists("local_phobert"):
            resources["tokenizer"] = AutoTokenizer.from_pretrained("local_phobert", local_files_only=True)
        else:
            resources["tokenizer"] = AutoTokenizer.from_pretrained("vinai/phobert-base")

        # 2. Model 1
        model1 = PhoBertHybridModel(Config())
        if os.path.exists("model_1.pth"):
            state_dict = torch.load("model_1.pth", map_location=device)
            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model1.load_state_dict(new_state_dict, strict=False)
            model1.to(device).eval()
            resources["model_1"] = model1
            print("✅ Model 1 Loaded")
        else:
            print("❌ Error: model_1.pth not found!")

        # 3. Model 2
        model2_path = "model2_head.pth" 
        if os.path.exists(model2_path):
            model2 = TypeAttackHead(input_dim=768, num_classes=19)
            model2.load_state_dict(torch.load(model2_path, map_location=device))
            model2.to(device).eval()
            resources["model_2"] = model2
            print("✅ Model 2 Loaded")
        else:
            print(f"❌ Error: {model2_path} not found!")

    except Exception as e:
        print(f"❌ Init Error: {e}")

class CommentRequest(BaseModel):
    id: str
    text: str

class BatchRequest(BaseModel):
    batch: List[CommentRequest]

@app.post("/predict_batch")
async def predict_batch(req: BatchRequest):
    start_time = time.time()
    items = req.batch
    
    # Tách data nhanh
    texts = [item.text for item in items]
    ids = [item.id for item in items]
    batch_size = len(items)
    
    # Kết quả mặc định
    final_targets = [[0,0,0]] * batch_size
    final_attacks = [""] * batch_size

    if "tokenizer" in resources and "model_1" in resources:
        tokenizer = resources["tokenizer"]
        model1 = resources["model_1"]
        
        # Tokenize (CPU Bound)
        inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=100)
        input_ids = inputs["input_ids"].to(device)
        attn_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            # --- MODEL 1 INFERENCE ---
            o1, o2, o3, cls_embeddings = model1(input_ids, attn_mask)
            
            # Lấy argmax ngay trên GPU để nhanh hơn
            pred_ind = torch.argmax(o1, dim=1)
            pred_grp = torch.argmax(o2, dim=1)
            pred_soc = torch.argmax(o3, dim=1)
            
            # Gom kết quả lại thành tensor (batch_size, 3)
            # targets_tensor: [[1,0,3], [0,2,0], ...]
            targets_tensor = torch.stack([pred_ind, pred_grp, pred_soc], dim=1)
            
            # --- TỐI ƯU LOGIC CHỌN MODEL 2 ---
            # Tạo mask: Dòng nào có ít nhất 1 nhãn >= 3 (Hate) thì là True
            # (target >= 3).any(dim=1) trả về [True, False, True...]
            mask_hate = (targets_tensor >= 3).any(dim=1)
            
            # Chuyển kết quả target về CPU list để trả về API
            final_targets = targets_tensor.cpu().tolist()

            # --- MODEL 2 INFERENCE (CHỈ CHẠY VỚI DÒNG HATE) ---
            if "model_2" in resources and mask_hate.any():
                model2 = resources["model_2"]
                
                # Lọc lấy embedding của các dòng Hate bằng boolean indexing (Rất nhanh)
                hate_embeddings = cls_embeddings[mask_hate] 
                
                # Chạy Model 2
                head_outputs = model2(hate_embeddings)
                probs = torch.sigmoid(head_outputs) # Vẫn trên GPU
                
                # Thresholding bằng vector (Nhanh hơn loop python)
                # Tạo ma trận 0/1: [1, 0, 1, 1...]
                binary_matrix = (probs > 0.3).int().cpu().numpy()
                
                # Map ngược lại vị trí gốc
                hate_indices_cpu = torch.nonzero(mask_hate).squeeze().cpu().numpy()
                # Nếu chỉ có 1 dòng hate, numpy trả về scalar, cần chuyển về mảng 1D
                if np.ndim(hate_indices_cpu) == 0:
                    hate_indices_cpu = [hate_indices_cpu]

                # Convert ma trận 0/1 thành chuỗi string "1010..."
                # Đoạn này bắt buộc dùng loop nhẹ nhưng số lượng ít hơn
                for i, row_bin in enumerate(binary_matrix):
                    orig_idx = hate_indices_cpu[i]
                    # Join mảng numpy thành chuỗi nhanh
                    bin_str = "".join(row_bin.astype(str))
                    # Cắt hoặc Pad cho đủ 19 ký tự (nếu model output khác 19)
                    if len(bin_str) > 19: bin_str = bin_str[:19]
                    elif len(bin_str) < 19: bin_str = bin_str.ljust(19, '0')
                    
                    final_attacks[orig_idx] = bin_str

    # Tạo response
    results = [
        {
            "id": ids[i],
            "text": texts[i],
            "targets": final_targets[i],
            "type_attack_binary": final_attacks[i]
        }
        for i in range(batch_size)
    ]
    
    # Log tốc độ để monitor
    process_time = time.time() - start_time
    print(f"⚡ Batch {batch_size} | Hate: {sum([1 for t in final_targets if any(x>=3 for x in t)])} | Time: {process_time:.4f}s")
    
    return {"results": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)