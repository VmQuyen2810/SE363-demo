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
import json

# ==============================================================================
# 1. CẤU HÌNH & LABEL MAP
# ==============================================================================
class Config:
    BERT_NAME = "local_phobert" 
    HIDDEN_SIZE = 768
    NUM_CLASSES = 4 
    GRU_HIDDEN = 128
    CNN_FILTERS = 64
    CNN_KERNEL_SIZES = [2, 3, 4]

# Label Map cho Model 2
TYPE_ATTACK_LABELS = [
    "Threat", "Scam", "Misinformation", "Boycott",
    "Body Shaming", "Sexual Harassment", "Intelligence", "Moral", "Victim Blaming",
    "Gender", "Regionalism", "Racism", "Classism", "Religion",
    "Politics", "Social Issues", "Product", "Community",
    "Other"
]

# ==============================================================================
# 2. KIẾN TRÚC MODEL
# ==============================================================================

# --- Model 1: ViTHSD (Hybrid) - CÓ SỬA ĐỔI ---
# Sửa hàm forward để trả về thêm Embedding cho Model 2 dùng ké
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
        
        # --- TRÍCH XUẤT EMBEDDING CHO MODEL 2 ---
        # Lấy CLS token (vector đại diện cho cả câu)
        # Model 2 của bạn được train dựa trên cái này
        cls_embedding = bert_out[:, 0, :] 
        
        # --- PHẦN CÒN LẠI CỦA MODEL 1 ---
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
        
        # Trả về cả 3 output gốc VÀ cls_embedding
        return self.fc_individual(x), self.fc_group(x), self.fc_societal(x), cls_embedding

# --- Model 2 Head (MLP nhẹ) ---
# Copy y nguyên từ notebook retrain.ipynb của bạn
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
# 3. SERVER SETUP
# ==============================================================================
app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Giới hạn thread CPU để tối ưu batch
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

        # 2. Model 1 (ViTHSD)
        model1 = PhoBertHybridModel(Config())
        if os.path.exists("model_1.pth"):
            state_dict = torch.load("model_1.pth", map_location=device)
            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            try:
                model1.load_state_dict(new_state_dict, strict=True)
                print("✅ Model 1 Loaded (Hybrid Core)")
            except:
                model1.load_state_dict(new_state_dict, strict=False)
                print("⚠️ Model 1 Loaded (Non-strict)")
            model1.to(device)
            model1.eval()
            resources["model_1"] = model1
        else:
            print("❌ Error: model_1.pth not found!")

        # 3. Model 2 (Head Only)
        # Lưu ý: File này là file bạn mới lưu từ retrain.ipynb (chỉ vài MB)
        model2_path = "model2_head.pth" 
        
        if os.path.exists(model2_path):
            model2 = TypeAttackHead(input_dim=768, num_classes=19)
            state_dict2 = torch.load(model2_path, map_location=device)
            
            try:
                model2.load_state_dict(state_dict2)
                print("✅ Model 2 Head Loaded (Lightweight MLP)")
            except Exception as e:
                print(f"❌ Error loading Model 2 Head: {e}")
                
            model2.to(device)
            model2.eval()
            resources["model_2"] = model2
        else:
            print(f"❌ Error: {model2_path} not found! (Hãy copy file từ notebook về)")

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
    texts = [item.text for item in items]
    ids = [item.id for item in items]
    
    # Init outputs
    p_ind, p_grp, p_soc = [0]*len(items), [0]*len(items), [0]*len(items)
    batch_attacks = [""] * len(items)

    if "tokenizer" in resources and "model_1" in resources:
        tokenizer = resources["tokenizer"]
        model1 = resources["model_1"]
        
        # Max length 100 là đủ
        inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=100)
        input_ids = inputs["input_ids"].to(device)
        attn_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            # --- CHẠY 1 LẦN DUY NHẤT CHO CẢ 2 MODEL ---
            # Model 1 trả về cả kết quả phân loại lẫn embedding
            o1, o2, o3, cls_embeddings = model1(input_ids, attn_mask)
            
            p_ind = torch.argmax(o1, dim=1).cpu().numpy()
            p_grp = torch.argmax(o2, dim=1).cpu().numpy()
            p_soc = torch.argmax(o3, dim=1).cpu().numpy()

            # --- MODEL 2 CHẠY KÉ (CỰC NHANH) ---
            if "model_2" in resources:
                model2 = resources["model_2"]
                
                # Tìm các câu Toxic để chạy tiếp Model 2
                hate_indices = [i for i in range(len(items)) if p_ind[i]>=2 or p_grp[i]>=2 or p_soc[i]>=2]
                
                if hate_indices:
                    # Lấy embedding của các câu hate (không cần tính toán gì thêm)
                    hate_embeddings = cls_embeddings[hate_indices] # Slice tensor
                    
                    # Chạy qua mạng MLP nhẹ hều
                    head_outputs = model2(hate_embeddings)
                    probs = torch.sigmoid(head_outputs).cpu().numpy()
                    
                    # Xử lý kết quả
                    for idx_in_batch, orig_idx in enumerate(hate_indices):
                        row_probs = probs[idx_in_batch]
                        bin_list = ["1" if p > 0.3 else "0" for p in row_probs] # Threshold 0.3
                        
                        # Pad/Cut cho đủ 19
                        if len(bin_list) > 19: bin_list = bin_list[:19]
                        elif len(bin_list) < 19: bin_list.extend(["0"]*(19-len(bin_list)))
                        
                        batch_attacks[orig_idx] = "".join(bin_list)

    results = []
    for i in range(len(items)):
        results.append({
            "id": ids[i],
            "text": texts[i],
            "targets": [int(p_ind[i]), int(p_grp[i]), int(p_soc[i])],
            "type_attack_binary": batch_attacks[i]
        })
    
    print(f"⚡ Combined Batch {len(items)}: {time.time() - start_time:.4f}s")
    return {"results": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)