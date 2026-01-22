import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import onnxruntime as ort
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn
import os
import time

# ==============================================================================
# 1. CẤU HÌNH & LABEL MAP
# ==============================================================================
TYPE_ATTACK_LABELS = [
    "Threat", "Scam", "Misinformation", "Boycott",
    "Body Shaming", "Sexual Harassment", "Intelligence", "Moral", "Victim Blaming",
    "Gender", "Regionalism", "Racism", "Classism", "Religion",
    "Politics", "Social Issues", "Product", "Community",
    "Other"
]

class Config:
    BERT_NAME = "local_phobert" 
    HIDDEN_SIZE = 768
    NUM_CLASSES = 4 
    GRU_HIDDEN = 128
    CNN_FILTERS = 64
    CNN_KERNEL_SIZES = [2, 3, 4]

# ==============================================================================
# 2. MODEL ARCHITECTURE (ViTHSD)
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
        
        return self.fc_individual(x), self.fc_group(x), self.fc_societal(x)

# ==============================================================================
# 3. SERVER SETUP
# ==============================================================================
app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resources = {}

@app.on_event("startup")
def load_resources():
    print(f">>> Server running on: {device}")
    try:
        # 1. Tokenizer
        if os.path.exists("local_phobert"):
            resources["tokenizer_1"] = AutoTokenizer.from_pretrained("local_phobert", local_files_only=True)
        else:
            resources["tokenizer_1"] = AutoTokenizer.from_pretrained("vinai/phobert-base")

        # 2. Model ViTHSD
        model = PhoBertHybridModel(Config())
        if os.path.exists("best_model.pth"):
            state_dict = torch.load("best_model.pth", map_location=device)
            # Fix key 'module.'
            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            try:
                model.load_state_dict(new_state_dict, strict=True)
                print("✅ ViTHSD Loaded Successfully (Strict Mode)")
            except Exception as e:
                print(f"❌ WARNING: Loading with strict=False. Error: {e}")
                model.load_state_dict(new_state_dict, strict=False)
                
            model.to(device)
            model.eval()
            resources["model_1"] = model
        
        # 3. Model ONNX
        if os.path.exists("type_attack"):
            resources["tokenizer_2"] = AutoTokenizer.from_pretrained("type_attack", local_files_only=True)
            onnx_path = "type_attack/model.onnx"
            if os.path.exists(onnx_path):
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
                resources["ort_session"] = ort.InferenceSession(onnx_path, providers=providers)
                print("✅ TypeAttack ONNX Loaded.")

    except Exception as e:
        print(f"❌ Critical Error: {e}")

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
    
    p_ind = [0] * len(items)
    p_grp = [0] * len(items)
    p_soc = [0] * len(items)
    batch_attacks = [""] * len(items)

    # --- PHASE 1: ViTHSD (Batch Processing) ---
    if "model_1" in resources:
        tok1 = resources["tokenizer_1"]
        mod1 = resources["model_1"]
        
        inputs = tok1(texts, return_tensors="pt", truncation=True, padding=True, max_length=128)
        input_ids = inputs["input_ids"].to(device)
        attn_mask = inputs["attention_mask"].to(device)
        
        with torch.no_grad():
            o1, o2, o3 = mod1(input_ids, attn_mask)
            p_ind = torch.argmax(o1, dim=1).cpu().numpy()
            p_grp = torch.argmax(o2, dim=1).cpu().numpy()
            p_soc = torch.argmax(o3, dim=1).cpu().numpy()
            
            # Log kiểm tra (Chỉ in 1 dòng đại diện cho cả Batch)
            print(f"🔹 Processed Batch {len(items)} items. Example: '{texts[0][:20]}...' -> [{p_ind[0]}, {p_grp[0]}, {p_soc[0]}]")

    # --- PHASE 2: Type Attack (Optimized Loop) ---
    # Lấy index các comment Toxic (>=2)
    hate_indices = []
    for i in range(len(items)):
        if p_ind[i] >= 2 or p_grp[i] >= 2 or p_soc[i] >= 2:
            hate_indices.append(i)

    if hate_indices and "ort_session" in resources:
        hate_texts = [texts[i] for i in hate_indices]
        tok2 = resources["tokenizer_2"]
        sess = resources["ort_session"]
        
        # TỐI ƯU: Tokenize 1 lần cho cả batch hate (Nhanh hơn tokenize trong vòng lặp)
        inputs2_batch = tok2(hate_texts, return_tensors="np", truncation=True, padding='max_length', max_length=128)
        
        # Vòng lặp chỉ chạy inference (nhẹ hơn)
        for idx_in_hate_list, orig_idx in enumerate(hate_indices):
            try:
                # Cắt input từ batch lớn ra
                input_ids_single = inputs2_batch["input_ids"][idx_in_hate_list:idx_in_hate_list+1]
                attn_mask_single = inputs2_batch["attention_mask"][idx_in_hate_list:idx_in_hate_list+1]
                
                ort_inputs = {
                    sess.get_inputs()[0].name: input_ids_single.astype(np.int64),
                    sess.get_inputs()[1].name: attn_mask_single.astype(np.int64)
                }
                
                # Inference
                logits = sess.run(None, ort_inputs)[0] 
                probs = 1 / (1 + np.exp(-logits))
                
                # Xử lý kết quả
                bin_list = ["1" if p > 0.3 else "0" for p in probs[0]]
                if len(bin_list) > 19: bin_list = bin_list[:19]
                elif len(bin_list) < 19: bin_list.extend(["0"]*(19-len(bin_list)))
                
                batch_attacks[orig_idx] = "".join(bin_list)
                
            except Exception as e:
                print(f"⚠️ ONNX Error at item {orig_idx}: {e}")

    # Build Response
    results = []
    for i in range(len(items)):
        results.append({
            "id": ids[i],
            "text": texts[i],
            "targets": [int(p_ind[i]), int(p_grp[i]), int(p_soc[i])],
            "type_attack_binary": batch_attacks[i]
        })
    
    # In thời gian xử lý để bạn yên tâm
    print(f"⏱️ Batch finished in {time.time() - start_time:.4f}s")
    
    return {"results": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)