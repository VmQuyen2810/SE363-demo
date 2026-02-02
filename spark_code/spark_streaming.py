from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StringType, StructType
import requests
import json
import pymongo
import datetime
import re
import pandas as pd # Import pandas để đọc file
import os
from pyvi import ViTokenizer

# --- CẤU HÌNH ---
API_URL = "http://host.docker.internal:8000/predict_batch"
MONGO_URI = "mongodb://mongodb:27017/"
DB_NAME = "hatespeech_db"
COL_NAME = "monitor_logs"
TEENCODE_PATH = "/app/teencode.xlsx" # Đường dẫn file trong Docker 

def init_system_resources():
    """
    Hàm này chạy 1 lần duy nhất khi Spark Driver khởi động.
    Nhiệm vụ: Tạo Index cho MongoDB để Dashboard query nhanh.
    """
    print("Initializing System Resources...")
    try:
        # Kết nối Mongo
        client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db = client[DB_NAME]
        col = db[COL_NAME]
        
        # 1. Tạo Index cho timestamp giảm dần 
        col.create_index([("timestamp", pymongo.DESCENDING)], background=True)
        print("✅ MongoDB Index created: timestamp_desc")
        
        # 2. Tạo Index cho id để tránh trùng lặp
        col.create_index([("id", pymongo.ASCENDING)], unique=True, background=True)
        print("✅ MongoDB Index created: id_unique")
        
        client.close()
    except Exception as e:
        print(f"⚠️ Warning: Could not init MongoDB indexes. Error: {e}")

# --- BIẾN TOÀN CỤC ---
GLOBAL_TEENCODE_DICT = None 

# 1. Stopwords
VIETNAMESE_STOPWORDS = set([
    "bị", "bởi", "cả", "các", "cái", "cần", "càng", "chỉ", "chiếc", "cho", "chứ", "chưa", "chuyện",
    "có", "có_thể", "cứ", "của", "cùng", "cũng", "đã", "đang", "đây", "để", "đến_nỗi", "đều", "điều",
    "do", "đó", "được", "dưới", "gì", "khi", "là", "lại", "lên", "lúc", "mà", "mỗi", "một_cách",
    "này", "nên", "nếu", "ngay", "nhiều", "như", "nhưng", "những", "nơi", "nữa", "phải", "qua", "ra",
    "rằng", "rất", "rồi", "sau", "sẽ", "so", "sự", "tại", "theo", "thì", "trên", "trước", "từ", "từng",
    "và", "vẫn", "vào", "vậy", "vì", "việc", "với", "vừa"
])

TYPE_ATTACK_LABELS = [
    "Threat", "Scam", "Misinformation", "Boycott", "Body Shaming", "Sexual Harassment",
    "Intelligence", "Moral", "Victim Blaming", "Gender", "Regionalism", "Racism",
    "Classism", "Religion", "Politics", "Social Issues", "Product", "Community", "Other"
]

# --- HÀM LOAD TEENCODE ---
def get_teencode_dict():
    global GLOBAL_TEENCODE_DICT
    if GLOBAL_TEENCODE_DICT is None:
        try:
            if os.path.exists(TEENCODE_PATH):
                df = pd.read_excel(TEENCODE_PATH) 
                # Convert thành dictionary 
                GLOBAL_TEENCODE_DICT = dict(zip(df.iloc[:, 0].astype(str), df.iloc[:, 1].astype(str)))
                print("Loaded Teencode File Successfully!")
            else:
                print("Teencode file not found, using empty dict.")
                GLOBAL_TEENCODE_DICT = {}
        except Exception as e:
            print(f"Error loading teencode: {e}")
            GLOBAL_TEENCODE_DICT = {}
    return GLOBAL_TEENCODE_DICT

# --- HÀM TIỀN XỬ LÝ ---
def clean_and_normalize(text):
    if not text: return ""
    text = str(text)

    # 1. Xoá tên tác giả
    text = re.sub(r'^.*?\n', '', text, count=1)

    # 2. Xoá URL, Email, Mentions, Ký tự đặc biệt
    pattern = r'(https?://\S+|www\.\S+)|(@\w+)|(\S+@\S+)|([^\w\s])'
    text = re.sub(pattern, ' ', text)

    # 3. Chuẩn hoá
    text = text.replace('\n', '. ').strip().lower()
    text = re.sub(r'\s+', ' ', text)

    # 4. Xử lý Teencode (Dùng dict đã load từ file)
    teencode_dict = get_teencode_dict()
    words = text.split()
    fixed_words = [teencode_dict.get(w, w) for w in words]
    text = " ".join(fixed_words)

    # 5. Tokenize (PyVi)
    text = ViTokenizer.tokenize(text)

    # 6. Stopwords removal
    final_words = [w for w in text.split() if w not in VIETNAMESE_STOPWORDS]
    
    return " ".join(final_words)

# --- LOGIC XỬ LÝ PARTITION ---
def process_partition(iterator):
    try:
        client = pymongo.MongoClient(MONGO_URI)
        db = client[DB_NAME]
        col_mongo = db[COL_NAME]
    except Exception as e:
        print(f" Mongo Connection Error: {e}")
        return iter([])

    rows = list(iterator)
    if not rows: return iter([])

    batch_payload = []
    # Map lưu thông tin để join lại sau khi có kết quả API
    data_map = {} 

    for row in rows:
        # Tạo cột cmt_processed
        cmt_processed = clean_and_normalize(row.cmt)
        
        # Lưu vào map để lát nữa ghi vào DB
        data_map[row.id] = {
            "cmt_raw": row.cmt,
            "cmt_processed": cmt_processed
        }
        
        if cmt_processed.strip():
            # API nhận vào text đã xử lý
            batch_payload.append({"id": row.id, "text": cmt_processed})

    if not batch_payload: return iter([])

    # Gửi API theo batch
    chunk_size = 128
    for i in range(0, len(batch_payload), chunk_size):
        chunk = batch_payload[i:i+chunk_size]
        try:
            # Gửi cmt_processed lên API
            resp = requests.post(API_URL, json={"batch": chunk}, timeout=60)
            
            if resp.status_code == 200:
                api_res = resp.json().get("results", [])
                mongo_docs = []
                current_time = datetime.datetime.utcnow()
                
                for res in api_res:
                    res_id = res["id"]
                    
                    # Lấy lại thông tin gốc từ map
                    original_info = data_map.get(res_id, {"cmt_raw": "", "cmt_processed": ""})
                    
                    t_lvls = res["targets"]
                    is_hate = any(l >= 2 for l in t_lvls)
                    
                    attacks = []
                    if is_hate and res["type_attack_binary"]:
                        for idx, char in enumerate(res["type_attack_binary"]):
                            if char == '1' and idx < len(TYPE_ATTACK_LABELS):
                                attacks.append(TYPE_ATTACK_LABELS[idx])
                    
                    lvl_map = {0: "Normal", 1: "Clean", 2: "Offensive", 3: "Hate"}
                    
                    doc = {
                        "id": res_id,
                        # Cột 'cmt': Là dữ liệu gốc (để hiển thị trên Dashboard cho dễ đọc)
                        "cmt": original_info["cmt_raw"],  
                        # Cột 'cmt_processed': Là dữ liệu đã sạch (để debug hoặc retrain sau này)
                        "cmt_processed": original_info["cmt_processed"], 
                        
                        "Individual": lvl_map.get(t_lvls[0], "Unknown"),
                        "Group": lvl_map.get(t_lvls[1], "Unknown"),
                        "Societal": lvl_map.get(t_lvls[2], "Unknown"),
                        "type_attack": attacks,
                        "is_hate": is_hate,
                        "timestamp": current_time
                    }
                    mongo_docs.append(doc)
                
                if mongo_docs:
                    col_mongo.insert_many(mongo_docs)
            else:
                print(f"API Error {resp.status_code}")
                
        except Exception as e:
            print(f"Error processing chunk: {e}")
            
    client.close()
    return iter([])

def main():
    init_system_resources()

    spark = SparkSession.builder \
        .appName("ToxicStreamRealTime") \
        .config("spark.executor.instances", "2") \
        .config("spark.executor.cores", "2") \
        .config("spark.sql.shuffle.partitions", "4") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    df = (
        spark.readStream 
            .format("kafka") 
            .option("kafka.bootstrap.servers", "kafka:9092") 
            .option("subscribe", "demo_topic") 
            .option("startingOffsets", "latest") 
            .option("failOnDataLoss", "false") 
            .option("maxOffsetsPerTrigger", 1000)  
            .load()
    )

    schema = StructType().add("id", StringType()).add("cmt", StringType())
    parsed = df.select(from_json(col("value").cast("string"), schema).alias("data")).select("data.*")
    parsed = parsed.repartition(4)

    def process_batch_wrapper(batch_df, batch_id):
        batch_df.rdd.mapPartitions(process_partition).count()
        print(f">>> Batch {batch_id} Finished.")

    query = parsed.writeStream \
        .foreachBatch(process_batch_wrapper) \
        .trigger(processingTime='1 seconds') \
        .start()

    query.awaitTermination()

if __name__ == "__main__":
    main()