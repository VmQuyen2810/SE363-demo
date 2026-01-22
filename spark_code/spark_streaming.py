from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StringType, StructType, ArrayType, BooleanType
import requests
import json
import pymongo
import datetime # Cần cài thư viện này trên Docker: pip install pymongo

# --- CẤU HÌNH ---
API_URL = "http://host.docker.internal:8000/predict_batch"
# URI kết nối Mongo từ bên trong Docker Worker
MONGO_URI = "mongodb://mongodb:27017/" 
DB_NAME = "toxic_db"
COL_NAME = "monitor_logs"

# Maps
TYPE_ATTACK_LABELS = [
    "Threat", "Scam", "Misinformation", "Boycott",
    "Body Shaming", "Sexual Harassment", "Intelligence", "Moral", "Victim Blaming",
    "Gender", "Regionalism", "Racism", "Classism", "Religion",
    "Politics", "Social Issues", "Product", "Community",
    "Other"
]

def process_partition(iterator):
    try:
        client = pymongo.MongoClient(MONGO_URI)
        db = client[DB_NAME]
        col_mongo = db[COL_NAME]
    except Exception as e:
        print(f"❌ Mongo Connection Error: {e}")
        return iter([])

    rows = list(iterator)
    if not rows: return iter([])

    batch_payload = [{"id": row.id, "text": row.cmt} for row in rows]
    chunk_size = 16 
    
    for i in range(0, len(batch_payload), chunk_size):
        chunk = batch_payload[i:i+chunk_size]
        try:
            resp = requests.post(API_URL, json={"batch": chunk}, timeout=15)
            if resp.status_code == 200:
                api_res = resp.json().get("results", [])
                mongo_docs = []
                
                # Lấy thời gian hiện tại cho cả lô này
                current_time = datetime.datetime.utcnow() # <--- 2. LẤY GIỜ QUỐC TẾ
                
                for res in api_res:
                    t_lvls = res["targets"]
                    is_hate = any(l >= 2 for l in t_lvls)
                    
                    attacks = []
                    if is_hate and res["type_attack_binary"]:
                        for idx, char in enumerate(res["type_attack_binary"]):
                            if char == '1' and idx < len(TYPE_ATTACK_LABELS):
                                attacks.append(TYPE_ATTACK_LABELS[idx])
                    
                    lvl_map = {0: "Normal", 1: "Clean", 2: "Offensive", 3: "Hate"}
                    
                    doc = {
                        "id": res["id"],
                        "cmt": res["text"],
                        "Individual": lvl_map.get(t_lvls[0], "Unknown"),
                        "Group": lvl_map.get(t_lvls[1], "Unknown"),
                        "Societal": lvl_map.get(t_lvls[2], "Unknown"),
                        "type_attack": attacks,
                        "is_hate": is_hate,
                        "timestamp": current_time  # <--- 3. QUAN TRỌNG NHẤT: GHI GIỜ VÀO DB
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
    spark = SparkSession.builder \
        .appName("ToxicStreamRealTime") \
        .config("spark.executor.instances", "2") \
        .config("spark.executor.cores", "2") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("WARN")

    # Đọc từ Kafka
    df = (
        spark.readStream 
            .format("kafka") 
            .option("kafka.bootstrap.servers", "kafka:9092") 
            .option("subscribe", "demo_topic") 
            .option("startingOffsets", "latest") 
            .option("failOnDataLoss", "false") 
            .option("maxOffsetsPerTrigger", 50)  
            .load()
    )
# Chỉ lấy tối đa 50 tin mỗi lần (để xử lý nhanh < 1s)
    schema = StructType().add("id", StringType()).add("cmt", StringType())
    parsed = df.select(from_json(col("value").cast("string"), schema).alias("data")).select("data.*")

    # Repartition để tận dụng đa luồng (ví dụ máy có 4 core thì chia 4)
    parsed = parsed.repartition(4)

    # Dùng foreachBatch nhưng bên trong gọi mapPartitions để tự xử lý IO
    def process_batch_wrapper(batch_df, batch_id):
        # Action count() để kích hoạt mapPartitions chạy
        count = batch_df.rdd.mapPartitions(process_partition).count()
        print(f">>> Batch {batch_id} Finished. Processed segments.")

    # Trigger liên tục
    query = parsed.writeStream \
        .foreachBatch(process_batch_wrapper) \
        .trigger(processingTime='1 seconds') \
        .start()

    query.awaitTermination()

if __name__ == "__main__":
    main()