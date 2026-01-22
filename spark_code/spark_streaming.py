from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StringType, StructType, ArrayType, BooleanType
import requests
import json

# Cấu hình
API_URL = "http://host.docker.internal:8000/predict_batch"
MONGO_URI = "mongodb://mongodb:27017"

# Maps
LEVEL_MAP = {0: "Normal", 1: "Clean", 2: "Offensive", 3: "Hate"}
TYPE_ATTACK_LABELS = [
    "Threat", "Scam", "Misinformation", "Boycott",
    "Body Shaming", "Sexual Harassment", "Intelligence", "Moral", "Victim Blaming",
    "Gender", "Regionalism", "Racism", "Classism", "Religion",
    "Politics", "Social Issues", "Product", "Community",
    "Other"
]

# Schema đầu ra (chỉ khai báo 1 lần)
output_schema = StructType() \
    .add("id", StringType()) \
    .add("cmt", StringType()) \
    .add("Individual", StringType()) \
    .add("Group", StringType()) \
    .add("Societal", StringType()) \
    .add("type_attack", ArrayType(StringType())) \
    .add("is_hate", BooleanType())

def process_partition(iterator):
    rows = list(iterator)
    if not rows: return iter([])

    batch_payload = [{"id": row.id, "text": row.cmt} for row in rows]
    results = []
    
    # Chia nhỏ batch 16 items để gửi API
    chunk_size = 16
    for i in range(0, len(batch_payload), chunk_size):
        chunk = batch_payload[i:i+chunk_size]
        try:
            resp = requests.post(API_URL, json={"batch": chunk}, timeout=15)
            if resp.status_code == 200:
                api_res = resp.json().get("results", [])
                for res in api_res:
                    t_lvls = res["targets"]
                    is_hate = any(l == 3 for l in t_lvls)
                    
                    attacks = []
                    if is_hate and res["type_attack_binary"]:
                        for idx, char in enumerate(res["type_attack_binary"]):
                            if char == '1' and idx < len(TYPE_ATTACK_LABELS):
                                attacks.append(TYPE_ATTACK_LABELS[idx])

                    results.append({
                        "id": res["id"],
                        "cmt": res["text"],
                        "Individual": LEVEL_MAP.get(t_lvls[0], "Unknown"),
                        "Group": LEVEL_MAP.get(t_lvls[1], "Unknown"),
                        "Societal": LEVEL_MAP.get(t_lvls[2], "Unknown"),
                        "type_attack": attacks, 
                        "is_hate": is_hate     
                    })
            else:
                print(f"API Error {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"Error calling API: {e}")
            
    return iter(results)

def main():
    spark = SparkSession.builder.appName("ToxicBatch").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # Đọc từ Kafka
    df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "kafka:9092") \
        .option("subscribe", "demo_topic") \
        .option("startingOffsets", "latest") \
        .option("failOnDataLoss", "false") \
        .load()

    schema = StructType().add("id", StringType()).add("cmt", StringType())
    parsed = df.select(from_json(col("value").cast("string"), schema).alias("data")).select("data.*")

    def write_mongo(batch_df, batch_id):
        if batch_df.isEmpty(): return
        
        # Map partition
        res_rdd = batch_df.rdd.mapPartitions(process_partition)
        
        if not res_rdd.isEmpty():
            # Tạo DF với Schema tường minh để tránh lỗi CANNOT_DETERMINE_TYPE
            final_df = spark.createDataFrame(res_rdd, schema=output_schema)
            
            final_df.write \
                .format("mongodb") \
                .mode("append") \
                .option("connection.uri", MONGO_URI) \
                .option("database", "toxic_db") \
                .option("collection", "monitor_logs") \
                .save()
            print(f">>> Batch {batch_id} Saved.")

    query = parsed.writeStream.foreachBatch(write_mongo).start()
    query.awaitTermination()

if __name__ == "__main__":
    main()