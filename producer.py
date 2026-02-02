import pandas as pd
from kafka import KafkaProducer
import json
import time
import uuid

# --- CẤU HÌNH ---
KAFKA_BROKER = 'localhost:29092'
TOPIC_NAME = 'demo_topic'
DATA_FILE = 'chat/demo.xlsx'

def json_serializer(data):
    return json.dumps(data).encode('utf-8')

# 1. Khởi tạo Producer
try:
    producer = KafkaProducer(
        bootstrap_servers=[KAFKA_BROKER],
        value_serializer=json_serializer,
        api_version=(0, 10, 1)
    )
    print(f" Connected to Kafka at {KAFKA_BROKER}")
except Exception as e:
    print(f" Failed to connect to Kafka: {e}")
    exit()

# 2. Đọc Data
try:
    df = pd.read_excel(DATA_FILE) 
    print(f"📂 Loaded {len(df)} comments")
except Exception as e:
    print(f"⚠️ Load file error, using dummy data.")
    df = pd.DataFrame({'cmt': ['Test comment'] * 1000})

# 3. XÁO TRỘN DỮ LIỆU 
df = df.sample(frac=1).reset_index(drop=True)
print("Data has been randomized!")

# 4. Gửi tin với kịch bản BURST
print(" Starting Stream...")
start_time_stream = time.time()

try:
    for index, row in df.iterrows():
        comment_text = str(row.get('cmt', row.get('cmt_processed', 'No content')))
        short_id = str(uuid.uuid4())[:8]
        
        message = {
            "id": short_id,
            "cmt": comment_text,
            "timestamp": time.time()
        }
        
        producer.send(TOPIC_NAME, value=message)
        
        # --- LOGIC ĐIỀU CHỈNH TỐC ĐỘ (SCENARIO) ---
        elapsed = time.time() - start_time_stream
        
        # Kịch bản:
        # 0s - 20s: Bình thường (Delay 0.1s ~ 10 tin/s)
        # 20s - 30s: TẤN CÔNG (Delay 0.005s ~ 200 tin/s)
        # > 30s   : Bình thường lại
        
        if 20 <= elapsed <= 30:
            delay = 0.005  
            status = "ATTACK"
        else:
            delay = 0.1    
            status = "NORMAL"

        if index % 10 == 0: # Chỉ in mỗi 10 tin để đỡ spam console
            print(f"[{status}] Time: {elapsed:.1f}s | Sent {index} msg")
        
        time.sleep(delay)
        
except KeyboardInterrupt:
    print("\nStopped by user.")
finally:
    producer.close()