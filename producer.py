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
        value_serializer=json_serializer
    )
    print(f"✅ Connected to Kafka at {KAFKA_BROKER}")
except Exception as e:
    print(f"❌ Failed to connect to Kafka: {e}")
    exit()

# 2. Đọc Data
try:
    df = pd.read_excel(DATA_FILE) 
    print(f"📂 Loaded {len(df)} comments from {DATA_FILE}")
except Exception as e:
    print(f"⚠️ Could not load file '{DATA_FILE}'. Using dummy data instead.")
    # Tạo data giả nếu không có file
    df = pd.DataFrame({'cmt': ['Test comment'] * 1000})

# 3. XÁO TRỘN DỮ LIỆU (RANDOM)
# frac=1 nghĩa là lấy 100% dữ liệu nhưng xáo trộn ngẫu nhiên
df = df.sample(frac=1).reset_index(drop=True)
print("🔀 Data has been randomized!")

# 4. Gửi tin với logic Tăng Tốc
print("🚀 Starting Stream...")
start_time_stream = time.time()

try:
    for index, row in df.iterrows():
        comment_text = str(row.get('cmt', row.get('cmt_processed', 'No content')))
        
        # ID ngắn gọn (8 ký tự)
        short_id = str(uuid.uuid4())[:8]
        
        message = {
            "id": short_id,
            "cmt": comment_text,
            "timestamp": time.time()
        }
        
        producer.send(TOPIC_NAME, value=message)
        
        # --- LOGIC ĐIỀU CHỈNH TỐC ĐỘ ---
        elapsed = time.time() - start_time_stream
        
        if elapsed < 20:
            # Giai đoạn 1: Chạy chậm để demo (10 tin/giây)
            delay = 0.1
            status = "NORMAL"
        else:
            # Giai đoạn 2: Tăng tốc tối đa (200 tin/giây)
            delay = 0.005 
            status = "TURBO 🔥"

        print(f"[{status}] Sent {index} | ID={short_id} | Time={elapsed:.1f}s")
        
        time.sleep(delay)
        
except KeyboardInterrupt:
    print("\n🛑 Stopped by user.")
finally:
    producer.close()