import pandas as pd
from kafka import KafkaProducer
import json
import time
import uuid

# --- CẤU HÌNH ---
KAFKA_BROKER = 'localhost:29092' # Cổng External của Kafka
TOPIC_NAME = 'demo_topic'
DATA_FILE = 'chat/demo.xlsx'       # Tên file data của bạn (có cột 'cmt_processed' hoặc 'content')

def json_serializer(data):
    return json.dumps(data).encode('utf-8')

# Khởi tạo Producer
try:
    producer = KafkaProducer(
        bootstrap_servers=[KAFKA_BROKER],
        value_serializer=json_serializer
    )
    print(f"✅ Connected to Kafka at {KAFKA_BROKER}")
except Exception as e:
    print(f"❌ Failed to connect to Kafka: {e}")
    exit()

# Đọc Data
try:
    df = pd.read_excel(DATA_FILE) 
    print(f"📂 Loaded {len(df)} comments from {DATA_FILE}")
except Exception as e:
    print(f"⚠️ Could not load file '{DATA_FILE}'. Using dummy data instead.")

# Gửi tin
print("🚀 Starting Stream...")
try:
    for index, row in df.iterrows():
        # Lấy nội dung comment
        comment_text = str(row.get('cmt_processed', 'No content'))
        
        message = {
            "id": str(uuid.uuid4()),
            "cmt": comment_text,
            "timestamp": time.time()
        }
        
        producer.send(TOPIC_NAME, value=message)
        print(f"Sent [{index}]: {comment_text[:50]}...")
        
        time.sleep(0.1) # Gửi chậm lại (1.5s/tin) để kịp nhìn Demo
        
except KeyboardInterrupt:
    print("\n🛑 Stopped by user.")
finally:
    producer.close()