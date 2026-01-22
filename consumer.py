from kafka import KafkaConsumer
import json

# --- CẤU HÌNH ---
KAFKA_BROKER = 'localhost:29092'
TOPIC_NAME = 'demo_topic'
GROUP_ID = 'demo_consumer_group'

def json_deserializer(data):
    return json.loads(data.decode('utf-8'))

# Khởi tạo Consumer
try:
    consumer = KafkaConsumer(
        TOPIC_NAME,
        bootstrap_servers=[KAFKA_BROKER],
        group_id=GROUP_ID,
        auto_offset_reset='earliest',   # đọc từ đầu topic
        enable_auto_commit=True,
        value_deserializer=json_deserializer
    )
    print(f"✅ Connected to Kafka at {KAFKA_BROKER}, listening topic '{TOPIC_NAME}'")
except Exception as e:
    print(f"❌ Failed to connect to Kafka: {e}")
    exit()

print("👂 Waiting for messages...\n")

try:
    for msg in consumer:
        value = msg.value
        print("📥 Received message:")
        print(f"  - partition: {msg.partition}")
        print(f"  - offset   : {msg.offset}")
        print(f"  - id       : {value.get('id')}")
        print(f"  - cmt      : {value.get('cmt')}")
        print(f"  - timestamp: {value.get('timestamp')}")
        print("-" * 60)

except KeyboardInterrupt:
    print("\n🛑 Consumer stopped by user.")
finally:
    consumer.close()
