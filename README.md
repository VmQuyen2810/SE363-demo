# SE363-demo

Demo dự án Big Data sử dụng **Apache Spark**, **Apache Kafka** và **MongoDB**.  
Mục tiêu: xây dựng pipeline xử lý dữ liệu streaming với Spark Structured Streaming.

---

## Mục tiêu

- Xây dựng pipeline ingest dữ liệu từ Kafka  
- Xử lý dữ liệu streaming bằng Spark  
- Ghi kết quả xuống MongoDB  
- Hỗ trợ chạy bằng Docker hoặc Local environment  

---

## Cài đặt môi trường với Docker

### 1. Khởi chạy toàn bộ container

```bash
docker-compose up -d --build
```

### 2. Cài đặt môi trường local 

```bash
pip install -r requirements.txt
```

### 3. Chạy Spark Streaming job

```bash
docker exec -it spark-master /opt/spark/bin/spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1,org.mongodb.spark:mongo-spark-connector_2.12:10.2.1 /app/code/spark_streaming.py
```
### 4. Chạy server
```bash
uvicorn model_server:app --host 0.0.0.0 --port 8000 --reload
```
## 5. Chạy producer và dashboard

```bash
python producer.py
streamlit run dashboard.py
```
