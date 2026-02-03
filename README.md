# SE363 - Big Data Streaming Demo

Dự án Demo môn học Big Data (SE363) xây dựng pipeline xử lý dữ liệu thời gian thực (Real-time Streaming) phát hiện ngôn từ thù ghét (Hate Speech).

## Kiến trúc hệ thống

Dự án được chia thành 2 phần chính hoạt động song song:

* **Infrastructure (Docker):** Chứa các dịch vụ hạ tầng gồm Apache Kafka, Zookeeper, MongoDB, Apache Spark (Master & Worker).
* **Application (Local):** Chứa mã nguồn ứng dụng chạy trên máy host gồm Producer giả lập dữ liệu, Model Server (AI), và Dashboard giám sát.

## Công nghệ sử dụng

* **Message Queue:** Apache Kafka
* **Processing Engine:** Apache Spark Structured Streaming
* **Storage:** MongoDB
* **Model Serving:** FastAPI
* **Visualization:** Streamlit
* **Containerization:** Docker & Docker Compose

## Cài đặt và Chuẩn bị

Trước khi chạy hệ thống, hãy đảm bảo máy tính đã cài đặt:
* Docker Desktop
* Python 3.8 trở lên

### 1. Khởi động Hạ tầng (Docker)

Bước này sẽ khởi chạy các container Kafka, MongoDB và Spark Cluster.

```bash
docker-compose up -d --build
```

### 2. Thiết lập Môi trường ảo (Local)

Vì các file ứng dụng chạy trên máy thật (Host), bạn cần cài đặt thư viện Python cho chúng.

```bash
# Tạo môi trường ảo (nếu chưa có)
python -m venv venv

# Kích hoạt môi trường ảo
# Đối với Windows:
.\venv\Scripts\activate
# Đối với Linux/MacOS:
source venv/bin/activate

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### Hướng dẫn chạy hệ thống
Để hệ thống hoạt động trơn tru, hãy mở 4 Terminal khác nhau và thực hiện lần lượt theo thứ tự sau:

#### Terminal 1: Chạy Model Server
Server này cung cấp API để Spark gọi sang dự đoán nhãn (Toxic/Clean).


```bash
# Đảm bảo đã kích hoạt venv
uvicorn model_server:app --host 0.0.0.0 --port 8000 --reload
```

Server sẽ chạy tại: http://localhost:8000

#### Terminal 2: Submit Spark Job (Docker)
Submit job vào Spark Container để bắt đầu lắng nghe dữ liệu từ Kafka, xử lý qua Model Server và ghi xuống MongoDB.


```bash
docker exec -it spark-master /opt/spark/bin/spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1,org.mongodb.spark:mongo-spark-connector_2.12:10.2.1 \
  /app/code/spark_streaming.py
  
```
#### Terminal 3: Chạy Dashboard (Local)
Giao diện giám sát dữ liệu và hiển thị cảnh báo theo thời gian thực.
```bash
# Đảm bảo đã kích hoạt venv
streamlit run dashboard.py
```
Dashboard sẽ tự động mở trên trình duyệt (thường là http://localhost:8501)

#### Terminal 4: Chạy Producer (Local)
Bắt đầu bắn dữ liệu giả lập vào Kafka để hệ thống xử lý.
```bash
# Đảm bảo đã kích hoạt venv
python producer.py
```
```bash
SE363-demo/
├── Dockerfile
├── docker-compose.yml           # Cấu hình hạ tầng Kafka, Spark, Mongo
├── requirements.txt             # Danh sách thư viện Python cho Local
├── spark_code/
│   └── spark_streaming.py       # Code xử lý chính của Spark
├── model_server.py              # API Server (FastAPI)
├── producer.py                  # Script giả lập gửi tin nhắn vào Kafka
├── dashboard.py                 # Giao diện giám sát (Streamlit)
├── chat/
│   └── demo.xlsx                # Dữ liệu mẫu đầu vào
└── teencode.xlsx                # Từ điển hỗ trợ xử lý text
```