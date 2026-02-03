SE363 - Big Data Streaming Demo
Dự án Demo môn học Big Data (SE363) xây dựng pipeline xử lý dữ liệu thời gian thực phát hiện ngôn từ thù ghét .

Kiến trúc hệ thống:

Infrastructure (Docker): Apache Kafka, Zookeeper, MongoDB, Apache Spark (Master & Worker).

Application (Local): Producer giả lập dữ liệu, Model Server, và Dashboard giám sát.

 Công nghệ sử dụng
Message Queue: Apache Kafka

Processing Engine: Apache Spark Structured Streaming

Storage: MongoDB

Model Serving: FastAPI 

Visualization: Streamlit

Containerization: Docker & Docker Compose

 Cài đặt & Chuẩn bị
Trước khi chạy, hãy đảm bảo bạn đã cài đặt:

Docker Desktop

Python 3.8+

1. Khởi động Hạ tầng (Docker)
Bước này sẽ khởi chạy Kafka, MongoDB và Spark Cluster.

Bash
docker-compose up -d --build
Lưu ý: Đợi khoảng 30s-1p để các container (đặc biệt là Kafka và Spark) khởi động hoàn toàn.

2. Thiết lập Môi trường ảo (Local)
Vì các file producer.py, dashboard.py, và model_server.py chạy trên máy thật (Host), bạn cần cài đặt thư viện cho chúng.

Bash
# Tạo môi trường ảo (nếu chưa có)
python -m venv venv

# Kích hoạt môi trường ảo
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
 Hướng dẫn chạy hệ thống 
Để hệ thống hoạt động trơn tru, hãy mở 4 Terminal khác nhau và thực hiện lần lượt:

Terminal 1: Chạy Model Server (Local)
Server này cung cấp API để Spark gọi sang dự đoán nhãn (Toxic/Clean).

Bash
# Đảm bảo đã kích hoạt venv
# Windows: .\venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

uvicorn model_server:app --host 0.0.0.0 --port 8000 --reload
Server sẽ chạy tại: http://localhost:8000

Terminal 2: Submit Spark Job (Docker)
Submit job vào Spark Container để bắt đầu lắng nghe dữ liệu từ Kafka.

Bash
docker exec -it spark-master /opt/spark/bin/spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1,org.mongodb.spark:mongo-spark-connector_2.12:10.2.1 \
  /app/code/spark_streaming.py
Terminal 3: Chạy Dashboard (Local)
Giao diện giám sát dữ liệu và cảnh báo theo thời gian thực.

Bash
# Đảm bảo đã kích hoạt venv
streamlit run dashboard.py
Dashboard sẽ tự động mở trên trình duyệt (thường là http://localhost:8501)

Terminal 4: Chạy Producer (Local)
Bắt đầu bắn dữ liệu giả lập vào Kafka để hệ thống xử lý.

Bash
# Đảm bảo đã kích hoạt venv
python producer.py
📂 Cấu trúc thư mục
SE363-demo/
├── docker-compose.yml       # Cấu hình Kafka, Spark, Mongo
├── requirements.txt         # Thư viện Python cho Local
├── spark_code/spark_streaming.py       # Code xử lý chính 
├── model_server.py          # API Server
├── producer.py              # Giả lập gửi tin nhắn
├── dashboard.py             # Giao diện Streamlit
├── demo.xlsx                # Dữ liệu mẫu
