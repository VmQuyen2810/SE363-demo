# Dùng tag ngắn gọn, chính xác của Apache Spark
FROM apache/spark:3.4.1

# Chuyển sang quyền root để cài thư viện
USER root

# Cài đặt thư viện requests (quan trọng)
RUN pip install requests pymongo datetime pyvi openpyxl pandas 

# Thiết lập thư mục làm việc
WORKDIR /app

# (Tùy chọn) Có thể quay về user mặc định của Spark nếu muốn an toàn, 
# nhưng để root cho dễ chạy demo tránh lỗi permission
USER root