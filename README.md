# ViHateStream: Real-time Vietnamese Hate Speech & Attack-Type Monitoring

This course demo project for Big Data (SE363) builds a real-time data streaming pipeline for Hate Speech detection.

## 📄 Research Paper & Abstract

*(Click on the image below to read the full research paper)*

[m23182-do paper.pdf](https://github.com/user-attachments/files/31167612/m23182-do.paper.pdf)
* <img width="514" height="584" alt="image" src="https://github.com/user-attachments/assets/eb1f4ef2-a04b-4b10-ad9c-6ed1390469ee" />

> **Note:** The paper provides in-depth details about the theoretical background, the AI model architecture for Hate Speech Detection, and the evaluation metrics applied in this streaming pipeline.

---

## System Architecture

The project is divided into two main components running in parallel:
...

* **Infrastructure (Docker): Contains the infrastructure services, including Apache Kafka, Zookeeper, MongoDB, and Apache Spark (Master & Worker).
* **Application (Local):** Contains the application source code running on the host machine, including a Data Producer, a Model Server (AI), and a Monitoring Dashboard.
## Technologies Used

* **Message Queue:** Apache Kafka
* **Processing Engine:** Apache Spark Structured Streaming
* **Storage:** MongoDB
* **Model Serving:** FastAPI
* **Visualization:** Streamlit
* **Containerization:** Docker & Docker Compose

## Prerequisites & Installation

Before running the system, please ensure you have the following installed on your machine:
* Docker Desktop
* Python 3.8

### 1. Start the Infrastructure (Docker)

This step will initialize the Kafka, MongoDB, and Spark Cluster containers.

```bash
docker-compose up -d --build
```

### 2. Set Up the Virtual Environment (Local)
Since the application scripts run on your host machine, you need to install the required Python libraries.

```bash
# Create a virtual environment (if you haven't already)
python -m venv venv

# Activate the virtual environment
# For Windows:
.\venv\Scripts\activate
# For Linux/MacOS:
source venv/bin/activate

# Install the required dependencies
pip install -r requirements.txt
```

### How to Run the System
For the system to operate smoothly, please open 4 separate terminals and execute the following steps in order:

#### Terminal 1: Run the Model Server
This server provides an API for Spark to call and predict labels (Toxic/Clean).


```bash
# Ensure your venv is activated
uvicorn model_server:app --host 0.0.0.0 --port 8000 --reload
```

The server will be running at: http://localhost:8000

#### Terminal 2: Submit the Spark Job (Docker)
Submit the job to the Spark Container to start consuming data from Kafka, processing it through the Model Server, and writing the results to MongoDB.


```bash
docker exec -it spark-master /opt/spark/bin/spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1,org.mongodb.spark:mongo-spark-connector_2.12:10.2.1 \
  /app/code/spark_streaming.py
  
```
#### Terminal 3: Run the Dashboard (Local)
Launch the interface for monitoring data and displaying real-time alerts.
```bash
# Ensure your venv is activated
streamlit run dashboard.py
```
The dashboard will automatically open in your browser (usually at http://localhost:8501).

#### Run the Producer (Local)
Start sending simulated data into Kafka for the system to process.
```bash
# Ensure your venv is activated
python producer.py
```
```bash
ViHateStream/
├── Dockerfile
├── docker-compose.yml           # Infrastructure config for Kafka, Spark, Mongo
├── requirements.txt             # Python dependencies for the local environment
├── spark_code/
│   └── spark_streaming.py       # Main data processing code for Spark
├── model_server.py              # API Server (FastAPI)
├── producer.py                  # Script to simulate sending messages to Kafka
├── dashboard.py                 # Monitoring Dashboard (Streamlit)
├── chat/
│   └── demo.xlsx                # Sample input data
└── teencode.xlsx                # Dictionary for text preprocessing
```

