import streamlit as st
import pymongo
import pandas as pd
import time
from bson.objectid import ObjectId
import datetime
import altair as alt

st.set_page_config(page_title="🛡️ BigData Toxic Monitor", layout="wide")

# --- 1. KẾT NỐI MONGODB ---
@st.cache_resource
def init_connection():
    return pymongo.MongoClient("mongodb://localhost:27017/")

try:
    client = init_connection()
    db = client["toxic_db"]
    col = db["monitor_logs"]
except Exception as e:
    st.error(f"Lỗi kết nối MongoDB: {e}")
    st.stop()

# --- 2. GLOBAL STATE (Lưu trữ dữ liệu toàn cục - Không mất khi F5) ---
# Dùng class để bọc dữ liệu, giúp Streamlit cache object này lại
class DataManager:
    def __init__(self):
        self.df = pd.DataFrame()
        self.last_id = None
        # Lấy mốc ID hiện tại khi khởi động server lần đầu
        last_doc = col.find_one(sort=[("_id", -1)])
        if last_doc:
            self.last_id = last_doc['_id']
        else:
            self.last_id = ObjectId.from_datetime(datetime.datetime.now())
        self.start_time = datetime.datetime.now()

    def update_data(self):
        # Chỉ query những dòng mới hơn last_id đã lưu
        query = {"_id": {"$gt": self.last_id}}
        cursor = col.find(query).sort("_id", 1) # Lấy cũ -> mới để append đúng thứ tự
        new_items = list(cursor)
        
        if new_items:
            # Cập nhật ID mới nhất
            self.last_id = new_items[-1]['_id']
            
            # Tạo DF mới
            new_df = pd.DataFrame(new_items)
            new_df['fetched_at'] = datetime.datetime.now()
            
            # Gộp vào DF tổng
            self.df = pd.concat([self.df, new_df], ignore_index=True)
            return True # Có dữ liệu mới
        return False

# @st.cache_resource đảm bảo object này chỉ tạo 1 lần duy nhất khi chạy `streamlit run`
# Tất cả các tab trình duyệt sẽ dùng chung object này -> Dữ liệu đồng bộ
@st.cache_resource
def get_manager():
    return DataManager()

manager = get_manager()

# --- 3. GIAO DIỆN ---
st.title("🛡️ Hệ Thống Giám Sát Real-time (Global Mode)")
st.caption(f"Server khởi động lúc: {manager.start_time.strftime('%H:%M:%S %d/%m/%Y')}")

# Cập nhật dữ liệu
has_new_data = manager.update_data()

# Lấy copy dữ liệu để hiển thị
df = manager.df.copy()

# Sắp xếp mới nhất lên đầu để hiển thị
if not df.empty:
    df = df.sort_values(by='fetched_at', ascending=False)

# --- 4. HIỂN THỊ METRICS ---
placeholder = st.empty()

with placeholder.container():
    if not df.empty:
        # A. THỐNG KÊ
        total = len(df)
        hate_df = df[df['is_hate'] == True]
        hate_count = len(hate_df)
        clean_count = total - hate_count
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tổng tin (Session)", total)
        c2.metric("Phát hiện HATE", hate_count, delta_color="inverse")
        c3.metric("Tin Sạch", clean_count)
        
        # B. CẢNH BÁO (5 phút gần nhất)
        now = datetime.datetime.now()
        recent_df = df[df['fetched_at'] > (now - datetime.timedelta(minutes=5))]
        recent_hate = len(recent_df[recent_df['is_hate'] == True])
        
        if recent_hate > 10:
            st.error(f"🚨 BÁO ĐỘNG: {recent_hate} tin độc hại trong 5 phút qua!")
        
        # C. BIỂU ĐỒ & BẢNG
        tab1, tab2 = st.tabs(["📊 Biểu đồ", "📋 Log chi tiết"])
        
        with tab1:
            if 'type_attack' in df.columns:
                # Explode list type_attack ra để đếm
                attacks = df[df['is_hate']==True]['type_attack'].explode().dropna()
                if not attacks.empty:
                    stats = attacks.value_counts().reset_index()
                    stats.columns = ['Type', 'Count']
                    
                    chart = alt.Chart(stats).mark_bar().encode(
                        x='Count', y=alt.Y('Type', sort='-x'), color='Type'
                    )
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info("Chưa có loại tấn công cụ thể.")
        
        with tab2:
            # Highlight
            def style_row(row):
                return ['background-color: #ffcccc'] * len(row) if row['is_hate'] else [''] * len(row)

            cols = ['cmt', 'Individual', 'Group', 'Societal', 'type_attack', 'is_hate']
            valid_cols = [c for c in cols if c in df.columns]
            
            st.dataframe(
                df[valid_cols].style.apply(style_row, axis=1),
                column_config={
                    "type_attack": st.column_config.ListColumn("Loại"),
                    "is_hate": st.column_config.CheckboxColumn("Toxic?", disabled=True)
                },
                use_container_width=True,
                height=600
            )
            
    else:
        st.info("⏳ Đang chờ dữ liệu đầu tiên...")

# Tự động refresh sau 1 giây
time.sleep(1)
st.rerun()