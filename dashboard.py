import streamlit as st
import pandas as pd
import pymongo
import time
import plotly.express as px
import plotly.graph_objects as go
import uuid
from datetime import datetime, timedelta

# ==========================================
# 1. CẤU HÌNH & CSS
# ==========================================
st.set_page_config(page_title="Hệ thống Giám sát Toxic Real-time", layout="wide", page_icon="🛡️")

# CSS Fix lỗi hiển thị bảng
st.markdown("""
<style>
    /* Tùy chỉnh Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 24px;
    }
    /* Tô màu bảng */
    .stDataFrame {
        border: 1px solid #444; /* Viền tối cho hợp dark mode */
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# URI kết nối
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "hatespeech_db"
COL_NAME = "monitor_logs"

@st.cache_resource
def init_connection():
    try:
        client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        client.server_info()
        return client
    except Exception as e:
        st.error(f"❌ Lỗi kết nối MongoDB: {e}")
        return None

client = init_connection()

# ==========================================
# 2. QUẢN LÝ TRẠNG THÁI (SESSION STATE)
# ==========================================
if 'monitor_df' not in st.session_state:
    st.session_state['monitor_df'] = pd.DataFrame()
if 'last_fetch_time' not in st.session_state:
    st.session_state['last_fetch_time'] = datetime.utcnow() - timedelta(seconds=10)
if 'start_time' not in st.session_state:
    st.session_state['start_time'] = datetime.now()

# ==========================================
# 3. HÀM LẤY DỮ LIỆU
# ==========================================
def fetch_new_data():
    if not client: return pd.DataFrame()
    db = client[DB_NAME]
    col = db[COL_NAME]
    
    query = {"timestamp": {"$gt": st.session_state['last_fetch_time']}}
    cursor = col.find(query).sort("timestamp", 1)
    
    new_data = list(cursor)
    
    if new_data:
        df_new = pd.DataFrame(new_data)
        
        # Cập nhật thời gian lấy mới nhất
        max_time = df_new['timestamp'].max()
        if isinstance(max_time, str):
            max_time = pd.to_datetime(max_time)
        st.session_state['last_fetch_time'] = max_time
        
        if '_id' in df_new.columns:
            df_new['_id'] = df_new['_id'].astype(str)
            
        # UTC -> UTC+7
        if 'timestamp' in df_new.columns:
            df_new['timestamp'] = pd.to_datetime(df_new['timestamp']) + timedelta(hours=7)
            
        return df_new
    
    return pd.DataFrame()

# ==========================================
# 4. GIAO DIỆN CHÍNH
# ==========================================
c1, c2 = st.columns([3, 1])
with c1:
    st.title("🛡️ Trung tâm Giám sát Không gian mạng")
with c2:
    if st.button("🔄 Reset Phiên Giám sát"):
        st.session_state['monitor_df'] = pd.DataFrame()
        st.session_state['last_fetch_time'] = datetime.utcnow()
        st.session_state['start_time'] = datetime.now()
        st.rerun()

st.caption(f"🚀 Phiên giám sát bắt đầu lúc: {st.session_state['start_time'].strftime('%H:%M:%S %d/%m/%Y')}")

placeholder = st.empty()

while True:
    new_df = fetch_new_data()
    
    if not new_df.empty:
        st.session_state['monitor_df'] = pd.concat([st.session_state['monitor_df'], new_df], ignore_index=True)
        if 'id' in st.session_state['monitor_df'].columns:
            st.session_state['monitor_df'].drop_duplicates(subset=['id'], keep='last', inplace=True)
    
    df = st.session_state['monitor_df'].copy()
    run_id = str(uuid.uuid4())[:8]

    with placeholder.container():
        if df.empty:
            st.info("📡 Đang lắng nghe dữ liệu mới từ Spark...")
        else:
            # --- METRICS ---
            total = len(df)
            toxic_count = df['is_hate'].sum()
            clean_count = total - toxic_count
            toxic_ratio = (toxic_count / total) * 100
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Tổng tin đã quét", f"{total:,}", delta="Real-time")
            m2.metric("Tin Độc hại", f"{toxic_count:,}", f"{toxic_ratio:.1f}%", delta_color="inverse")
            m3.metric("Tin Sạch", f"{clean_count:,}", delta_color="normal")
            
            all_types = []
            if 'type_attack' in df.columns:
                for x in df['type_attack']:
                    if isinstance(x, list): all_types.extend(x)
            top_type = max(set(all_types), key=all_types.count) if all_types else "N/A"
            m4.metric("Loại tấn công Top 1", top_type)

            st.markdown("---")

            # --- CHARTS ---
            col_types, col_targets = st.columns([1.5, 1])
            
            with col_types:
                st.subheader("📊 Phân bố Loại hình Tấn công")
                if all_types:
                    type_counts = pd.Series(all_types).value_counts().reset_index()
                    type_counts.columns = ['Loại tấn công', 'Số lượng']
                    fig_bar = px.bar(type_counts, x='Số lượng', y='Loại tấn công', orientation='h', text='Số lượng', color='Số lượng', color_continuous_scale='Reds')
                    fig_bar.update_layout(height=350, margin=dict(l=0, r=0, t=0, b=0))
                    st.plotly_chart(fig_bar, use_container_width=True, key=f"bar_{run_id}")
                else:
                    st.info("Chưa phát hiện loại tấn công cụ thể.")

            with col_targets:
                st.subheader("🎯 Mục tiêu Tấn công")
                dangerous = ['Offensive', 'Hate']
                targets_data = {
                    "Cá nhân": df[df['Individual'].isin(dangerous)].shape[0] if 'Individual' in df.columns else 0,
                    "Nhóm/Tổ chức": df[df['Group'].isin(dangerous)].shape[0] if 'Group' in df.columns else 0,
                    "Xã hội": df[df['Societal'].isin(dangerous)].shape[0] if 'Societal' in df.columns else 0
                }
                fig_target = go.Figure(data=[go.Pie(labels=list(targets_data.keys()), values=list(targets_data.values()), hole=.5, marker_colors=['#FF9999', '#FF6666', '#FF0000'])])
                fig_target.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_target, use_container_width=True, key=f"target_{run_id}")

            # --- LOGS TABLE (ĐÃ SỬA) ---
            st.subheader("📝 Nhật ký Giám sát (Real-time Logs)")
            
            df_show = df.sort_values(by='timestamp', ascending=False).head(100)
            
            # 1. Bổ sung cột cmt_processed vào danh sách hiển thị
            cols = ['timestamp', 'cmt', 'cmt_processed', 'type_attack', 'Individual', 'Group', 'Societal']
            # Lọc để tránh lỗi nếu cột chưa có
            cols = [c for c in cols if c in df_show.columns]
            
            # 2. Hàm tô màu mới: Ép chữ màu ĐEN (color: black) để đọc được trên nền sáng
            def style_row(row):
                # Màu nền: Đỏ nhạt (độc hại) / Xanh nhạt (sạch)
                bg_color = '#ffcdd2' if row.get('is_hate') else '#c8e6c9'
                # QUAN TRỌNG: Thêm 'color: black' để ghi đè mặc định màu trắng của Dark Mode
                return [f'background-color: {bg_color}; color: black; border-bottom: 1px solid white'] * len(row)

            st.dataframe(
                df_show[cols].style.apply(style_row, axis=1),
                use_container_width=True,
                height=500,
                column_config={
                    "timestamp": st.column_config.DatetimeColumn("Thời gian", format="HH:mm:ss"),
                    "cmt": st.column_config.TextColumn("Bình luận Gốc", width="medium"),
                    "cmt_processed": st.column_config.TextColumn("Đã Xử lý (Clean)", width="medium"),
                    "type_attack": "Loại tấn công",
                    "Individual": "Cá nhân",
                    "Group": "Tổ chức",
                    "Societal": "Xã hội"
                }
            )
            
    time.sleep(1)