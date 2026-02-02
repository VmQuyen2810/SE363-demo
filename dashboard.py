import streamlit as st
import pandas as pd
import pymongo
import time
import plotly.express as px
import plotly.graph_objects as go
import uuid
from datetime import datetime, timedelta

# 1. CẤU HÌNH & CSS
st.set_page_config(page_title="Hệ thống Giám sát Hate Speech Real-time", layout="wide", page_icon="🛡️")

st.markdown("""
<style>
    div[data-testid="stMetricValue"] { font-size: 24px; }
    .stDataFrame { border: 1px solid #444; border-radius: 5px; }
    
    /* Animation cho Cảnh báo */
    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    .alert-box {
        background-color: #ff4b4b;
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        animation: blink 1s infinite;
        margin-bottom: 20px;
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
        st.error(f"Lỗi kết nối MongoDB: {e}")
        return None

client = init_connection()

# 2. QUẢN LÝ SESSION
if 'monitor_df' not in st.session_state:
    st.session_state['monitor_df'] = pd.DataFrame()
if 'last_fetch_time' not in st.session_state:
    st.session_state['last_fetch_time'] = datetime.utcnow() - timedelta(seconds=10)
if 'start_time' not in st.session_state:
    st.session_state['start_time'] = datetime.now()

# 3. HÀM LẤY DỮ LIỆU MỚI TỪ MONGODB
def fetch_new_data():
    if not client: return pd.DataFrame()
    db = client[DB_NAME]
    col = db[COL_NAME]
    
    query = {"timestamp": {"$gt": st.session_state['last_fetch_time']}}
    cursor = col.find(query).sort("timestamp", 1)
    new_data = list(cursor)
    
    if new_data:
        df_new = pd.DataFrame(new_data)
        max_time = df_new['timestamp'].max()
        if isinstance(max_time, str): max_time = pd.to_datetime(max_time)
        st.session_state['last_fetch_time'] = max_time
        
        if '_id' in df_new.columns: df_new['_id'] = df_new['_id'].astype(str)
        if 'timestamp' in df_new.columns:
            df_new['timestamp'] = pd.to_datetime(df_new['timestamp']) + timedelta(hours=7)
        return df_new
    return pd.DataFrame()

# 4. GIAO DIỆN CHÍNH
c1, c2 = st.columns([3, 1])
with c1: st.title("Giám sát Hate Speech ️")
with c2:
    if st.button("Reset"):
        st.session_state['monitor_df'] = pd.DataFrame()
        st.session_state['last_fetch_time'] = datetime.utcnow()
        st.session_state['start_time'] = datetime.now()
        st.rerun()

st.caption(f" Bắt đầu lúc: {st.session_state['start_time'].strftime('%H:%M:%S %d/%m/%Y')}")
alert_placeholder = st.empty() 
placeholder = st.empty()

while True:
    new_df = fetch_new_data()
    if not new_df.empty:
        st.session_state['monitor_df'] = pd.concat([st.session_state['monitor_df'], new_df], ignore_index=True)
        if 'id' in st.session_state['monitor_df'].columns:
            st.session_state['monitor_df'].drop_duplicates(subset=['id'], keep='last', inplace=True)
    
    df = st.session_state['monitor_df'].copy()
    run_id = str(uuid.uuid4())[:8]

    # --- LOGIC CẢNH BÁO (ALERT SYSTEM) ---
    is_under_attack = False
    toxic_velocity = 0
    
    if not df.empty and 'timestamp' in df.columns:
        # Lấy thời gian hiện tại (Server time)
        now = datetime.now()
        # Lọc các tin trong 10 giây gần nhất
        recent_df = df[df['timestamp'] >= (now - timedelta(seconds=10))]
        
        if not recent_df.empty:
            # Đếm số lượng tin Toxic trong 10s qua
            toxic_recent = recent_df['is_hate'].sum()
            total_recent = len(recent_df)
            
            # Tiêu chí cảnh báo: Có trên 20 tin Toxic trong 10s VÀ Tỷ lệ Toxic > 50%
            if toxic_recent > 20: 
                is_under_attack = True
                toxic_velocity = toxic_recent / 10.0 # tin/giây

    # Hiển thị Cảnh báo
    with alert_placeholder.container():
        if is_under_attack:
            st.markdown(f"""
            <div class="alert-box">
                CẢNH BÁO: PHÁT HIỆN SỐ LƯỢNG NGÔN TỪ THÙ GHÉT TẤN CÔNG CAO BẤT THƯỜNG! <br>
                Tốc độ: {toxic_velocity:.1f} tin/giây
            </div>
            """, unsafe_allow_html=True)
        else:
            st.empty() 

    with placeholder.container():
        if df.empty:
            st.info(" Đang kết nối...")
        else:
            # METRICS
            total = len(df)
            toxic_count = df['is_hate'].sum()
            clean_count = total - toxic_count
            toxic_ratio = (toxic_count / total) * 100 if total > 0 else 0
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Tổng tin", f"{total:,}")
            m2.metric("Độc hại", f"{toxic_count:,}", f"{toxic_ratio:.1f}%", delta_color="inverse")
            m3.metric("Sạch", f"{clean_count:,}")
            
            all_types = []
            if 'type_attack' in df.columns:
                for x in df['type_attack']:
                    if isinstance(x, list): all_types.extend(x)
            top_type = max(set(all_types), key=all_types.count) if all_types else "N/A"
            m4.metric("Top Attack", top_type)

            st.markdown("---")

            # CHARTS
            col_chart1, col_chart2 = st.columns([2, 1])
            
            with col_chart1:
                st.subheader("📈 Lưu lượng Tấn công (Real-time)")
                if 'timestamp' in df.columns:
                    # Gom nhóm theo từng 5 giây để thấy rõ đỉnh (Peak) tấn công
                    df_trend = df.copy()
                    df_trend['time_block'] = df_trend['timestamp'].dt.floor('5s')
                    
                    trend_data = df_trend.groupby('time_block')['is_hate'].sum().reset_index()
                    
                    fig = px.area(trend_data, x='time_block', y='is_hate', 
                                  title="Số lượng tin Toxic (theo mỗi 5s)",
                                  labels={'is_hate': 'Toxic Count', 'time_block': 'Time'},
                                  color_discrete_sequence=['#ff4b4b'])
                    
                    # Highlight vùng nguy hiểm
                    fig.add_hrect(y0=10, y1=100, line_width=0, fillcolor="red", opacity=0.1, annotation_text="Vùng Nguy Hiểm")
                    
                    fig.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig, use_container_width=True, key=f"trend_{run_id}")

            with col_chart2:
                st.subheader("🎯 Mục tiêu")
                dg = ['Offensive', 'Hate']
                t_data = {
                    "Cá nhân": df[df['Individual'].isin(dg)].shape[0] if 'Individual' in df.columns else 0,
                    "Tổ chức": df[df['Group'].isin(dg)].shape[0] if 'Group' in df.columns else 0,
                    "Xã hội": df[df['Societal'].isin(dg)].shape[0] if 'Societal' in df.columns else 0
                }
                fig = go.Figure(data=[go.Pie(labels=list(t_data.keys()), values=list(t_data.values()), hole=.5)])
                fig.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
                st.plotly_chart(fig, use_container_width=True, key=f"pie_{run_id}")

            # DATA TABLE
            st.subheader("Live Logs")
            df_show = df.sort_values(by='timestamp', ascending=False).head(50) # Giảm xuống 50 cho nhẹ
            
            cols = ['timestamp', 'cmt', 'type_attack', 'Individual', 'Group', 'Societal']
            cols = [c for c in cols if c in df_show.columns]
            
            def style_row(row):
                bg = '#ffcdd2' if row.get('is_hate') else '#c8e6c9'
                return [f'background-color: {bg}; color: black'] * len(row)

            st.dataframe(
                df_show[cols].style.apply(style_row, axis=1),
                use_container_width=True,
                height=400,
                column_config={
                    "timestamp": st.column_config.DatetimeColumn("Thời gian", format="HH:mm:ss"),
                    "cmt": st.column_config.TextColumn("Nội dung", width="large"),
                }
            )
            
    time.sleep(1)