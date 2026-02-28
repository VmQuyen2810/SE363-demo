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

# 3. HÀM LẤY DỮ LIỆU
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
with c1: st.title(" Giám sát ngôn từ Thù ghét thời gian thực")
with c2:
    if st.button("Reset"):
        st.session_state['monitor_df'] = pd.DataFrame()
        st.session_state['last_fetch_time'] = datetime.utcnow()
        st.session_state['start_time'] = datetime.now()
        st.rerun()

st.caption(f"Bắt đầu lúc: {st.session_state['start_time'].strftime('%H:%M:%S %d/%m/%Y')}")
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

# --- LOGIC CẢNH BÁO  ---
    is_under_attack = False
    toxic_velocity = 0
    increase_pct = 0
    
    # Cấu hình ngưỡng 
    WINDOW_SECONDS = 3       # Chỉ xét 3 giây gần nhất để nhạy với Burst
    BASELINE_RATE = 2.0      # Mức bình thường 
    ALERT_THRESHOLD = 5.0    # Ngưỡng báo động (lớn hơn mức bình thường gấp đôi)

    if not df.empty and 'timestamp' in df.columns:
        now = datetime.now()
        # Lấy dữ liệu trong cửa sổ trượt
        recent_df = df[df['timestamp'] >= (now - timedelta(seconds=WINDOW_SECONDS))]
        
        if not recent_df.empty:
            # Tính số lượng toxic thực tế trong cửa sổ
            toxic_count_window = recent_df['is_hate'].sum()
            
            # Tính tốc độ trung bình (Tin toxic / giây)
            current_rate = toxic_count_window / WINDOW_SECONDS
            
            # Kiểm tra điều kiện tấn công
            if current_rate > ALERT_THRESHOLD:
                is_under_attack = True
                toxic_velocity = current_rate
                # Tính % tăng trưởng so với mức nền
                increase_pct = ((current_rate - BASELINE_RATE) / BASELINE_RATE) * 100

    # Hiển thị Cảnh báo
    with alert_placeholder.container():
        if is_under_attack:
            st.markdown(f"""
            <div class="alert-box">
                CẢNH BÁO: PHÁT HIỆN TẤN CÔNG BẤT THƯỜNG! <br>
                Tốc độ lây lan: {toxic_velocity:.1f} tin toxic/s <br>
                <span style="font-size: 16px; color: #ffcccc;">
                    (Tăng trưởng đột biến: +{increase_pct:.0f}% so với mức ổn định)
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.empty()
    with placeholder.container():
        if df.empty:
            st.info(" Đang kết nối...")
        else:
            # --- METRICS ---
            total = len(df)
            toxic_count = df['is_hate'].sum()
            clean_count = total - toxic_count
            toxic_ratio = (toxic_count / total) * 100 if total > 0 else 0
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Tổng tin", f"{total:,}")
            m2.metric("Độc hại", f"{toxic_count:,}", f"{toxic_ratio:.1f}%", delta_color="inverse")
            m3.metric("Sạch", f"{clean_count:,}")
            
            # Tính toán Type Attack (để dùng cho metric và biểu đồ)
            all_types = []
            if 'type_attack' in df.columns:
                for x in df['type_attack']:
                    if isinstance(x, list): all_types.extend(x)
            top_type = max(set(all_types), key=all_types.count) if all_types else "N/A"
            m4.metric("Top Attack", top_type)

            st.markdown("---")

            # --- CHARTS ---
            
            # Tầng 1: Biểu đồ thời gian 
            st.subheader("Lưu lượng ngôn từ thù ghét (Real-time)")
            if 'timestamp' in df.columns:
                df_trend = df.copy()
                df_trend['time_block'] = df_trend['timestamp'].dt.floor('5s')
                trend_data = df_trend.groupby('time_block')['is_hate'].sum().reset_index()
                
                fig = px.area(trend_data, x='time_block', y='is_hate', 
                              labels={'is_hate': 'Toxic Count', 'time_block': 'Time'},
                              color_discrete_sequence=['#ff4b4b'])
                fig.add_hrect(y0=0, y1=50, line_width=0, fillcolor="red", opacity=0.1)
                fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig, use_container_width=True, key=f"trend_{run_id}")

            st.markdown("<br>", unsafe_allow_html=True) # Khoảng cách

            # Tầng 2: Chia 2 cột cho Type Attack và Target
            col_chart_left, col_chart_right = st.columns([1.5, 1])
            
            # Biểu đồ Loại hình Tấn công 
            with col_chart_left:
                st.subheader("Phân bố Loại hình Tấn công ngôn từ thù ghét")
                if all_types:
                    type_counts = pd.Series(all_types).value_counts().reset_index()
                    type_counts.columns = ['Loại tấn công', 'Số lượng']
                    
                    fig_bar = px.bar(
                        type_counts, 
                        x='Số lượng', y='Loại tấn công', 
                        orientation='h', 
                        text='Số lượng',
                        color='Số lượng',
                        color_continuous_scale='Reds'
                    )
                    fig_bar.update_layout(height=350, margin=dict(l=0, r=0, t=0, b=0))
                    st.plotly_chart(fig_bar, use_container_width=True, key=f"bar_{run_id}")
                else:
                    st.info("Chưa phát hiện loại tấn công cụ thể.")

            # Biểu đồ Mục tiêu (Pie Chart)
            with col_chart_right:
                st.subheader("Mục tiêu")
                dg = ['Offensive', 'Hate']
                t_data = {
                    "Cá nhân": df[df['Individual'].isin(dg)].shape[0] if 'Individual' in df.columns else 0,
                    "Tổ chức": df[df['Group'].isin(dg)].shape[0] if 'Group' in df.columns else 0,
                    "Xã hội": df[df['Societal'].isin(dg)].shape[0] if 'Societal' in df.columns else 0
                }
                fig = go.Figure(data=[go.Pie(labels=list(t_data.keys()), values=list(t_data.values()), hole=.5)])
                fig.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0), showlegend=True)
                st.plotly_chart(fig, use_container_width=True, key=f"pie_{run_id}")

            # --- DATA TABLE ---
            st.subheader("Live Logs")
            df_show = df.sort_values(by='timestamp', ascending=False).head(50)
            
            # 1. Đổi tên cột sang tiếng Việt 
            rename_dict = {
                'timestamp': 'Thời gian',
                'cmt': 'Bình luận gốc',
                'cmt_processed': 'Bình luận xử lý',
                'type_attack': 'Loại tấn công',
                'Individual': 'Cá nhân',
                'Group': 'Tổ chức',
                'Societal': 'Xã hội'
            }
            cols = list(rename_dict.keys())
            cols = [c for c in cols if c in df_show.columns]
            
            df_display = df_show[cols].rename(columns=rename_dict)
            
            # Format lại cột thời gian cho gọn
            if 'Thời gian' in df_display.columns:
                df_display['Thời gian'] = df_display['Thời gian'].dt.strftime('%H:%M:%S')
            
            # 2. Hàm tô màu nền
            def style_row_display(row):
                # Lấy trạng thái is_hate từ df_show gốc (dựa trên index)
                bg = '#ffcdd2' if df_show.loc[row.name, 'is_hate'] else '#c8e6c9'
                return [f'background-color: {bg}; color: black; border-bottom: 1px solid #ccc;'] * len(row)

            # 3. Áp dụng Style và ép Wrap Text bằng CSS của Pandas
            styled_df = df_display.style.apply(style_row_display, axis=1) \
                .set_properties(subset=['Bình luận gốc', 'Bình luận xử lý'], **{
                    'white-space': 'pre-wrap',   # Bắt buộc rớt dòng
                    'word-break': 'break-word',  # Cắt chữ nếu quá dài
                    'min-width': '250px'         # Đảm bảo cột đủ rộng
                })
            
            # 4. Hiển thị bằng st.table bên trong một container có thanh cuộn
            with st.container(height=500):
                st.table(styled_df)
            
    time.sleep(1)