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

st.markdown("""
<style>
    /* Tùy chỉnh Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 24px;
    }
    /* Tô màu bảng */
    .stDataFrame {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# URI kết nối
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "toxic_db"
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
# Khởi tạo bộ nhớ phiên làm việc nếu chưa có
if 'monitor_df' not in st.session_state:
    st.session_state['monitor_df'] = pd.DataFrame()
if 'last_fetch_time' not in st.session_state:
    # Mặc định lấy dữ liệu từ thời điểm hiện tại trở đi (hoặc lùi lại 1 chút)
    st.session_state['last_fetch_time'] = datetime.utcnow() - timedelta(seconds=10)
if 'start_time' not in st.session_state:
    st.session_state['start_time'] = datetime.now()

# ==========================================
# 3. HÀM LẤY DỮ LIỆU MỚI (INCREMENTAL FETCH)
# ==========================================
def fetch_new_data():
    if not client: return pd.DataFrame()
    db = client[DB_NAME]
    col = db[COL_NAME]
    
    # Chỉ lấy các bản ghi MỚI HƠN lần cập nhật cuối cùng
    # Điều này giúp dashboard nhẹ hơn, không phải load lại cả triệu dòng
    query = {"timestamp": {"$gt": st.session_state['last_fetch_time']}}
    cursor = col.find(query).sort("timestamp", 1) # Lấy cũ nhất đến mới nhất để append
    
    new_data = list(cursor)
    
    if new_data:
        df_new = pd.DataFrame(new_data)
        
        # Cập nhật mốc thời gian để lần sau chỉ lấy cái mới hơn nữa
        max_time = df_new['timestamp'].max()
        if isinstance(max_time, str):
            max_time = pd.to_datetime(max_time)
        st.session_state['last_fetch_time'] = max_time
        
        # Xử lý ID và Timezone
        if '_id' in df_new.columns:
            df_new['_id'] = df_new['_id'].astype(str)
            
        # CHUYỂN ĐỔI MÚI GIỜ (UTC -> VIETNAM UTC+7)
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

# Hiển thị thời gian bắt đầu chạy
st.caption(f"🚀 Phiên giám sát bắt đầu lúc: {st.session_state['start_time'].strftime('%H:%M:%S %d/%m/%Y')}")

placeholder = st.empty()

while True:
    # 1. Lấy dữ liệu mới
    new_df = fetch_new_data()
    
    # 2. Cộng dồn vào Session State
    if not new_df.empty:
        st.session_state['monitor_df'] = pd.concat([st.session_state['monitor_df'], new_df], ignore_index=True)
        # Loại bỏ trùng lặp nếu có (dựa trên ID)
        if 'id' in st.session_state['monitor_df'].columns:
            st.session_state['monitor_df'].drop_duplicates(subset=['id'], keep='last', inplace=True)
    
    # Lấy DataFrame từ session ra để vẽ
    df = st.session_state['monitor_df'].copy()
    run_id = str(uuid.uuid4())[:8]

    with placeholder.container():
        if df.empty:
            st.info("📡 Đang lắng nghe dữ liệu mới từ Spark...")
        else:
            # --- A. METRICS TỔNG QUAN ---
            total = len(df)
            toxic_count = df['is_hate'].sum()
            clean_count = total - toxic_count
            toxic_ratio = (toxic_count / total) * 100
            
            # Layout 4 cột chỉ số chính
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Tổng tin đã quét", f"{total:,}", delta="Real-time")
            m2.metric("Tin Độc hại", f"{toxic_count:,}", f"{toxic_ratio:.1f}%", delta_color="inverse")
            m3.metric("Tin Sạch", f"{clean_count:,}", delta_color="normal")
            
            # Đếm loại tấn công phổ biến nhất
            all_types = []
            if 'type_attack' in df.columns:
                for x in df['type_attack']:
                    if isinstance(x, list): all_types.extend(x)
            top_type = max(set(all_types), key=all_types.count) if all_types else "N/A"
            m4.metric("Loại tấn công Top 1", top_type)

            st.markdown("---")

            # --- B. THỐNG KÊ CHI TIẾT (TARGETS & TYPES) ---
            col_types, col_targets = st.columns([1.5, 1])
            
            with col_types:
                st.subheader("📊 Phân bố Loại hình Tấn công (Type Attack)")
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

            with col_targets:
                st.subheader("🎯 Mục tiêu Tấn công (ViTHSD)")
                # Lọc ra các nhãn nguy hiểm để đếm
                dangerous = ['Offensive', 'Hate']
                
                targets_data = {
                    "Cá nhân": df[df['Individual'].isin(dangerous)].shape[0] if 'Individual' in df.columns else 0,
                    "Nhóm/Tổ chức": df[df['Group'].isin(dangerous)].shape[0] if 'Group' in df.columns else 0,
                    "Xã hội": df[df['Societal'].isin(dangerous)].shape[0] if 'Societal' in df.columns else 0
                }
                
                # Vẽ biểu đồ Radar hoặc Bar đơn giản cho Target
                fig_target = go.Figure(data=[go.Pie(
                    labels=list(targets_data.keys()), 
                    values=list(targets_data.values()), 
                    hole=.5,
                    marker_colors=['#FF9999', '#FF6666', '#FF0000']
                )])
                fig_target.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0), title_text="Tỷ lệ Mục tiêu")
                st.plotly_chart(fig_target, use_container_width=True, key=f"target_{run_id}")

            # --- C. BIỂU ĐỒ DIỄN BIẾN THEO THỜI GIAN ---
            st.subheader("📈 Diễn biến Tấn công theo Thời gian")
            if 'timestamp' in df.columns and not df.empty:
                df_time = df.copy()
                # Gom nhóm theo từng phút
                df_time['time_min'] = df_time['timestamp'].dt.floor('1min')
                
                time_agg = df_time.groupby('time_min').agg(
                    Tin_Doc_Hai=('is_hate', 'sum'),
                    Tong_Tin=('id', 'count')
                ).reset_index()
                
                fig_line = px.area(time_agg, x='time_min', y=['Tong_Tin', 'Tin_Doc_Hai'],
                                   labels={'value': 'Số lượng tin', 'time_min': 'Thời gian'},
                                   color_discrete_map={'Tong_Tin': '#cfd8dc', 'Tin_Doc_Hai': '#ff5252'})
                fig_line.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0), hovermode="x unified")
                st.plotly_chart(fig_line, use_container_width=True, key=f"line_{run_id}")

            # --- D. LOGS CHI TIẾT ---
            st.subheader("📝 Nhật ký Giám sát (Mới nhất lên đầu)")
            
            # Sắp xếp mới nhất lên đầu để dễ theo dõi
            df_show = df.sort_values(by='timestamp', ascending=False).head(100)
            
            cols = ['timestamp', 'cmt', 'type_attack', 'Individual', 'Group', 'Societal']
            cols = [c for c in cols if c in df_show.columns]
            
            def style_row(row):
                color = '#ffebee' if row.get('is_hate') else '#e8f5e9'
                return [f'background-color: {color}'] * len(row)

            st.dataframe(
                df_show[cols].style.apply(style_row, axis=1),
                use_container_width=True,
                height=400,
                column_config={
                    "timestamp": st.column_config.DatetimeColumn("Thời gian", format="HH:mm:ss DD/MM"),
                    "type_attack": "Loại tấn công",
                    "cmt": "Nội dung bình luận"
                }
            )
            
    time.sleep(1)