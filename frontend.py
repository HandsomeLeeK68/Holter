#.\venv\Scripts\activate
#streamlit run frontend.py
#chạy thử với id=35
import streamlit as st
import pandas as pd
import numpy as np
import ast # Dùng để parse string list trong CSV nếu cần
from backend import (
    load_arrhythmia_model, 
    get_model_input_length,
    denoise_signal_wavelet, 
    detect_and_segment, 
    predict_from_segments, 
    plot_beat_segment,
    plot_raw_signal_with_peaks,
    CLASS_INFO
)
import os
import json

# --- Cấu hình trang ---
st.set_page_config(
    layout="wide",
    page_title="ECG Arrhythmia Classification",
    page_icon="🫀"
)

# --- Tải Model ---
@st.cache_resource
def setup_model(model_path):
    if not os.path.exists(model_path):
        return None, 0
    model = load_arrhythmia_model(model_path)
    # Tự động lấy độ dài input từ model
    input_len = get_model_input_length(model)
    return model, input_len

MODEL_PATH = "model\\ecg_model_code 17_t5.h5"
model, REQUIRED_LENGTH = setup_model(MODEL_PATH)

st.title("🫀 Phân loại Rối loạn Nhịp tim (ECG)")
st.caption("Hệ thống Phân loại Rối loạn Nhịp tim với cấu trúc CNN 1D + LSTM + Attention  \n Với các bước tiền xử lý: Lọc nhiễu Wavelet + Phát hiện đỉnh R + Phân đoạn nhịp tim")
st.caption(" Hệ thống được phát triển bởi: Lê Nghĩa Hiệp  \n Mssv: 20235326")

if model is None:
    st.error(f"⚠️ Không tìm thấy file `{MODEL_PATH}`. Vui lòng copy file model vào cùng thư mục với `app.py`.")
else:
    with st.sidebar:
        st.title("⚙️ Cấu hình")
        
        # Tùy chọn Dark Mode
        st.subheader("Giao diện")
        is_dark_mode = st.toggle("Chế độ Tối", value=False)
        
        st.divider()
        
        if is_dark_mode:
            dark_css = """
            <style>
                /* Nền chính: Xám Chì (Sáng hơn đen cũ) */
                .stApp {
                    background-color: #262730; 
                    color: #FAFAFA; /* Màu chữ trắng kem cho đỡ gắt */
                }
                
                /* Sidebar: Chỉnh cho khác biệt nhẹ với nền chính */
                [data-testid="stSidebar"] {
                    background-color: #31333F;
                    color: #FAFAFA;
                }
                
                /* Chỉnh màu các input/box cho hợp với nền xám */
                .stTextInput, .stSelectbox, .stNumberInput {
                    color: white;
                }
            </style>
            """
            st.markdown(dark_css, unsafe_allow_html=True)

        else:
            # LIGHT MODE
            light_css = """
            <style>
                /* Nền chính: Trắng sứ (Không dùng trắng tinh #FFF) */
                .stApp {
                    background-color: #F8F9FA;
                    color: #212529;
                }
                [data-testid="stSidebar"] {
                    background-color: #E9ECEF;
                    color: #212529;
                }
            </style>
            """
            st.markdown(light_css, unsafe_allow_html=True)

        # Hiển thị thông tin model đã tải
        st.success("✅ Đã uploaded model thành công! (CNN 1D + LSTM + Attention)")
        st.info(f"📏 Model yêu cầu độ dài nhịp tim: **{REQUIRED_LENGTH}** điểm dữ liệu.")
    
    # --- UPLOAD DATA (JSON & CSV) ---
    st.subheader("Tải lên dữ liệu điện tim (JSON hoặc CSV)")   
    uploaded_file = st.file_uploader("Tải lên dữ liệu điện tim (JSON hoặc CSV)", type=["json", "csv"])
    
    raw_ecg = None
    data_source_name = ""

    if uploaded_file is not None:
        try:
            patient_data_map = {}
            patient_ids = []

            # XỬ LÝ JSON
            if uploaded_file.name.endswith('.json'):
                data = json.load(uploaded_file)
                if isinstance(data, list):
                    for i, d in enumerate(data):
                        pid = d.get("id", f"Bản ghi {i}") if isinstance(d, dict) else f"Bản ghi {i}"
                        reading = d["reading"] if isinstance(d, dict) else d
                        patient_data_map[pid] = reading
                        patient_ids.append(pid)
                elif isinstance(data, dict):
                    patient_data_map = data
                    patient_ids = list(data.keys())

            # XỬ LÝ CSV
            elif uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                # Trường hợp 1: Có cột 'reading' chứa string list "[0.1, 0.2...]"
                if 'reading' in df.columns:
                    for i, row in df.iterrows():
                        pid = str(row['id']) if 'id' in df.columns else f"Row {i}"
                        try:
                            # Chuyển string "[...]" thành list thực
                            reading_val = row['reading']
                            if isinstance(reading_val, str):
                                reading = ast.literal_eval(reading_val)
                            else:
                                reading = reading_val # Nếu đã là list hoặc np array
                            patient_data_map[pid] = reading
                            patient_ids.append(pid)
                        except:
                            continue
                # Trường hợp 2: File CSV thuần số (mỗi dòng là 1 reading hoặc mỗi cột là 1 reading)
                else:
                    # Giả sử mỗi dòng là một chuỗi tín hiệu
                    for i in range(len(df)):
                        pid = f"Dòng {i}"
                        reading = df.iloc[i].values.tolist()
                        # Chỉ lấy dòng nào đủ dài
                        if len(reading) > 100:
                            patient_data_map[pid] = reading
                            patient_ids.append(pid)

            selected_id = st.selectbox("Chọn bản ghi để phân tích:", patient_ids)
            raw_ecg = np.array(patient_data_map[selected_id])

            # --- Tùy chỉnh tham số ---
            with st.expander("⚙️ Cấu hình nâng cao (Wavelet & Peak Detection)", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Lọc nhiễu (DWT)**")
                    wavelet_type = st.selectbox("Loại Wavelet", ['sym8', 'db4', 'db8', 'coif5'], index=0)
                    wavelet_level = st.number_input("Level", 1, 9, 1)
                with col2:
                    st.markdown("**Phát hiện đỉnh R**")
                    r_peak_height = st.number_input("Chiều cao tối thiểu", 0.1, 10.0, 0.5, 0.1)
                    r_peak_distance = st.number_input("Khoảng cách tối thiểu", 50, 500, 150, 10)
        except Exception as e:
            st.error(f"Lỗi đọc file: {e}")

# --- PHÂN TÍCH ---
    if raw_ecg is not None:
        # Nút bấm chỉ làm nhiệm vụ TÍNH TOÁN và LƯU VÀO SESSION STATE
        if st.button("🚀 Bắt đầu Chẩn đoán", type="primary"):
            
            # 1. Lọc nhiễu & Phân đoạn
            with st.spinner("Đang xử lý tín hiệu..."):
                denoised_ecg = denoise_signal_wavelet(raw_ecg, wavelet=wavelet_type)
                segments, valid_peaks = detect_and_segment(
                    denoised_ecg, 
                    r_peak_height, 
                    r_peak_distance, 
                    output_length=REQUIRED_LENGTH
                )

            if len(segments) == 0:
                st.warning("Không phát hiện được nhịp tim nào. Hãy thử giảm 'Chiều cao đỉnh R'.")
                # Xóa kết quả cũ nếu tính toán thất bại
                if 'analysis_result' in st.session_state:
                    del st.session_state.analysis_result
            else:
                # 2. Dự đoán
                with st.spinner("AI đang phân tích từng nhịp tim..."):
                    predicted_codes, predicted_indices = predict_from_segments(segments, model)
                
                st.success(f"Hoàn tất! Đã phân tích {len(segments)} nhịp tim.")

                # LƯU KẾT QUẢ VÀO SESSION STATE
                st.session_state.analysis_result = {
                    "raw_ecg": raw_ecg, # Lưu lại tín hiệu gốc tương ứng với kết quả này
                    "segments": segments,
                    "valid_peaks": valid_peaks,
                    "predicted_codes": predicted_codes
                }

        # PHẦN HIỂN THỊ (Nằm ngoài khối if st.button)
        # Kiểm tra xem đã có kết quả trong Session State chưa
        if 'analysis_result' in st.session_state:
            
            # Lấy dữ liệu từ Session State ra để hiển thị
            res = st.session_state.analysis_result
            
            # Kiểm tra an toàn: Nếu người dùng đổi file khác mà chưa bấm nút chạy lại, 
            # dữ liệu raw_ecg hiện tại sẽ khác dữ liệu đã lưu. 
            # Ta có thể cảnh báo hoặc vẫn hiện kết quả cũ. Ở đây ta cứ hiện kết quả đã lưu.
            saved_raw_ecg = res["raw_ecg"]
            saved_segments = res["segments"]
            saved_peaks = res["valid_peaks"]
            saved_codes = res["predicted_codes"]

            # --- KẾT QUẢ TỔNG QUAN ---
            st.subheader("1. Biểu đồ Điện tâm đồ (ECG)")
            # Lưu ý: Dùng saved_raw_ecg để đảm bảo đồng bộ với đỉnh R đã tìm
            fig_raw = plot_raw_signal_with_peaks(saved_raw_ecg, saved_peaks, saved_codes, dark_mode=is_dark_mode)
            st.pyplot(fig_raw)

            # --- THỐNG KÊ & LỜI KHUYÊN ---
            st.subheader("2. Kết quả Chẩn đoán & Lời khuyên")
            
            # Đếm số lượng từng loại
            counts = pd.Series(saved_codes).value_counts()
            
            col_left, col_right = st.columns([1, 1.5])
            
            with col_left:
                st.markdown("### Thống kê nhịp")
                for code, count in counts.items():
                    info = CLASS_INFO[code]
                    percent = (count / len(saved_segments)) * 100
                    
                    delta_value = f"{percent:.1f}%"
    
                    if code != 'N': 
                        delta_for_color = -percent
                    else:
                        delta_for_color = percent

                    st.metric(
                        label=info['name'], 
                        value=f"{count} nhịp", 
                        delta=delta_for_color, 
                        delta_color="normal" 
                    )
                    st.markdown(f"<div style='margin-top: -15px; margin-bottom: 20px; font-size: 13px; color: grey;'>({delta_value})</div>", unsafe_allow_html=True)

            with col_right:
                st.markdown("### Lời khuyên Bác sĩ AI")
                detected_codes = counts.index.tolist()
                priority_order = ['V', 'S', 'F', 'Q', 'N']
                detected_codes.sort(key=lambda x: priority_order.index(x) if x in priority_order else 99)

                for code in detected_codes:
                    info = CLASS_INFO[code]
                    box_class = "success-box" if code == 'N' else "danger-box" if code in ['V', 'F'] else "warning-box"
                    
                    st.markdown(f"""
                    <div class="advice-box {box_class}">
                        <strong>{info['name']}</strong> ({counts[code]} lần)<br>
                        {info['advice']}
                    </div>
                    """, unsafe_allow_html=True)


            # --- CHI TIẾT TỪNG NHỊP ---
            st.subheader("3. Soi chi tiết từng nhịp")
            
            # Slider nằm ở đây, khi kéo nó sẽ re-run, nhưng vì 'analysis_result' vẫn còn trong session_state
            # nên đoạn code này vẫn được thực thi -> Hình ảnh sẽ cập nhật theo slider
            beat_idx = st.slider("Kéo để xem từng nhịp tim:", 0, len(saved_segments)-1, 0)
            
            curr_code = saved_codes[beat_idx]
            curr_info = CLASS_INFO[curr_code]
            
            col_b1, col_b2 = st.columns([3, 1])
            with col_b1:
                # Chú ý: index mảng bắt đầu từ 0, beat_idx lấy từ slider
                fig_seg = plot_beat_segment(saved_segments[beat_idx], curr_code, dark_mode=is_dark_mode)
                st.pyplot(fig_seg)
            with col_b2:
                st.info(f"**Nhịp thứ:** {beat_idx + 1}") # Hiển thị +1 cho người dùng dễ đọc
                st.markdown(f"**Phân loại:**\n\n{curr_info['name']}")

            # Bảng dữ liệu thô
            with st.expander("Xem bảng dữ liệu chi tiết"):
                df_res = pd.DataFrame({
                    "STT": range(1, len(saved_codes)+1),
                    "Vị trí (Sample)": saved_peaks,
                    "Mã": saved_codes,
                    "Chẩn đoán": [CLASS_INFO[c]['name'] for c in saved_codes]
                })
                st.dataframe(df_res, use_container_width=True)