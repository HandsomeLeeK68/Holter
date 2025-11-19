import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from scipy.signal import find_peaks # Phát hiện đỉnh R
import pywt # Thư viện Wavelet
# --- CẤU HÌNH DỮ LIỆU & LỜI KHUYÊN ---

# Định nghĩa thông tin chi tiết cho 5 lớp (Classes)
CLASS_INFO = {
    'N': {
        "name": "Bình thường (Normal)",
        "color": "green",
        "advice": "Nhịp tim của bạn đang ở trạng thái ổn định. Hãy duy trì lối sống lành mạnh, tập thể dục đều đặn và ăn uống cân bằng."
    },
    'S': {
        "name": "Ngoại tâm thu trên thất (SVEB)",
        "color": "orange",
        "advice": "Thường lành tính nhưng có thể do căng thẳng, caffeine hoặc thiếu ngủ. Nên hạn chế chất kích thích, nghỉ ngơi hợp lý. Nếu thấy hồi hộp nhiều, hãy đi khám."
    },
    'V': {
        "name": "Ngoại tâm thu thất (VEB)",
        "color": "red",
        "advice": "Có thể gây cảm giác hẫng nhịp. Nguyên nhân có thể do rối loạn điện giải, bệnh tim nền hoặc stress. Cần theo dõi tần suất, nếu xuất hiện dày đặc hoặc gây chóng mặt, cần gặp bác sĩ tim mạch ngay."
    },
    'F': {
        "name": "Nhịp hỗn hợp (Fusion Beat)",
        "color": "purple",
        "advice": "Là sự kết hợp giữa nhịp bình thường và nhịp bất thường. Đây là dấu hiệu cần được bác sĩ chuyên khoa đánh giá kỹ hơn qua Holter ECG."
    },
    'Q': {
        "name": "Nhịp không xác định (Unknown)",
        "color": "gray",
        "advice": "Tín hiệu bị nhiễu hoặc không rõ ràng. Vui lòng kiểm tra lại thiết bị đo, tiếp xúc điện cực và đo lại trong trạng thái tĩnh. Hoặc đi khám chuyên khoa để được đánh giá chính xác hơn."
    }
}

CLASSES_KEYS = ['N', 'S', 'V', 'F', 'Q']

def load_arrhythmia_model(model_path="model\\ecg_model_code 17_t5.h5"):
    try:
        model = load_model(model_path, compile=False)
        return model
    except Exception as e:
        print(f"Lỗi tải model: {e}")
        return None

def get_model_input_length(model):
    """Tự động lấy độ dài input đầu vào của model"""
    try:
        input_shape = model.input_shape
        if input_shape and len(input_shape) >= 2 and input_shape[1] is not None:
            return int(input_shape[1])      #input_shape = (None, 187, 1) = (batch_size, input_length, features)
        # Nếu không lấy được từ input_shape, thử lấy từ lớp đầu tiên
        first_layer = model.layers[0]
        if hasattr(first_layer, 'input_shape'):
            cfg_shape = first_layer.input_shape
            if cfg_shape and len(cfg_shape) >= 2 and cfg_shape[1] is not None:
                 return int(cfg_shape[1])
    except:
        pass
    return 187 # Fallback

def denoise_signal_wavelet(signal, wavelet='sym8', level=1):
    """Lọc nhiễu Wavelet"""
    if len(signal) < 10:
        return signal
    try:
        coeffs = pywt.wavedec(signal, wavelet, mode='per', level=level) 
        detail_coeffs = coeffs[-1]
        sigma = np.median(np.abs(detail_coeffs)) / 0.6745 
        thresh = sigma * np.sqrt(2 * np.log(len(signal)))
        new_coeffs = [coeffs[0]]
        for c in coeffs[1:]:
            new_coeffs.append(pywt.threshold(c, thresh, mode='soft'))
        denoised_signal = pywt.waverec(new_coeffs, wavelet, mode='per')
        
        if len(denoised_signal) > len(signal):
            denoised_signal = denoised_signal[:len(signal)]
        elif len(denoised_signal) < len(signal):
            pad_width = len(signal) - len(denoised_signal)
            denoised_signal = np.pad(denoised_signal, (0, pad_width), 'edge')
        return denoised_signal
    except:
        return signal

def detect_and_segment(denoised_ecg_signal, r_peak_height=0.5, r_peak_distance=150, output_length=187):
    """Phát hiện đỉnh R và phân đoạn"""
    peaks, _ = find_peaks(denoised_ecg_signal, height=r_peak_height, distance=r_peak_distance)
    
    ratio_before = 99 / 187
    window_before = int(output_length * ratio_before)
    window_after = output_length - window_before - 1
    
    segments = []
    valid_peak_locations = []
    
    for peak_loc in peaks:
        start = peak_loc - window_before
        end = peak_loc + window_after + 1
        if start < 0 or end > len(denoised_ecg_signal):
            continue
        segment = denoised_ecg_signal[start : end]
        if len(segment) == output_length:
            segments.append(segment)
            valid_peak_locations.append(peak_loc)
        
    if not segments:
        return np.array([]), np.array([])
        
    return np.array(segments), np.array(valid_peak_locations)

def predict_from_segments(segments_array, model):
    """Dự đoán và trả về mã lớp (N, S, V...)"""
    if segments_array.ndim == 2:
        X = segments_array.reshape(-1, segments_array.shape[1], 1)  # Thêm chiều features=1
    else:
        X = segments_array

    y_pred_probs = model.predict(X)
    y_pred_indices = np.argmax(y_pred_probs, axis=1)
    
    # Trả về mã ký tự (N, S, V...) để frontend tra cứu trong CLASS_INFO
    predicted_codes = [CLASSES_KEYS[i] for i in y_pred_indices]
    return predicted_codes, y_pred_indices

# --- HÀM VẼ ĐỒ THỊ (HỖ TRỢ DARK MODE) ---
def set_plot_style(dark_mode=True):
    """Cấu hình style cho Matplotlib"""
    if dark_mode:
        plt.style.use('dark_background')
        plt.rcParams.update({
            "axes.facecolor": "#0e1117",
            "figure.facecolor": "#0e1117",
            "grid.color": "#444444",
            "text.color": "white",
            "xtick.color": "white",
            "ytick.color": "white"
        })
    else:
        plt.style.use('default')
        plt.rcParams.update({
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "grid.color": "#e6e6e6",
            "text.color": "black",
            "xtick.color": "black",
            "ytick.color": "black"
        })

def plot_raw_signal_with_peaks(raw_ecg, peaks, predicted_codes, dark_mode=False):
    set_plot_style(dark_mode)
    fig, ax = plt.subplots(figsize=(15, 5))
    
    line_color = "#00d4ff" if dark_mode else "#1f77b4"
    ax.plot(raw_ecg, label="Tín hiệu ECG", color=line_color, alpha=0.8, linewidth=1.2)
    
    # Vẽ các điểm R với màu tương ứng
    for code in CLASSES_KEYS:
        info = CLASS_INFO[code]
        # Lấy các đỉnh thuộc lớp này
        # predicted_codes là list, cần chuyển thành np array để so sánh
        mask = np.array(predicted_codes) == code
        class_peaks = peaks[mask]
        
        if len(class_peaks) > 0:
            ax.scatter(class_peaks, raw_ecg[class_peaks], 
                       color=info['color'], 
                       label=info['name'], 
                       s=70, zorder=5, edgecolors='white' if dark_mode else 'black')

    ax.set_title("Phân loại trên toàn bộ tín hiệu")
    ax.set_xlabel("Mẫu (Sample)")
    ax.set_ylabel("Biên độ (mV)")
    ax.legend(loc='upper right')
    plt.tight_layout()
    return fig

def plot_beat_segment(beat_data, pred_code=None, dark_mode=False):
    set_plot_style(dark_mode)
    fig, ax = plt.subplots(figsize=(8, 3))
    
    info = CLASS_INFO.get(pred_code, {'name': 'Unknown', 'color': 'gray'})
    
    ax.plot(beat_data.squeeze(), color=info['color'], linewidth=2)
    
    title = f"Nhịp: {info['name']}" if pred_code else "Hình dạng nhịp tim"
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    return fig