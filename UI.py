import streamlit as st
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import pandas as pd

# ====== Cấu hình trang ======
st.set_page_config(
    page_title="Nhận Dạng Bệnh Cây Trồng",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====== Custom CSS ======
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .stTitle {
        color: #2d5016;
        text-align: center;
        font-size: 3rem !important;
        font-weight: bold;
        padding: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .upload-section {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        margin: 1rem 0;
    }
    .metric-box {
        background: rgba(255,255,255,0.2);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
    }
    .info-box {
        background: #e8f5e9;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
        margin: 1rem 0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        width: 100%;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)


# ====== Load model ======
@st.cache_resource
def load_model():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = 'model.pth'
    CSV_PATH = 'dataset_labels.csv'

    # Đọc danh sách nhãn
    df = pd.read_csv(CSV_PATH)
    class_names = sorted(df["label"].unique().tolist())

    # Khởi tạo model
    num_classes = len(class_names)
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    return model, class_names, DEVICE


# Tiền xử lý ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ====== Sidebar ======
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=150)
    st.title("📋 Hướng Dẫn")
    st.markdown("""
    ### Cách sử dụng:
    1. 📤 Tải lên ảnh cây trồng
    2. ⏳ Đợi hệ thống phân tích
    3. 📊 Xem kết quả dự đoán

    ### Định dạng ảnh:
    - JPG, PNG, JPEG
    - Chất lượng tốt
    - Rõ nét, đủ ánh sáng

    ### Lưu ý:
    ⚠️ Kết quả chỉ mang tính chất tham khảo
    """)

    st.markdown("---")
    st.markdown("### 🔧 Thông Tin Hệ Thống")
    device_type = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    st.info(f"**Thiết bị:** {device_type}")

# ====== Main Content ======
st.title("🌿 HỆ THỐNG NHẬN DẠNG BỆNH CÂY TRỒNG")
st.markdown(
    "<p style='text-align: center; color: #666; font-size: 1.2rem;'>Sử dụng AI để phát hiện bệnh và chăm sóc cây trồng hiệu quả</p>",
    unsafe_allow_html=True)

# Load model
try:
    model, class_names, DEVICE = load_model()
    st.success("✅ Model đã được tải thành công!")
except Exception as e:
    st.error(f"❌ Lỗi khi tải model: {str(e)}")
    st.stop()

# Upload section
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Chọn ảnh cây trồng của bạn",
        type=["jpg", "png", "jpeg"],
        help="Tải lên ảnh cây trồng để phát hiện bệnh"
    )
    st.markdown("</div>", unsafe_allow_html=True)

# Processing and Results
if uploaded_file is not None:
    col_img, col_result = st.columns(2)

    with col_img:
        st.markdown("### 📸 Ảnh Đầu Vào")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True, caption="Ảnh bạn đã tải lên")

    with col_result:
        st.markdown("### 🔍 Đang Phân Tích...")

        # Progress bar
        progress_bar = st.progress(0)
        for i in range(100):
            progress_bar.progress(i + 1)

        # Dự đoán
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_idx].item()

        pred_label = class_names[pred_idx]

        # Phân tích nhãn
        if "_" in pred_label:
            plant, disease = pred_label.split("_", 1)
        else:
            plant, disease = pred_label, "Không phát hiện bệnh"

        # Hiển thị kết quả
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.markdown("### 🎯 KẾT QUẢ PHÂN TÍCH")

        st.markdown(f"""
        <div class='metric-box'>
            <h3>🌱 Loại Cây: {plant.capitalize()}</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='metric-box'>
            <h3>🦠 Tình Trạng: {disease.replace('_', ' ').title()}</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='metric-box'>
            <h3>📊 Độ Tin Cậy: {confidence * 100:.2f}%</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # Recommendations
    st.markdown("---")
    st.markdown("### 💡 Khuyến Nghị")

    if confidence > 0.8:
        confidence_text = "Độ tin cậy cao - Kết quả đáng tin cậy"
        confidence_color = "#4caf50"
    elif confidence > 0.6:
        confidence_text = "Độ tin cậy trung bình - Nên kiểm tra thêm"
        confidence_color = "#ff9800"
    else:
        confidence_text = "Độ tin cậy thấp - Hãy tham khảo ý kiến chuyên gia"
        confidence_color = "#f44336"

    st.markdown(f"""
    <div class='info-box' style='border-left-color: {confidence_color}; background: {confidence_color}20;'>
        <h4 style='color: {confidence_color};'>⚡ {confidence_text}</h4>
        <p><b>Lời khuyên:</b></p>
        <ul>
            <li>Theo dõi cây trồng định kỳ</li>
            <li>Tham khảo thêm ý kiến chuyên gia nếu cần</li>
            <li>Áp dụng biện pháp phòng trừ phù hợp</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)



else:
    # Welcome message
    st.markdown("""
    <div class='info-box'>
        <h3>👋 Chào mừng đến với hệ thống nhận dạng bệnh cây trồng!</h3>
        <p>Hệ thống sử dụng AI (ResNet50) để phát hiện và chẩn đoán bệnh trên cây trồng.</p>
        <p><b>Hãy tải lên một bức ảnh để bắt đầu!</b></p>
    </div>
    """, unsafe_allow_html=True)

    # Example images section
    st.markdown("### 📸 Ảnh Mẫu")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/2917/2917995.png", caption="Ảnh rõ nét",
                 use_container_width=True)
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/2917/2917994.png", caption="Đủ ánh sáng",
                 use_container_width=True)
    with col3:
        st.image("https://cdn-icons-png.flaticon.com/512/2917/2917993.png", caption="Chụp cận cảnh",
                 use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🌿 Phát triển với ❤️ bởi Nhóm 9</p>
    <p style='font-size: 0.9rem;'>Powered by PyTorch & Streamlit</p>
</div>
""", unsafe_allow_html=True)