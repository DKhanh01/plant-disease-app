import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from ultralytics import YOLO
import streamlit as st

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
    .detection-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


# ====== Load models ======
@st.cache_resource
def load_models():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load ResNet50 model
    RESNET_MODEL_PATH = 'model.pth'
    CSV_PATH = 'dataset_labels.csv'

    # Đọc danh sách nhãn
    df = pd.read_csv(CSV_PATH)
    class_names = sorted(df["label"].unique().tolist())

    # Khởi tạo ResNet model
    num_classes = len(class_names)
    resnet_model = resnet50(weights=ResNet50_Weights.DEFAULT)
    in_features = resnet_model.fc.in_features
    resnet_model.fc = torch.nn.Linear(in_features, num_classes)
    resnet_model.load_state_dict(torch.load(RESNET_MODEL_PATH, map_location=DEVICE))
    resnet_model.eval()

    # Load YOLO model
    YOLO_MODEL_PATH = 'yolo11n.pt'
    yolo_model = YOLO(YOLO_MODEL_PATH)

    return resnet_model, yolo_model, class_names, DEVICE


def draw_bounding_boxes(image, detections):
    """Vẽ bounding box lên ảnh"""
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    # Tạo font (nếu không có font, sẽ dùng font mặc định)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

    for idx, det in enumerate(detections):
        box = det['box']
        label = det['label']

        color = colors[idx % len(colors)]

        # Vẽ bounding box
        draw.rectangle(box, outline=color, width=3)

        # Vẽ label (bỏ confidence)
        text = f"{label}"

        # Vẽ background cho text
        bbox = draw.textbbox((box[0], box[1] - 25), text, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((box[0], box[1] - 25), text, fill="white", font=font)

    return img_draw


def detect_with_yolo(yolo_model, image):
    """Phát hiện đối tượng với YOLO"""
    # Convert PIL to OpenCV format
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Chạy YOLO detection
    results = yolo_model(img_cv, conf=0.25)  # confidence threshold 0.25

    detections = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Lấy tọa độ bounding box
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            # Lấy confidence và class
            confidence = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            label = result.names[cls]

            detections.append({
                'box': (int(x1), int(y1), int(x2), int(y2)),
                'label': label,
                'confidence': confidence,
                'class_id': cls
            })

    return detections


def crop_detection_for_classification(image, box):
    """Cắt vùng detection để phân loại bằng ResNet"""
    x1, y1, x2, y2 = box
    return image.crop((x1, y1, x2, y2))


# Tiền xử lý ảnh cho ResNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ====== Sidebar ======
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=150)
    st.title("📋 Hướng Dẫn")

    # Chọn chế độ
    detection_mode = st.radio(
        "Chọn chế độ phát hiện:",
        ["YOLO + Classification", "Classification Only"],
        help="YOLO sẽ phát hiện vị trí bệnh, sau đó phân loại chi tiết"
    )

    st.markdown("""
    ### Cách sử dụng:
    1. 📤 Tải lên ảnh cây trồng
    2. 🎯 Chọn chế độ phát hiện
    3. ⏳ Đợi hệ thống phân tích
    4. 📊 Xem kết quả dự đoán

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
    st.info(f"**Chế độ:** {detection_mode}")

# ====== Main Content ======
st.title("🌿 HỆ THỐNG NHẬN DẠNG BỆNH CÂY TRỒNG")
st.markdown(
    "<p style='text-align: center; color: #666; font-size: 1.2rem;'>Sử dụng AI (YOLO + ResNet50) để phát hiện và chẩn đoán bệnh trên cây trồng</p>",
    unsafe_allow_html=True)

# Load models
try:
    resnet_model, yolo_model, class_names, DEVICE = load_models()
except Exception as e:
    st.error(f"❌ Lỗi khi tải model: {str(e)}")
    st.info("💡 Lưu ý: Đảm bảo bạn đã có file 'model' trong thư mục dự án")
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
    image = Image.open(uploaded_file).convert("RGB")

    if detection_mode == "YOLO + Classification":
        # ====== YOLO Detection Mode ======
        st.markdown("### 🎯 Phát hiện với YOLO + Phân loại với ResNet50")

        col_original, col_detected = st.columns(2)

        with col_original:
            st.markdown("#### 📸 Ảnh Gốc")
            st.image(image, use_container_width=True, caption="Ảnh bạn đã tải lên")

        with col_detected:
            st.markdown("#### 🔍 Đang Phân Tích...")
            progress_bar = st.progress(0)

            # YOLO Detection
            progress_bar.progress(30)
            detections = detect_with_yolo(yolo_model, image)

            progress_bar.progress(60)

            if len(detections) > 0:
                # Vẽ bounding boxes
                img_with_boxes = draw_bounding_boxes(image, detections)
                st.image(img_with_boxes, use_container_width=True,
                         caption=f"Phát hiện {len(detections)} vùng bệnh")
                progress_bar.progress(100)
            else:
                st.warning("⚠️ Không phát hiện được vùng bệnh nào")
                progress_bar.progress(100)

        # Hiển thị kết quả chi tiết
        if len(detections) > 0:
            st.markdown("---")
            st.markdown("### 📊 Kết Quả Chi Tiết Từng Vùng")

            for idx, det in enumerate(detections, 1):
                with st.expander(f"🔬 Vùng {idx}: {det['label']}"):
                    col_crop, col_class = st.columns(2)

                    with col_crop:
                        # Hiển thị vùng đã crop
                        cropped_img = crop_detection_for_classification(image, det['box'])
                        st.image(cropped_img, caption=f"Vùng phát hiện {idx}",
                                 use_container_width=True)

                    with col_class:
                        # Phân loại chi tiết với ResNet
                        st.markdown("**🧬 Phân loại chi tiết:**")

                        img_tensor = transform(cropped_img).unsqueeze(0).to(DEVICE)
                        with torch.no_grad():
                            outputs = resnet_model(img_tensor)
                            probs = torch.nn.functional.softmax(outputs, dim=1)
                            pred_idx = torch.argmax(probs, dim=1).item()

                        pred_label = class_names[pred_idx]

                        # Phân tích nhãn
                        if "_" in pred_label:
                            plant, disease = pred_label.split("_", 1)
                        else:
                            plant, disease = pred_label, "Không phát hiện bệnh"

                        st.markdown(f"""
                        <div class='detection-box'>
                            <p><b>🌱 Loại cây:</b> {plant.capitalize()}</p>
                            <p><b>🦠 Bệnh:</b> {disease.replace('_', ' ').title()}</p>
                        </div>
                        """, unsafe_allow_html=True)

    else:
        # ====== Classification Only Mode ======
        col_img, col_result = st.columns(2)

        with col_img:
            st.markdown("### 📸 Ảnh Đầu Vào")
            st.image(image, use_container_width=True, caption="Ảnh bạn đã tải lên")

        with col_result:
            st.markdown("### 🔍 Đang Phân Tích...")

            progress_bar = st.progress(0)
            for i in range(100):
                progress_bar.progress(i + 1)

            # Dự đoán
            img_tensor = transform(image).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                outputs = resnet_model(img_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()

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

            st.markdown("</div>", unsafe_allow_html=True)

        # Recommendations
        st.markdown("---")
        st.markdown("### 💡 Khuyến Nghị")

        st.markdown(f"""
        <div class='info-box'>
            <h4 style='color: #4caf50;'>⚡ Kết quả phân tích</h4>
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
        <p>Hệ thống sử dụng AI tiên tiến:</p>
        <ul>
            <li><b>YOLO</b>: Phát hiện vị trí bệnh với bounding box</li>
            <li><b>ResNet50</b>: Phân loại chi tiết loại bệnh</li>
        </ul>
        <p><b>Hãy tải lên một bức ảnh để bắt đầu!</b></p>
    </div>
    """, unsafe_allow_html=True)

    # Example images section
    st.markdown("### 📸 Ảnh Mẫu")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/2917/2917995.png",
                 caption="Ảnh rõ nét", use_container_width=True)
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/2917/2917994.png",
                 caption="Đủ ánh sáng", use_container_width=True)
    with col3:
        st.image("https://cdn-icons-png.flaticon.com/512/2917/2917993.png",
                 caption="Chụp cận cảnh", use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🌿 Phát triển với ❤️ bởi Nhóm 9</p>
    <p style='font-size: 0.9rem;'>Powered by YOLO, PyTorch & Streamlit</p>
</div>
""", unsafe_allow_html=True)