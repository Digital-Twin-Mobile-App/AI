import tensorflow as tf
import cv2
import numpy as np
import json
import os

# Đường dẫn
model_path = 'D:/DADN/oppd_model.h5'
labels_path = 'D:/DADN/labels.json'
test_image_path = 'D:/DADN/test_image.jpg'

# Hàm tiền xử lý ảnh cho CNN
def preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")
    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    return img

# Hàm phân tích mức độ phát triển của mầm (không cần nhãn)
def analyze_sprout_growth(image_path):
    # Đọc ảnh gốc
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")
    
    # Chuyển sang không gian màu HSV để phát hiện mầm (xanh nhạt hoặc nâu)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Ngưỡng màu cho mầm (xanh nhạt, có thể điều chỉnh)
    lower_sprout = np.array([30, 30, 30])  # Ngưỡng dưới
    upper_sprout = np.array([90, 255, 255])  # Ngưỡng trên
    mask = cv2.inRange(img_hsv, lower_sprout, upper_sprout)
    
    # Làm mờ để giảm nhiễu
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # Tìm đường viền của vùng mầm
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "Không phát hiện được mầm", 0.0
    
    # Lấy đường viền lớn nhất (giả sử là mầm chính)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Tính chiều cao của mầm
    x, y, w, h = cv2.boundingRect(largest_contour)
    sprout_height = h  # Chiều cao của vùng mầm (theo pixel)
    
    # Chuẩn hóa chiều cao (so với chiều cao ảnh)
    img_height = img.shape[0]
    height_ratio = sprout_height / img_height
    
    # Xác định giai đoạn dựa trên chiều cao
    if height_ratio < 0.10:  # Mầm rất nhỏ
        stage = "Mới nảy mầm (stage_1)"
    elif height_ratio < 0.25:  # Mầm trung bình
        stage = "Mầm đang phát triển (stage_2)"
    else:  # Mầm lớn
        stage = "Mầm trưởng thành (stage_3)"
    
    return stage, height_ratio

# Hàm dự đoán
def predict_tree_stability(model, image_path, labels):
    # Dự đoán loài cây bằng CNN
    img = preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)[0]
    
    class_idx = np.argmax(prediction)
    species = list(labels.keys())[class_idx]
    confidence = float(prediction[class_idx])
    
    # Phân tích mức độ phát triển của mầm
    stage, height_ratio = analyze_sprout_growth(image_path)
    
    return {"species": species, "stage": stage, "confidence": confidence, "height_ratio": height_ratio}

# Tải mô hình
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Không tìm thấy mô hình tại {model_path}")
model = tf.keras.models.load_model(model_path)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Tải nhãn
if not os.path.exists(labels_path):
    raise FileNotFoundError(f"Không tìm thấy file nhãn tại {labels_path}")
with open(labels_path, 'r') as f:
    labels = json.load(f)

# Dự đoán
try:
    result = predict_tree_stability(model, test_image_path, labels)
    print(f"Dự đoán: {result['species']}")
    print(f"Giai đoạn: {result['stage']}")
    print(f"Độ tin cậy (phân loại loài): {result['confidence']:.4f}")
    print(f"Tỷ lệ chiều cao mầm: {result['height_ratio']:.4f}")
except Exception as e:
    print(f"Lỗi khi dự đoán: {str(e)}")