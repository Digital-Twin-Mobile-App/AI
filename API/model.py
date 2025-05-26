import tensorflow as tf
import numpy as np
import json
import os
import base64
from io import BytesIO
from PIL import Image
import cv2


class AIModel:
    def __init__(self, model_path, labels_path):
        self.model_path = model_path
        self.labels_path = labels_path
        self.model = self.load_model()
        self.labels = self.load_labels()
        print(f"Mô hình đã được tải từ {self.model_path}")
        print(f"Nhãn đã được tải từ {self.labels_path}")

    def load_model(self):
        if self.model_path.endswith(".h5"):
            return tf.keras.models.load_model(self.model_path)
        else:
            raise ValueError("Chỉ hỗ trợ định dạng .h5 cho mô hình phân loại.")

    def load_labels(self):
        if not os.path.exists(self.labels_path):
            raise FileNotFoundError(f"Không tìm thấy file nhãn tại {self.labels_path}")
        with open(self.labels_path, 'r') as f:
            return json.load(f)

    def analyze_sprout_growth_cv_image(self, img):
        if img is None:
            raise ValueError("Ảnh đầu vào không hợp lệ.")
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_sprout = np.array([30, 30, 30])
        upper_sprout = np.array([90, 255, 255])
        mask = cv2.inRange(img_hsv, lower_sprout, upper_sprout)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return "Không phát hiện được mầm", 0.0
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        sprout_height = h
        img_height = img.shape[0]
        height_ratio = sprout_height / img_height
        if height_ratio < 0.10:
            stage = "Mới nảy mầm (stage_1)"
        elif height_ratio < 0.25:
            stage = "Mầm đang phát triển (stage_2)"
        else:
            stage = "Mầm trưởng thành (stage_3)"
        return stage, height_ratio
                
# Khởi tạo mô hình
MODEL_PATH = os.environ.get("MODEL_PATH", "./oppd_model.h5")
LABELS_PATH = os.environ.get("LABELS_PATH", "./labels.json")
model_instance = AIModel(MODEL_PATH, LABELS_PATH)
