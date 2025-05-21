import tensorflow as tf
import numpy as np
import json
import os

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

    def predict(self, input_data):
        # input_data: numpy array with shape [1, height, width, 3], dtype float32, values in [0, 1]
        predictions = self.model.predict(input_data)[0]  # shape: [num_classes]
        class_idx = int(np.argmax(predictions))
        class_name = list(self.labels.keys())[class_idx]
        confidence = float(predictions[class_idx])
        return {
            "class": class_name,
            "confidence": confidence
        }

# Khởi tạo mô hình
model_instance = AIModel("./oppd_model.h5", "./labels.json")