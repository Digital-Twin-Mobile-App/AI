import tensorflow as tf
import base64
from io import BytesIO
from PIL import Image
from model import model_instance
import numpy as np
import cv2


def decode_base64_image(base64_str):
    try:
        image_data = base64.b64decode(base64_str)
        image_pil = Image.open(BytesIO(image_data)).convert("RGB")
        return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)  # return as BGR (OpenCV format)
    except Exception as e:
        raise ValueError(f"Lỗi khi giải mã ảnh base64: {str(e)}")

def preprocess_cv_image(img, target_size=(224, 224)):
    if img is None:
        raise ValueError("Ảnh đầu vào không hợp lệ.")
    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    return img


def predict_process(input_value):
    
    # Decode the base64 image
    img_cv = decode_base64_image(input_value)

    # CNN prediction
    img_preprocessed = preprocess_cv_image(img_cv)
    img_tensor = np.expand_dims(img_preprocessed, axis=0)
    predictions = model_instance.model.predict(img_tensor)[0]
    class_idx = int(np.argmax(predictions))
    species = list(model_instance.labels.keys())[class_idx]
    confidence = float(predictions[class_idx])

    # Sprout stage analysis
    stage, height_ratio = model_instance.analyze_sprout_growth_cv_image(img_cv)
    
    return {
            "species": species,
            "stage": stage,
            "confidence": confidence,
            "height_ratio": height_ratio
        }
