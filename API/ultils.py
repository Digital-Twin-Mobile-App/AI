import tensorflow as tf
import base64
from io import BytesIO
from PIL import Image
from model import model_instance
import numpy as np

def preprocess_input(input_value, target_size=(224, 224)):
    # Decode the base64 string into raw image bytes
    image_data = base64.b64decode(input_value)
    
    # Load the image using PIL
    image = Image.open(BytesIO(image_data))
    image = image.convert("RGB")  # Ensure the image has 3 channels (RGB)
    image = image.resize(target_size)
    
    # Convert the image to a NumPy array and scale to [0, 1]
    image_array = np.array(image).astype(np.float32) / 255.0

    # Add batch dimension to get [1, height, width, 3]
    input_tensor = np.expand_dims(image_array, axis=0)
    
    print(f"Input tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
    
    # Final checks
    if len(input_tensor.shape) != 4 or input_tensor.shape[0] != 1 or input_tensor.shape[-1] != 3:
        raise ValueError("Input must have shape [1, height, width, 3]")

    return input_tensor

def predict(input_value):
    # Preprocess the input
    input_tensor = preprocess_input(input_value)
    
    # Get model predictions
    result = model_instance.predict(input_tensor)
    
    return result