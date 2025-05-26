# API Documentation

## Endpoints
```
    http://0.0.0.0:8000 
```

### 1. Health Check

**GET /**

- **Description:** Check if the API is running.

- **Response:**
```json
    {
        "message": "Welcome to the Tree State API!"
    }
```

### 2. Predict Tree State

**POST /predict//**

- **Description:** Predict the state of a tree from a base64-encoded image.

- **Request Body:**

```json
    {
    "input_data": "<base64-encoded-image>"
    }
```
    Note: input_data: Base64-encoded string of an image (JPEG/PNG).


- **Response:**
```json
   {
        "prediction": {
            "species": "OPPD-master-DATA-images_plants-APESV",
            "stage": "Mầm trưởng thành (stage_3)",
            "confidence": 1.0,
            "height_ratio": 0.675
        }
    }
```

- **Error Responses:**

    422 Unprocessable Entity: Invalid input or image format.
    400 Bad Request: Other errors.

### Example Request (Python)
```
import requests
import base64

with open("your_image.jpg", "rb") as img_file:
    img_b64 = base64.b64encode(img_file.read()).decode()

response = requests.post(
    "http://localhost:8000/predict/",
    json={"input_data": img_b64}
)
print(response.json())
```
