# Tree State Classification API

This service provides an API for classifying the state of trees from images using a TensorFlow Keras model.

---

## 0. Requirements

- Python 3.8+
- See `requirements.txt` for Python dependencies.

---

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Model Setup

- Place your trained Keras model (`.h5` file) and label file (`labels.json`) in the project directory.
- **Edit the model path in `model.py`:**

```python
# filepath: /home/hung-le/hcmut/AI/API/model.py
model_instance = AIModel("path-to-your-model.h5", "path-to-your-labels.json")
```

Replace `"path-to-your-model.h5"` and `"path-to-your-labels.json"` with the actual paths to your files.

---

## 3. Run the Server (Local)

```bash
uvicorn main:app --reload
```

The API will be available at [http://localhost:8000](http://localhost:8000).

---

## 4. API Usage

See `API.md` for detailed API documentation and example requests.

---