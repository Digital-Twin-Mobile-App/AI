# Tree State Classification API

This service provides an API for classifying the state of trees from images using a TensorFlow Keras model.

---

## 0. Requirements

- Python 3.8+ (for local run)
- See `requirements.txt` for Python dependencies.
- Note: recommend using Linux

---

## 1. Install Dependencies (Local)

```bash
pip install -r requirements.txt
```

---

## 2. Model Setup

- Place your trained Keras model (`.h5` file) and label file (`labels.json`) in the project directory **or specify their locations using environment variables**.
- The API will automatically read the following environment variables:
  - `MODEL_PATH`: Path to your Keras `.h5` model file (default: `./oppd_model.h5`)
  - `LABELS_PATH`: Path to your `labels.json` file (default: `./labels.json`)

You **do not need to edit any code** to change the model or label file paths—just set the environment variables as needed.

---

## 3. Run the Server (Local)

```bash
export MODEL_PATH=/path/to/your/model.h5
export LABELS_PATH=/path/to/your/labels.json
uvicorn main:app --reload
```

The API will be available at [http://localhost:8000](http://localhost:8000).

---

## 4. Run with Docker

The app is available as a Docker image: [`hungle142/tree-state-api`](https://hub.docker.com/r/hungle142/tree-state-api).

### Pull the image

```bash
docker pull hungle142/tree-state-api
```

### Run the container

```bash
docker run -d \
  -p 8000:8000 \
  -v /local/path/to/model.h5:/app/oppd_model.h5 \
  -v /local/path/to/labels.json:/app/labels.json \
  -e MODEL_PATH=/app/oppd_model.h5 \
  -e LABELS_PATH=/app/labels.json \
  --name tree-state-api \
  hungle142/tree-state-api
```

- Replace `/local/path/to/model.h5` and `/local/path/to/labels.json` with the actual paths to your files.
- The `-e` flags set the environment variables for the container.
- The `-v` flags mount your model and label files into the container.

* Example:
  + Model's path: /home/hung-le/hcmut/digital twin app/AI/API/oppd_model.h5
  + Label's path: /home/hung-le/hcmut/digital twin app/AI/API/labels.json
```
docker run -d \
  -p 8000:8000 \
  -v "/home/hung-le/hcmut/digital twin app/AI/API/oppd_model.h5":/app/oppd_model.h5 \
  -v "/home/hung-le/hcmut/digital twin app/AI/API/labels.json":/app/labels.json \
  -e MODEL_PATH=/app/oppd_model.h5 \
  -e LABELS_PATH=/app/labels.json \
  --name tree-state-api \
  hungle142/tree-state-api
```
### Show logs of the container:

```
docker logs tree-state-api
```

The API will be available at [http://localhost:8000](http://localhost:8000).

---

## 5. API Usage

See `API.md` for detailed API documentation and example requests.

---