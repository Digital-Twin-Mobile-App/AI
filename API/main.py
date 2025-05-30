from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from ultils import predict_process

app = FastAPI()

class InputData(BaseModel):
    input_data: str  # base64 string

@app.get("/")
async def root():
    return {"message": "Welcome to the Tree State API!"}

@app.post("/predict/")
async def get_prediction(data: InputData):
    try:
        input_value = data.input_data
        result = predict_process(input_value)  # nhận base64 string
        return {"prediction": result}
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_file/")
async def predict_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()  # bytes của ảnh upload
        result = predict_process(contents)  # nhận bytes ảnh
        return {"prediction": result}
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
