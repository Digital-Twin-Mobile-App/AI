from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ultils import predict

app = FastAPI()

class InputData(BaseModel): # structure of the input data (JSON format)
    input_data: str

@app.get("/")
async def root():
    return {"message": "Welcome to the Tree State API!"}

@app.post("/predict/")
async def get_prediction(data: InputData):
    try:
        # Process the input data
        input_value = data.input_data
        result = predict(input_value)
        return {"prediction": result}
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)