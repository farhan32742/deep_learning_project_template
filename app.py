from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
import uvicorn
import os
import shutil
from Deep_learning_projects.utils import log
import ultralytics
from Deep_learning_projects.pipeline.prediction_pipeline import PredictionPipeline

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <head>
            <title>YOLO Prediction API</title>
        </head>
        <body>
            <h1>Welcome to YOLO Prediction API</h1>
            <p>Use <a href="/docs">/docs</a> to test the endpoints.</p>
            <ul>
                <li>POST /predict : Upload an image to get object detection results.</li>
            </ul>
        </body>
    </html>
    """

@app.post("/predict")
async def predict_route(file: UploadFile = File(...)):
    try:
        # Create uploads directory if not exists
        os.makedirs("uploads", exist_ok=True)
        
        file_path = os.path.join("uploads", file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        pipeline = PredictionPipeline()
        predictions = pipeline.predict(file_path)
        
        # Clean up
        if os.path.exists(file_path):
            os.remove(file_path)
            
        return JSONResponse(content={"filename": file.filename, "predictions": predictions})
        
    except Exception as e:
        log.exception(e)
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
