from fastapi import FastAPI, BackgroundTasks, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
import uvicorn
import yaml
import os
import shutil
from typing import Dict, Any, Optional
from pydantic import BaseModel

from Deep_learning_projects.utils import log
from Deep_learning_projects.pipeline.prediction_pipeline import PredictionPipeline
from Deep_learning_projects.pipeline.stage01_data_ingestion_pipeline import DataIngestionTrainingPipeline
from Deep_learning_projects.pipeline.stage02_prepare_base_model import PrepareBaseModelTrainingPipeline
from Deep_learning_projects.pipeline.stage03_model_training_pipeline import ModelTrainingPipeline

app = FastAPI()

PARAMS_PATH = "params.yaml"
CONFIG_PATH = "config/config.yaml"

# --- Models ---
class TrainingConfig(BaseModel):
    # params.yaml
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    augmentation: Optional[bool] = None
    
    # config.yaml (model selection)
    model_name: Optional[str] = None # e.g., 'yolov8n', 'yolo12n'

# --- Utils ---
def read_yaml(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}

def write_yaml(path: str, content: Dict[str, Any]):
    with open(path, 'w') as f:
        yaml.safe_dump(content, f, default_flow_style=False)

def run_training_pipeline():
    try:
        log.info(">>> Training Pipeline Triggered by API <<<")
        
        # 1. Data Ingestion (Optional if data already exists, but safer to run)
        log.info(">> Stage 1: Data Ingestion")
        DataIngestionTrainingPipeline().main()
        
        # 2. Prepare Base Model (Crucial if model_name changed)
        log.info(">> Stage 2: Prepare Base Model")
        PrepareBaseModelTrainingPipeline().main()
        
        # 3. Training
        log.info(">> Stage 3: Training")
        ModelTrainingPipeline().main()
        
        log.info(">>> Training Pipeline Completed Successfully <<<")
    except Exception as e:
        log.exception(f"Training Pipeline Failed: {e}")

# --- Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <head><title>YOLO Training API</title></head>
        <body>
            <h1>YOLO Training API</h1>
            <ul>
                <li>POST /predict : Inference</li>
                <li>GET /train/config : Get current config</li>
                <li>POST /train/config : Update config</li>
                <li>POST /train/start : Start training</li>
            </ul>
        </body>
    </html>
    """

@app.post("/predict")
async def predict_route(file: UploadFile = File(...)):
    # Reuse logic from app.py
    try:
        os.makedirs("uploads", exist_ok=True)
        file_path = os.path.join("uploads", file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        pipeline = PredictionPipeline()
        predictions = pipeline.predict(file_path)
        
        if os.path.exists(file_path):
            os.remove(file_path)
            
        return JSONResponse(content={"filename": file.filename, "predictions": predictions})
        
    except Exception as e:
        log.exception(e)
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/train/config")
async def get_config():
    try:
        params = read_yaml(PARAMS_PATH)
        config = read_yaml(CONFIG_PATH)
        
        # Extract relevant fields
        response_data = {
            "epochs": params.get("EPOCHS"),
            "batch_size": params.get("BATCH_SIZE"),
            "learning_rate": params.get("LEARNING_RATE"),
            "augmentation": params.get("AUGMENTATION"),
            "model_name": config.get("prepare_base_model", {}).get("model_name")
        }
        return JSONResponse(content=response_data)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/train/config")
async def update_config(cfg: TrainingConfig):
    try:
        params = read_yaml(PARAMS_PATH)
        config = read_yaml(CONFIG_PATH)
        
        # Update params.yaml
        if cfg.epochs is not None: params["EPOCHS"] = cfg.epochs
        if cfg.batch_size is not None: params["BATCH_SIZE"] = cfg.batch_size
        if cfg.learning_rate is not None: params["LEARNING_RATE"] = cfg.learning_rate
        if cfg.augmentation is not None: params["AUGMENTATION"] = cfg.augmentation
        
        write_yaml(PARAMS_PATH, params)
        
        # Update config.yaml for model_name
        if cfg.model_name is not None:
            if "prepare_base_model" in config:
                # Sanitize model name
                model_name = cfg.model_name.strip()
                
                # If user just gives "yolov8", default to "yolov8n.pt"
                if model_name.lower().endswith("yolov8"):
                     model_name = model_name + "n.pt"
                # If user gives "yolov8n", ensure it has ".pt"
                elif not model_name.endswith(".pt"):
                     model_name = model_name + ".pt"
                     
                config["prepare_base_model"]["model_name"] = model_name
                
                # Also update the base_model_path to match the new model name
                # We assume the root_dir is already there, just replacing the filename
                root_dir = config["prepare_base_model"]["root_dir"]
                config["prepare_base_model"]["base_model_path"] = os.path.join(root_dir, model_name)
                
                write_yaml(CONFIG_PATH, config)
            else:
                 raise HTTPException(status_code=500, detail="Invalid config structure: 'prepare_base_model' not found")

        return JSONResponse(content={"message": "Configuration updated successfully", "new_config": cfg.dict(exclude_none=True)})
        
    except Exception as e:
        log.exception(e)
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/train/start")
async def start_training(background_tasks: BackgroundTasks):
    # Verify we can modify files? pipeline does that.
    background_tasks.add_task(run_training_pipeline)
    return {"message": "Training started in background. Check logs for progress."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
