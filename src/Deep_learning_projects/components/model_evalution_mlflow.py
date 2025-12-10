import os
import mlflow
import mlflow.pytorch # Changed from mlflow.keras to pytorch for YOLO
from pathlib import Path
from urllib.parse import urlparse
from ultralytics import YOLO
from Deep_learning_projects.utils.common import save_json
from Deep_learning_projects.entity.config_entity import EvaluationConfig
from Deep_learning_projects.utils import log
class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def evaluation(self):
        """
        Loads the YOLO model and runs validation on the test/val set defined in data.yaml.
        """
        # 1. Load the model (best.pt)
        self.model = YOLO(self.config.path_of_model)
        
        # 2. Run Validation
        # YOLO's val() method automatically handles data loading using the 'data.yaml' file.
        # We don't need ImageDataGenerator anymore.
        self.results = self.model.val(
            data=self.config.training_data,
            imgsz=self.config.params_image_size[0],
            batch=self.config.params_batch_size,
            split='test' # Validates on the 'val' split defined in data.yaml
        )

        # 3. Extract Metrics
        # Unlike Keras (loss, acc), YOLO Object Detection uses Mean Average Precision (mAP)
        self.score = {
            "mAP_50": self.results.box.map50,       # mAP at IoU=0.50
            "mAP_50_95": self.results.box.map,      # mAP at IoU=0.50:0.95
            "precision": self.results.box.mp,       # Mean Precision
            "recall": self.results.box.mr           # Mean Recall
        }
        
        # 4. Save to local JSON
        self.save_score()

    def save_score(self):
        """
        Saves the metrics dictionary to scores.json
        """
        save_json(path=Path("scores.json"), data=self.score)

    def log_into_mlflow(self):
        """
        Logs parameters, metrics, and the model to MLflow.
        """
        # Set the URI (DagsHub or Local)
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            # 1. Log Hyperparameters (from params.yaml)
            mlflow.log_params(self.config.all_params)
            
            # 2. Log Metrics (mAP, Precision, Recall)
            mlflow.log_metrics(self.score)
            
            # 3. Log Model
            # YOLO is PyTorch based, so we use mlflow.pytorch
            if tracking_url_type_store != "file":
                mlflow.pytorch.log_model(
                    self.model, 
                    "model", 
                    registered_model_name="YOLOv8Model"
                )
            else:
                mlflow.pytorch.log_model(self.model, "model")