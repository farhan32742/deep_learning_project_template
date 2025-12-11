import os
import yaml
import shutil
import mlflow
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from Deep_learning_projects.utils import log
from ultralytics import YOLO
from Deep_learning_projects.entity.config_entity import TrainingConfig

# 1. Force load .env
load_dotenv(find_dotenv())

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None

        # 2. CREDENTIAL CLEANING (The Fix)
        username = os.getenv("MLFLOW_TRACKING_USERNAME")
        password = os.getenv("MLFLOW_TRACKING_PASSWORD")
        uri = os.getenv("MLFLOW_TRACKING_URI")

        if password:
            # Aggressively remove quotes and whitespace
            clean_password = password.strip().replace('"', '').replace("'", '')
            
            # Update the environment variable with the clean version
            os.environ["MLFLOW_TRACKING_PASSWORD"] = clean_password
            
            log.info(f"Credentials loaded for user: {username}")
            log.info(f"Original Password Length: {len(password)}")
            log.info(f"Cleaned Password Length:  {len(clean_password)}")
            
            # 3. Explicitly set MLflow configuration
            mlflow.set_tracking_uri(uri)
        else:
            log.info("MLFLOW_TRACKING_PASSWORD is missing from .env")

    def get_base_model(self):
        self.model = YOLO(self.config.updated_base_model_path)

    def update_data_yaml_paths(self):
        data_yaml_path = self.config.training_data
        
        with open(data_yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        dataset_root = os.path.dirname(data_yaml_path)

        data['train'] = os.path.abspath(os.path.join(dataset_root, 'train'))
        data['val'] = os.path.abspath(os.path.join(dataset_root, 'valid')) 
        
        if not os.path.exists(data['val']):
            val_path_alternative = os.path.abspath(os.path.join(dataset_root, 'val'))
            if os.path.exists(val_path_alternative):
                data['val'] = val_path_alternative

        test_path = os.path.join(dataset_root, 'test')
        if os.path.exists(test_path):
            data['test'] = os.path.abspath(test_path)
        else:
            data['test'] = data['val'] 

        data['nc'] = self.config.params_classes

        with open(data_yaml_path, 'w') as f:
            yaml.dump(data, f)
            
        log.info(f"Updated data.yaml paths to absolute paths at: {dataset_root}")

    def train(self):
        self.update_data_yaml_paths()

        project_dir = self.config.root_dir 
        run_name = "yolo_run"

        # YOLO will now use the CLEANED password from os.environ
        self.model.train(
            data=str(self.config.training_data),
            epochs=self.config.params_epochs,
            batch=self.config.params_batch_size,
            imgsz=self.config.params_image_size[0], 
            lr0=self.config.params_learning_rate,
            project=str(project_dir),
            name=run_name,
            exist_ok=True, 
            augment=self.config.params_is_augmentation 
        )

        generated_weight_path = os.path.join(project_dir, run_name, "weights", "best.pt")
        
        if os.path.exists(generated_weight_path):
            shutil.copy(generated_weight_path, self.config.trained_model_path)
            log.info(f"Final model copied to: {self.config.trained_model_path}")
        else:
            log.error(f"Training completed but could not find 'best.pt' at {generated_weight_path}")