import os
import yaml
import shutil
from pathlib import Path
from dotenv import load_dotenv  # <--- NEW: Loads .env variables (URI, User, Password)
from Deep_learning_projects.utils import log # <--- Using 'log' as you requested
from ultralytics import YOLO
from Deep_learning_projects.entity.config_entity import TrainingConfig
from dotenv import load_dotenv
# Load environment variables from .env file immediately
load_dotenv()

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None

    def get_base_model(self):
        """
        Loads the YOLO model (yolov8n.pt) prepared in the previous stage.
        """
        self.model = YOLO(self.config.updated_base_model_path)

    def update_data_yaml_paths(self):
        """
        Rewrites 'train', 'val', and 'test' paths in data.yaml 
        to absolute paths on the current machine.
        """
        data_yaml_path = self.config.training_data
        
        # 1. Read the existing YAML
        with open(data_yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        # 2. Get the root directory where data.yaml is located
        dataset_root = os.path.dirname(data_yaml_path)

        # 3. Update paths to be Absolute Paths
        data['train'] = os.path.abspath(os.path.join(dataset_root, 'train'))
        data['val'] = os.path.abspath(os.path.join(dataset_root, 'valid')) 
        
        # Check for 'val' vs 'valid' naming convention
        if not os.path.exists(data['val']):
            val_path_alternative = os.path.abspath(os.path.join(dataset_root, 'val'))
            if os.path.exists(val_path_alternative):
                data['val'] = val_path_alternative

        # Handle Test set
        test_path = os.path.join(dataset_root, 'test')
        if os.path.exists(test_path):
            data['test'] = os.path.abspath(test_path)
        else:
            # Fallback to val if test doesn't exist
            data['test'] = data['val'] 

        # 4. Enforce class count from params.yaml
        data['nc'] = self.config.params_classes

        # 5. Write back to data.yaml
        with open(data_yaml_path, 'w') as f:
            yaml.dump(data, f)
            
        log.info(f"Updated data.yaml paths to absolute paths at: {dataset_root}")

    def train(self):
        """
        Executes YOLO training with MLflow logging enabled.
        """
        # 1. Fix the paths in data.yaml
        self.update_data_yaml_paths()

        # 2. Define output directory
        project_dir = self.config.root_dir # artifacts/training
        run_name = "yolo_run"

        # 3. Train the model
        # YOLO will automatically detect the MLFLOW variables loaded by load_dotenv()
        self.model.train(
            data=self.config.training_data,
            epochs=self.config.params_epochs,
            batch=self.config.params_batch_size,
            imgsz=self.config.params_image_size[0], 
            lr0=self.config.params_learning_rate,
            project=project_dir,
            name=run_name,
            exist_ok=True, 
            augment=self.config.params_is_augmentation 
        )

        # 4. Copy the best model to the simplified path
        generated_weight_path = os.path.join(project_dir, run_name, "weights", "best.pt")
        
        if os.path.exists(generated_weight_path):
            shutil.copy(generated_weight_path, self.config.trained_model_path)
            log.info(f"Final model copied to: {self.config.trained_model_path}")
        else:
            log.error(f"Training completed but could not find 'best.pt' at {generated_weight_path}")