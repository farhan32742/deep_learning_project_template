import os
import yaml
import shutil
from pathlib import Path
from Deep_learning_projects.utils import log
from ultralytics import YOLO
from Deep_learning_projects.entity.config_entity import TrainingConfig

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None

    def get_base_model(self):
        """
        Loads the YOLO model (yolov8n.pt) prepared in the previous stage.
        """
        # We load the weights we saved in artifacts/prepare_base_model/best.pt
        self.model = YOLO(self.config.updated_base_model_path)

    def update_data_yaml_paths(self):
        """
        CRITICAL FUNCTION:
        Downloaded datasets often have hardcoded paths in data.yaml (e.g., /content/drive/...).
        This function rewrites 'train', 'val', and 'test' paths in data.yaml 
        to absolute paths on YOUR current machine.
        """
        data_yaml_path = self.config.training_data
        
        # 1. Read the existing YAML
        with open(data_yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        # 2. Get the root directory where data.yaml is located
        # Example: artifacts/data_ingestion
        dataset_root = os.path.dirname(data_yaml_path)

        # 3. Update paths to be Absolute Paths based on current location
        # Assumption: The 'train', 'valid', 'test' folders are in the same dir as data.yaml
        data['train'] = os.path.abspath(os.path.join(dataset_root, 'train'))
        data['val'] = os.path.abspath(os.path.join(dataset_root, 'valid')) 
        
        # Some datasets name it 'valid', some 'val'. Check which one exists.
        if not os.path.exists(data['val']):
            # If 'valid' folder doesn't exist, try 'val'
            val_path_alternative = os.path.abspath(os.path.join(dataset_root, 'val'))
            if os.path.exists(val_path_alternative):
                data['val'] = val_path_alternative

        # Handle Test set (optional in some datasets)
        test_path = os.path.join(dataset_root, 'test')
        if os.path.exists(test_path):
            data['test'] = os.path.abspath(test_path)
        else:
            # If no test set, point test to validation just to satisfy YOLO requirements
            data['test'] = data['val'] 

        # 4. Enforce class count from params.yaml (Safety Check)
        data['nc'] = self.config.params_classes
        # You can also optionally update names if you have them in params, 
        # but usually we trust the dataset's names.

        # 5. Write back to data.yaml
        with open(data_yaml_path, 'w') as f:
            yaml.dump(data, f)
            
        log.info(f"Updated data.yaml paths to absolute paths at: {dataset_root}")

    def train(self):
        """
        Executes YOLO training.
        """
        # 1. Fix the paths in data.yaml before passing it to YOLO
        self.update_data_yaml_paths()

        # 2. Define output directory for this specific run
        project_dir = self.config.root_dir # artifacts/training
        run_name = "yolo_run"

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

        # 4. Copy the best model to the simplified path defined in config.yaml
        # YOLO saves result at: artifacts/training/yolo_run/weights/best.pt
        generated_weight_path = os.path.join(project_dir, run_name, "weights", "best.pt")
        
        if os.path.exists(generated_weight_path):
            shutil.copy(generated_weight_path, self.config.trained_model_path)
            log.info(f"Final model copied to: {self.config.trained_model_path}")
        else:
            log.error(f"Training completed but could not find 'best.pt' at {generated_weight_path}")