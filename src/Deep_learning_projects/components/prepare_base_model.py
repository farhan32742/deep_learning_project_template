import os
import shutil
from pathlib import Path
from ultralytics import YOLO
from Deep_learning_projects.utils import log
from Deep_learning_projects.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        """
        Downloads the YOLO model (e.g., yolov8n.pt) from Ultralytics.
        """
        try:
            log.info(f"Downloading/Loading base model: {self.config.model_name}")

            destination_path = Path(self.config.base_model_path)

            # If model already exists at destination, skip download
            if destination_path.exists() and destination_path.stat().st_size > 0:
                log.info(f"Base model already present at {destination_path}. Skipping download.")
                return str(destination_path)

            # Ensure destination directory exists
            destination_path.parent.mkdir(parents=True, exist_ok=True)

            # Trigger YOLO to download the model into working dir
            _ = YOLO(self.config.model_name)  # e.g., 'yolov8n.pt'

            # Ultralytics saves model as '<model_name>.pt' in cwd
            if self.config.model_name.endswith(".pt"):
                source_path = Path(self.config.model_name)
            else:
                source_path = Path(f"{self.config.model_name}.pt")

            # Move/Copy the file to artifacts folder
            if source_path.exists():
                shutil.copy(str(source_path), str(destination_path))
                log.info(f"Base model saved at: {destination_path}")

                # Cleanup: remove the copy in the root dir to keep workspace clean
                try:
                    source_path.unlink()
                except Exception:
                    pass
            else:
                raise FileNotFoundError(f"Could not find downloaded file: {source_path}")

        except Exception as e:
            raise e

    def update_base_model(self):
        try:
            log.info("YOLO does not require manual model updating/compiling.")
            log.info(f"Copying base model to updated path for pipeline consistency.")
            
            shutil.copy(
                self.config.base_model_path, 
                self.config.updated_base_model_path
            )
            
            log.info(f"Updated base model saved at: {self.config.updated_base_model_path}")
            
        except Exception as e:
            raise e
