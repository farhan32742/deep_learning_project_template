from Deep_learning_projects.config.configuration import ConfigurationManager
from Deep_learning_projects.components.model_training import Training
from Deep_learning_projects.utils import log
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)
STAGE_NAME = "Training"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        
        # 1. Load the pre-trained weights (yolov8n.pt)
        training.get_base_model()
        
        # 2. REMOVED: training.train_valid_generator() 
        # YOLO does not use Keras generators. It reads directly from data.yaml during train().
        
        # 3. Start Training (Includes updating data.yaml paths internally)
        training.train()


if __name__ == '__main__':
    try:
        log.info(f"*******************")
        log.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        
        # Debug Check for DVC
        if not os.getenv("MLFLOW_TRACKING_PASSWORD"):
            log.warning("⚠️  DVC Warning: MLFLOW_TRACKING_PASSWORD not found in environment!")
        else:
            log.info("✅ DVC successfully loaded .env credentials.")

        obj = ModelTrainingPipeline()
        obj.main()
        log.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        log.exception(e)
        raise e