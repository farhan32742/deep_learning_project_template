from Deep_learning_projects.config.configuration import ConfigurationManager
from Deep_learning_projects.components.prepare_base_model import PrepareBaseModel
from Deep_learning_projects.utils import log

STAGE_NAME = "Prepare base model"


class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        
        # 1. This now downloads 'yolov8n.pt' from Ultralytics
        prepare_base_model.get_base_model()
        
        # 2. This now simply copies 'yolov8n.pt' to 'artifacts/prepare_base_model/base_model_updated.pt'
        #    (It no longer compiles/freezes layers since YOLO doesn't need that here)
        prepare_base_model.update_base_model()


    
if __name__ == '__main__':
    try:
        log.info(f"*******************")
        log.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        log.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        log.exception(e)
        raise e