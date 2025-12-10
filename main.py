from Deep_learning_projects.utils import log
from Deep_learning_projects.pipeline.stage01_data_ingestion_pipeline import DataIngestionTrainingPipeline
from Deep_learning_projects.pipeline.stage02_prepare_base_model import PrepareBaseModelTrainingPipeline
from Deep_learning_projects.pipeline.stage03_model_training_pipeline import ModelTrainingPipeline
from Deep_learning_projects.pipeline.stage04_evalution_pipeline import EvaluationPipeline
from dotenv import load_dotenv
load_dotenv()



STAGE_NAME = "Data Ingestion stage"
try:
   log.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   log.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        log.exception(e)
        raise e



STAGE_NAME = "Prepare base model"
try: 
   log.info(f"*******************")
   log.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   prepare_base_model = PrepareBaseModelTrainingPipeline()
   prepare_base_model.main()
   log.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        log.exception(e)
        raise e


STAGE_NAME = "Training"
try: 
   log.info(f"*******************")
   log.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_trainer = ModelTrainingPipeline()
   model_trainer.main()
   log.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
         log.exception(e)
         raise e



STAGE_NAME = "Evaluation stage"
try:
   log.info(f"*******************")
   log.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_evalution = EvaluationPipeline()
   model_evalution.main()
   log.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
        log.exception(e)
        raise e
