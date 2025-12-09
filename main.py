from Deep_learning_projects.utils import log
from Deep_learning_projects.pipeline.stage01_data_ingestion_pipeline import DataIngestionTrainingPipeline





STAGE_NAME = "Data Ingestion stage"
try:
   log.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   log.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        log.exception(e)
        raise e