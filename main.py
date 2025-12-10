import os
import sys
import dagshub
import mlflow
from urllib.parse import urlparse
from dotenv import load_dotenv, find_dotenv

from Deep_learning_projects.utils import log
from Deep_learning_projects.pipeline.stage01_data_ingestion_pipeline import DataIngestionTrainingPipeline
from Deep_learning_projects.pipeline.stage02_prepare_base_model import PrepareBaseModelTrainingPipeline
from Deep_learning_projects.pipeline.stage03_model_training_pipeline import ModelTrainingPipeline
from Deep_learning_projects.pipeline.stage04_evalution_pipeline import EvaluationPipeline

# -------------------------------------------------------------------------
# 1. ROBUST INITIALIZATION FUNCTION
# -------------------------------------------------------------------------
def init_dagshub_mlflow():
    """
    Dynamically initializes DagsHub and MLflow using the .env file.
    This removes the need to hardcode repo_owner and repo_name.
    """
    # Force load .env
    load_dotenv(find_dotenv(), override=True)
    
    uri = os.getenv("MLFLOW_TRACKING_URI")
    
    if not uri:
        log.warning("MLFLOW_TRACKING_URI not found in .env. Skipping DagsHub init.")
        return

    try:
        # Parse the URI to get details automatically
        # Example URI: https://dagshub.com/farhanfiaz79/deep_learning_project_template.mlflow
        parsed = urlparse(uri)
        path_parts = parsed.path.strip('/').split('/')
        
        # logic: path_parts[0] is owner, path_parts[1] is repo.mlflow
        if len(path_parts) >= 2:
            owner = path_parts[0]
            repo = path_parts[1].replace('.mlflow', '')
            
            log.info(f"Initializing DagsHub for Owner: {owner}, Repo: {repo}")
            
            dagshub.init(repo_owner=owner, repo_name=repo, mlflow=True)
            mlflow.set_tracking_uri(uri)
            
            # REMOVED EMOJI TO PREVENT WINDOWS CRASH
            log.info("[OK] DagsHub & MLflow initialized successfully.")
            
        else:
            log.warning("Could not parse DagsHub URI. Using local MLflow.")

    except Exception as e:
        log.error(f"Failed to initialize DagsHub: {e}")
        pass

# -------------------------------------------------------------------------
# 2. RUN INITIALIZATION
# -------------------------------------------------------------------------
init_dagshub_mlflow()


# -------------------------------------------------------------------------
# 3. PIPELINE STAGES
# -------------------------------------------------------------------------

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