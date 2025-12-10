from Deep_learning_projects.entity.config_entity import (DataIngestionConfig,
                                                         PrepareBaseModelConfig,
                                                         TrainingConfig,
                                                         EvaluationConfig)
from pathlib import Path
import os
from Deep_learning_projects.utils.common import read_yaml, create_directories,save_json
from Deep_learning_projects.constants import *
from Deep_learning_projects.utils import log


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=config.root_dir,
            base_model_path=config.base_model_path,
            updated_base_model_path=config.updated_base_model_path,
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_classes=self.params.CLASSES,
            model_name=config.model_name
        )

        return prepare_base_model_config
    

    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        
        # YOLO specific: We need the path to 'data.yaml', not just the folder.
        # We take this directly from the updated config.yaml
        training_data = training.training_data 

        create_directories([
            Path(training.root_dir)
        ])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE,
            # Added these two to match your updated entity_config.py
            params_classes=params.CLASSES,
            params_learning_rate=params.LEARNING_RATE
        )

        return training_config


    def get_evaluation_config(self) -> EvaluationConfig:
        # 1. Load the evaluation config section we just created in YAML
        eval_config = self.config.evaluation
        
        # 2. Get params
        params = self.params
        create_directories([eval_config.root_dir])
        log.info("checking mlflow uri from env variable")
        mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
        log.info(f"mlflow uri: {mlflow_uri}")

        # 5. Create the Entity
        evaluation_config = EvaluationConfig(
            path_of_model=Path(eval_config.path_of_model),
            training_data=Path(eval_config.training_data),
            mlflow_uri=mlflow_uri,
            all_params=params,
            params_image_size=params.IMAGE_SIZE,
            params_batch_size=params.BATCH_SIZE
        )
        
        return evaluation_config