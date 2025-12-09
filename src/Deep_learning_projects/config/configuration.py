from Deep_learning_projects.entity.config_entity import DataIngestionConfig
from pathlib import Path
import os
from Deep_learning_projects.utils.common import read_yaml, create_directories,save_json
from Deep_learning_projects.constants import *
class ConfigurationManager:
    def __init__(self, config_file_path: CONFIG_FILE_PATH,
                  params_file_path: PARAMS_FILE_PATH):
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)
        create_directories([self.config.artifact_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        data_ingestion_config = DataIngestionConfig(
            src_url=config.src_url,
            root_dir=config.root_dir,
            zip_dir=config.zip_dir,
            data_dir=config.data_dir
        )
        return data_ingestion_config