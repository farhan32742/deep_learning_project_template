import os
import zipfile
import gdown
import yaml
from pathlib import Path
from typing import Optional

from Deep_learning_projects.utils import log
from Deep_learning_projects.utils.common import get_size
from Deep_learning_projects.entity.config_entity import (DataIngestionConfig)



class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    
    def download_file(self)-> str:
        '''
        Fetch data from the url
        '''

        try: 
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            log.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix+file_id,zip_download_dir)

            log.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

        except Exception as e:
            raise e
        
    

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)

    def save_data_splits_to_yaml(
    train_dir: Path, test_dir: Path, valid_dir: Path, dest_path: Optional[Path] = None
    ) -> Path:

        try:
            train_p = Path(train_dir)
            test_p = Path(test_dir)
            valid_p = Path(valid_dir)

            if dest_path is None:
                repo_root = Path(__file__).resolve().parents[3]
                artifacts_dir = repo_root / "artifacts"
                artifacts_dir.mkdir(parents=True, exist_ok=True)
                dest_path = artifacts_dir / "data.yaml"
            else:
                dest_path = Path(dest_path)
                dest_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "train": str(train_p),
                "test": str(test_p),
                "valid": str(valid_p),
            }

            with open(dest_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f)

            log.info(f"Saved data splits to YAML: {dest_path}")
            return dest_path

        except Exception as e:
            log.error(f"Failed to save data splits to YAML: {e}")
            raise




