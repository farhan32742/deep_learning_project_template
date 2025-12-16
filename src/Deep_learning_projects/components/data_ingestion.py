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
            zip_download_path = Path(self.config.local_data_file)
            unzip_path = Path(self.config.unzip_dir)

            # If extracted data already exists (checking key file 'data.yaml'), skip download and extraction
            if (unzip_path / "data.yaml").exists():
                log.info(f"Data already present at {unzip_path}. Skipping download.")
                return str(unzip_path)

            # Ensure parent dir for zip exists
            zip_download_path.parent.mkdir(parents=True, exist_ok=True)

            # If zip already downloaded, just extract
            if zip_download_path.exists():
                log.info(f"Zip file already exists at {zip_download_path}. Extracting.")
                self.extract_zip_file()
                return str(unzip_path)

            log.info(f"Downloading data from {dataset_url} into file {zip_download_path}")

            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix + file_id, str(zip_download_path))

            log.info(f"Downloaded data from {dataset_url} into file {zip_download_path}")
            self.extract_zip_file()
            return str(unzip_path)

        except Exception as e:
            raise e
        
    

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = Path(self.config.unzip_dir)
        unzip_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        log.info(f"Extracted zip file to {unzip_path}")





