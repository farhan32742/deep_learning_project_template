from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_classes: int
    model_name: str

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list
    params_classes: int         # <--- ADDED: To validate dataset against params.yaml
    params_learning_rate: float


@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path       # Path to artifacts/training/best.pt
    training_data: Path       # Path to data.yaml
    all_params: dict          # param.yaml content
    mlflow_uri: str           # MLflow tracking URI
    params_image_size: list   # [640, 640, 3]
    params_batch_size: int 
