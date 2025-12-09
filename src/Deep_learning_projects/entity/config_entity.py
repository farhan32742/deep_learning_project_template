from dataclasses import dataclass
import os
from pathlib import Path

@dataclass
class DataIngestionConfig:
    src_url: str
    root_dir: Path
    zip_dir: Path
    data_dir: Path