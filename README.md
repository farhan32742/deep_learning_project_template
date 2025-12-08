# Deep Learning Project Template

A comprehensive and production-ready template for deep learning projects with organized structure, configuration management, and MLflow integration.

## 📁 Project Structure

```
deep_learning_project_template/
├── src/
│   └── Deep_learning_projects/
│       ├── __init__.py
│       ├── components/
│       │   ├── __init__.py
│       │   ├── data_ingestion.py          # Data loading and preprocessing
│       │   ├── model_training.py           # Model training logic
│       │   ├── model_evalution_mlflow.py   # Model evaluation with MLflow tracking
│       │   └── prepare_base_model.py       # Base model preparation
│       ├── config/
│       │   ├── __init__.py
│       │   └── configuration.py            # Configuration management
│       ├── constants/
│       │   └── __init__.py                 # Project constants
│       ├── entity/
│       │   ├── __init__.py
│       │   └── config_entity.py            # Configuration entities
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── training_pipeline.py        # Training workflow orchestration
│       │   └── prediction_pipeline.py      # Inference pipeline
│       └── utils/
│           ├── __init__.py
│           └── common.py                   # Utility functions
├── config/
│   └── config.yaml                         # Configuration file
├── research/
│   └── trials.ipynb                        # Jupyter notebook for experimentation
├── templates/
│   └── index.html                          # Web interface template
├── .github/
│   └── workflows/                          # CI/CD workflows
├── dvc.yaml                                # DVC pipeline configuration
├── params.yaml                             # Model parameters
├── requirements.txt                        # Python dependencies
├── setup.py                                # Package setup
├── main.py                                 # Main entry point
├── Dockerfile                              # Docker containerization
├── .env                                    # Environment variables
└── README.md                               # Project documentation
```

## 🚀 Features

- **Modular Architecture**: Organized components for data ingestion, model training, and evaluation
- **Configuration Management**: YAML-based configuration for easy parameter management
- **MLflow Integration**: Built-in model tracking and evaluation with MLflow
- **DVC Support**: Data version control for managing datasets and pipelines
- **Docker Support**: Containerization for reproducible environments
- **Jupyter Notebooks**: Research directory for experimentation
- **Web Interface**: HTML templates for model deployment
- **CI/CD Ready**: GitHub workflows directory for automation

## 📋 Requirements

Ensure you have Python 3.8+ installed. Install dependencies using:

```bash
pip install -r requirements.txt
```

## 🔧 Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/farhan32742/deep_learning_project_template.git
   cd deep_learning_project_template
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your project**:
   - Edit `config/config.yaml` with your project settings
   - Update `params.yaml` with model hyperparameters
   - Set environment variables in `.env` file

## 🏃 Usage

### Training Pipeline

Run the complete training pipeline:

```bash
python main.py
```

Or execute directly:

```python
from src.Deep_learning_projects.pipeline.training_pipeline import TrainingPipeline

pipeline = TrainingPipeline()
pipeline.main()
```

### Prediction Pipeline

Use the prediction pipeline for inference:

```python
from src.Deep_learning_projects.pipeline.prediction_pipeline import PredictionPipeline

predictor = PredictionPipeline()
predictions = predictor.predict(data)
```

### Experimentation

Use the Jupyter notebook for experimentation:

```bash
jupyter notebook research/trials.ipynb
```

## 📊 Components

### Data Ingestion (`components/data_ingestion.py`)
- Handles data loading from various sources
- Data validation and preprocessing

### Model Training (`components/model_training.py`)
- Model architecture definition
- Training loop implementation
- Checkpoint saving

### Model Evaluation (`components/model_evalution_mlflow.py`)
- Performance metrics calculation
- MLflow tracking integration
- Model logging and versioning

### Base Model Preparation (`components/prepare_base_model.py`)
- Transfer learning model setup
- Pre-trained model loading
- Model architecture modification

## 📝 Configuration

### config.yaml
Define your project-specific configurations:
```yaml
data:
  path: "data/"
  train_size: 0.8
  
model:
  architecture: "resnet50"
  pretrained: true
  
training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
```

### params.yaml
Store model hyperparameters for DVC tracking:
```yaml
learning_rate: 0.001
epochs: 100
batch_size: 32
```

## 🐳 Docker

Build and run the project using Docker:

```bash
docker build -t deep-learning-project .
docker run -it deep-learning-project
```

## 📦 DVC Pipeline

Track and reproduce experiments with DVC:

```bash
dvc repro dvc.yaml
```

## 🔄 MLflow Tracking

Monitor experiments with MLflow:

```bash
mlflow ui
```

## 🛠️ Development

### Project Structure Generation

Generate the project structure automatically:

```bash
python template.py
```

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📧 Contact

For questions or collaboration, reach out to the project maintainers.

---

**Happy Deep Learning! 🚀**
