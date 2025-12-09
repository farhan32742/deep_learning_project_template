import logging
import os
from logging.handlers import RotatingFileHandler

# Create logs directory if it doesn't exist
logs_dir = os.path.join(os.path.dirname(__file__), '../../..', 'logs')
os.makedirs(logs_dir, exist_ok=True)

# Configure logging
log = logging.getLogger('deep_learning_projects')
log.setLevel(logging.DEBUG)

# Create formatters
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# File handler with rotation
log_file = os.path.join(logs_dir, 'app.log')
file_handler = RotatingFileHandler(
    log_file,
    maxBytes=10485760,  # 10MB
    backupCount=5
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Add handlers to logger
log.addHandler(file_handler)
log.addHandler(console_handler)

__all__ = ['log']
