import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime
from utils.config import config

def setup_logger(name="fashion_retrieval", log_level=logging.INFO):
    """
    Setup centralized logger with console and file output
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
        
    # Formatters
    console_format = logging.Formatter('%(message)s')  # Simple format for console
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 1. Console Handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # 2. File Handler (Rotating)
    log_dir = config.LOGS_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create unique log file for each session
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"{current_time}.log"
    
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10*1024*1024, # 10 MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger

# Create global logger instance
logger = setup_logger()
