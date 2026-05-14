import logging
import os

def setup_logger(name: str, log_file: str, level=logging.INFO):
    # ensure directories exist for log outputs
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # file handler for persistent tracking
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    # console handler for live monitoring
    console = logging.StreamHandler()
    console.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(console)

    return logger