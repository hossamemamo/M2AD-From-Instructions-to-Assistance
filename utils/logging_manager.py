import logging
import sys
from datetime import datetime
import os

def setup_logging(log_dir="logs"):
    """Sets up the logging configuration for the project. To be called at the beginning of the script.

    Parameters
    ----------
    log_dir : str, optional
        The directory where the logs will be saved, by default "logs"
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"log_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file), # Save logs to a file
            logging.StreamHandler(sys.stdout) # Print logs to console
        ]
    )