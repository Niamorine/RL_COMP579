import logging
import os
from datetime import datetime

def setup_logging(log_dir="logs", filename_prefix="output"):
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(log_dir, f"{filename_prefix}_{timestamp}.log")

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()  # This prints to console
        ]
    )
    logging.info(f"Logging started. Output will be written to: {log_path}")
