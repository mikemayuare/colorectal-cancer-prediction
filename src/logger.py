import logging
from datetime import datetime
from pathlib import Path

LOGS_DIR = Path("logs")
Path.mkdir(LOGS_DIR, exist_ok=True)

LOG_FILE = Path(f"{LOGS_DIR}/log_{datetime.now().strftime('%Y-%m-%d')}.log")

logging.basicConfig(
    filename=LOG_FILE,
    format="%(asctime)s - %(levelname)s - [%(filename)s] - %(message)s",
    level=logging.INFO,
)


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger
