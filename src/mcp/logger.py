# ===============================
# Logger.py
# ===============================

import logging
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("mcp")

logger.setLevel(logging.INFO)

handler = logging.FileHandler(
    LOG_DIR / "mcp.log"
)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
)

handler.setFormatter(formatter)

logger.addHandler(handler)