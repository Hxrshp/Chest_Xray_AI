"""
Logging Configuration
---------------------
Configures application-wide logging formats and severity levels.
"""

import sys
import logging
from typing import Optional


def get_logger(name: str, level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Creates and returns a preconfigured logger instance.

    Args:
        name (str): Name of the logger (typically __name__).
        level (str): Logging level string ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        log_file (Optional[str]): Path to output log file if file logging is desired.

    Returns:
        logging.Logger: Configured Python logging instance.
    """
    logger = logging.getLogger(name)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Avoid adding duplicate handlers if already configured
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console Stream Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File Handler (Optional)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


# Default logger instance for quick imports
logger = get_logger("ChestXRayAI")
