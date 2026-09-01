"""
Central Project Configuration
-----------------------------
Loads and validates environment parameters using python-dotenv with standard library fallback.
"""

import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

# Load environment variables using python-dotenv if available, else standard file parsing
try:
    from dotenv import load_dotenv
    if ENV_PATH.exists():
        load_dotenv(dotenv_path=ENV_PATH)
except ImportError:
    # Basic fallback parsing for .env file if python-dotenv is not yet installed
    if ENV_PATH.exists():
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key not in os.environ:
                        os.environ[key] = val


class Config:
    """Project configuration properties loaded from environment or defaults."""

    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "Chest X-Ray AI Diagnosis System")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    SEED: int = int(os.getenv("SEED", "42"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Directory Paths
    ROOT_DIR: Path = PROJECT_ROOT
    DATA_DIR: Path = PROJECT_ROOT / os.getenv("DATA_DIR", "./data")
    MODELS_DIR: Path = PROJECT_ROOT / os.getenv("MODELS_DIR", "./models")
    REPORTS_DIR: Path = PROJECT_ROOT / os.getenv("REPORTS_DIR", "./reports")

    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres_password@localhost:5432/chest_xray_db"
    )

    # Medical Governance Safety Flag
    MEASUREMENT_STATUS: str = os.getenv("MEASUREMENT_STATUS", "NOT YET MEASURED")

    @classmethod
    def display_summary(cls) -> dict:
        """Returns non-sensitive configuration parameters for logging and inspection."""
        return {
            "Project": cls.PROJECT_NAME,
            "Environment": cls.ENVIRONMENT,
            "Random Seed": cls.SEED,
            "Log Level": cls.LOG_LEVEL,
            "Root Directory": str(cls.ROOT_DIR),
            "Medical Metric Status": cls.MEASUREMENT_STATUS,
        }


config = Config()
