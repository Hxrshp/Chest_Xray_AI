"""
Foundation Verification Tests
-----------------------------
Sanity checks to verify configuration loading, logger initialization,
random seed setting, and environment safety flags.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.utils.config import config
from ml.utils.logger import get_logger
from ml.utils.seed import set_seed


def test_seed_setting():
    """Verify set_seed produces deterministic outputs."""
    s1 = set_seed(42)
    assert s1 == 42


def test_config_loading():
    """Verify configuration fields exist and safety rule compliance."""
    assert config.PROJECT_NAME is not None
    assert config.SEED == 42
    # Verify strict Medical AI governance safety rule
    assert config.MEASUREMENT_STATUS == "NOT YET MEASURED"


def test_logger_instantiation():
    """Verify custom logger instantiates cleanly without exceptions."""
    logger = get_logger("TestLogger")
    assert logger is not None
    assert logger.name == "TestLogger"


if __name__ == "__main__":
    print("Running foundation verification checks...")
    test_seed_setting()
    test_config_loading()
    test_logger_instantiation()
    print("All foundation verification tests PASSED!")
