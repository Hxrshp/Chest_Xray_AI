"""
Seed and Reproducibility Utilities
------------------------------------
Ensures deterministic random seed initialization across Python built-in random,
NumPy, PyTorch CPU, and PyTorch CUDA.
"""

import random
import os


def set_seed(seed: int = 42) -> int:
    """
    Sets the random seed across all key random number generators for strict reproducibility.

    Args:
        seed (int): The integer seed value to use. Defaults to 42.

    Returns:
        int: The seed value that was set.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy seed (if installed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    # PyTorch seed & CUDA deterministic flags (if installed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed


def seed_everything(seed: int = 42) -> int:
    """
    Alias for set_seed for standardized reproducibility calls.
    """
    return set_seed(seed)

