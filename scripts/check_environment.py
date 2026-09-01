"""
Pre-Flight Environment Check Script
------------------------------------
Executes real empirical checks against the Python runtime and dependencies.
"""

import sys
import os
import importlib
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_checks():
    results = {}

    # 1. Python Version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    is_py311 = (sys.version_info.major == 3 and sys.version_info.minor == 11)
    results['python'] = {
        'version': py_version,
        'is_311': is_py311,
        'executable': sys.executable
    }

    # 2. Virtual Environment
    venv_dir = PROJECT_ROOT / ".venv"
    in_venv = (sys.prefix != sys.base_prefix) or ("venv" in sys.executable.lower())
    results['venv'] = {
        'exists': venv_dir.exists(),
        'path': str(venv_dir),
        'currently_active': in_venv
    }

    # Helper function for checking imports
    def check_package(module_name, import_name=None):
        target = import_name or module_name
        try:
            mod = importlib.import_module(target)
            ver = getattr(mod, '__version__', 'Installed (no __version__ attr)')
            return True, str(ver), None
        except Exception as e:
            return False, "Not Installed", str(e)

    # 3. PyTorch
    torch_installed, torch_ver, torch_err = check_package("torch")
    results['torch'] = {'installed': torch_installed, 'version': torch_ver, 'error': torch_err}

    # 4. Torchvision
    tv_installed, tv_ver, tv_err = check_package("torchvision")
    results['torchvision'] = {'installed': tv_installed, 'version': tv_ver, 'error': tv_err}

    # 5. NumPy
    np_installed, np_ver, np_err = check_package("numpy")
    results['numpy'] = {'installed': np_installed, 'version': np_ver, 'error': np_err}

    # 6. Pandas
    pd_installed, pd_ver, pd_err = check_package("pandas")
    results['pandas'] = {'installed': pd_installed, 'version': pd_ver, 'error': pd_err}

    # 7. scikit-learn
    sk_installed, sk_ver, sk_err = check_package("scikit-learn", "sklearn")
    results['sklearn'] = {'installed': sk_installed, 'version': sk_ver, 'error': sk_err}

    # 8. Pillow
    pil_installed, pil_ver, pil_err = check_package("Pillow", "PIL")
    results['pil'] = {'installed': pil_installed, 'version': pil_ver, 'error': pil_err}

    # 9. FastAPI
    fa_installed, fa_ver, fa_err = check_package("fastapi")
    results['fastapi'] = {'installed': fa_installed, 'version': fa_ver, 'error': fa_err}

    # 10. PostgreSQL Configuration check
    from ml.utils.config import config
    db_config_exists = bool(config.DATABASE_URL)
    results['postgres_config'] = {
        'exists': db_config_exists,
        'user': os.getenv("POSTGRES_USER", "postgres"),
        'host': os.getenv("POSTGRES_HOST", "localhost"),
        'port': os.getenv("POSTGRES_PORT", "5432"),
        'database': os.getenv("POSTGRES_DB", "chest_xray_db")
    }

    # 11. GPU / CUDA
    if torch_installed:
        import torch
        cuda_available = torch.cuda.is_available()
        results['cuda'] = {
            'available': cuda_available,
            'device_count': torch.cuda.device_count() if cuda_available else 0,
            'device_name': torch.cuda.get_device_name(0) if cuda_available else "N/A (CPU Mode)",
            'cuda_version': torch.version.cuda if cuda_available else "N/A"
        }
    else:
        results['cuda'] = {
            'available': False,
            'device_count': 0,
            'device_name': "N/A (Torch not installed)",
            'cuda_version': "N/A"
        }

    # 12 & 13. DenseNet-121 and Synthetic Inference
    if torch_installed and tv_installed:
        try:
            import torch
            import torch.nn as nn
            import torchvision.models as models

            # Instantiate without pretrained weights
            model = models.densenet121(weights=None)
            
            # Replace classifier with 5 outputs
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, 5)
            model.eval()

            # Synthetic forward pass
            dummy_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                output = model(dummy_input)

            results['densenet'] = {
                'instantiated': True,
                'input_shape': list(dummy_input.shape),
                'output_shape': list(output.shape),
                'passed_synthetic_test': (list(output.shape) == [1, 5])
            }
        except Exception as e:
            results['densenet'] = {
                'instantiated': False,
                'error': str(e),
                'passed_synthetic_test': False
            }
    else:
        results['densenet'] = {
            'instantiated': False,
            'error': "PyTorch or Torchvision missing",
            'passed_synthetic_test': False
        }

    # 14. Data & Medical AI governance check
    data_raw = PROJECT_ROOT / "data" / "raw"
    data_processed = PROJECT_ROOT / "data" / "processed"
    raw_files = [f.name for f in data_raw.iterdir() if f.name != ".gitkeep"] if data_raw.exists() else []
    processed_files = [f.name for f in data_processed.iterdir() if f.name != ".gitkeep"] if data_processed.exists() else []

    results['governance'] = {
        'no_dataset_downloaded': len(raw_files) == 0 and len(processed_files) == 0,
        'measurement_status': config.MEASUREMENT_STATUS,
        'raw_file_count': len(raw_files),
        'processed_file_count': len(processed_files)
    }

    return results


if __name__ == "__main__":
    import json
    res = run_checks()
    print("=== ENVIRONMENT CHECK JSON OUTPUT ===")
    print(json.dumps(res, indent=2))
