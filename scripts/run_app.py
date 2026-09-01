"""
Phase 8P — Application Launcher Script
--------------------------------------
CLI tool to launch the Streamlit research web application.

Usage:
    python scripts/run_app.py
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def launch():
    main_py = PROJECT_ROOT / "app" / "main.py"
    print("==================================================")
    print("LAUNCHING CHEST X-RAY AI RESEARCH WEB APP")
    print("==================================================")
    print(f"App Path: {main_py}")
    
    cmd = [sys.executable, "-m", "streamlit", "run", str(main_py)]
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(PROJECT_ROOT))


if __name__ == "__main__":
    launch()
