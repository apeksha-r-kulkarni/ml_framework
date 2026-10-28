"""
Alias entrypoint for model_test_runner.
Enables running:
    python3 tests/manual/test_runner.py <model_path>
"""

import sys
from pathlib import Path

MANUAL_DIR = Path(__file__).resolve().parent
if str(MANUAL_DIR) not in sys.path:
    sys.path.insert(0, str(MANUAL_DIR))

try:
    from .model_test_runner import main
except ImportError:
    from model_test_runner import main

if __name__ == "__main__":
    sys.exit(main())
