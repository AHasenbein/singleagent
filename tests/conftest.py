"""Put the project root (and this folder, for the shared fakes) on sys.path."""

import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
for path in (TESTS_DIR.parent, TESTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
