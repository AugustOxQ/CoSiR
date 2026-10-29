"""Make this round importable for its explicitly targeted tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def pytest_configure(config):
    config.addinivalue_line('markers', 'real_data: checks frozen research inputs (CPU only)')
