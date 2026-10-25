"""Put the round folder on sys.path for its tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
