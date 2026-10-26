"""Put the round folder (and through rg_paths the older rounds) on sys.path for its tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rg_paths  # noqa: E402,F401
