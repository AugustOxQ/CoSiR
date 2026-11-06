"""Keep this non-package round importable alongside older conftest modules."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tcl_paths  # noqa: F401,E402
