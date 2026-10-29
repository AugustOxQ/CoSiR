"""Round-local code, SSD outputs, and read-only frozen inputs from the main checkout."""
import os
import sys
from pathlib import Path

ROUND = Path(__file__).resolve().parent
TEST = ROUND.parent
_sweep = str(TEST / '20261027_recipe_sweep')
if _sweep not in sys.path:
    sys.path.insert(0, _sweep)
import rs_paths  # noqa: E402

MAIN_TEST, CT_OUT, PRE = rs_paths.MAIN_TEST, rs_paths.CT_OUT, rs_paths.PRE
OUTPUTS = Path(os.environ.get('RT_OUTPUTS', '/data/SSD2/cosir_r11/20261029_retrieval/outputs'))
ASPECTS = ('emotion', 'style', 'genre')
SEEDS = (0, 1, 2, 3, 4)
N_HELD, N_VAL = 12281, 6152
N_GENRE = {'held': 9949, 'val': 4960}
RANGES_FILE = ROUND / 'stage0a_ranges.json'
