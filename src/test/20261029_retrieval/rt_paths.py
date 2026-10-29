"""Round-local code, SSD outputs, and read-only frozen inputs from the main checkout."""
import json
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
# The committed file holds only what a rerun of 0a reproduces exactly; wall-clock times stay in <outputs>/stage0a/ranges.json.
RANGES_KEYS = ('map', 'ndcg', 'inputs', 'smoke')


def committed_ranges(doc):
    """The part of a stage 0a doc that RANGES_FILE holds, in file order."""
    return {k: doc[k] for k in RANGES_KEYS}


def ranges_match(outputs_doc_path):
    """True when RANGES_FILE equals the committed keys of the stage 0a doc at `outputs_doc_path` (parsed JSON)."""
    return json.loads(Path(RANGES_FILE).read_text()) == committed_ranges(json.loads(Path(outputs_doc_path).read_text()))
