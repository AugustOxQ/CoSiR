"""r17 shared paths and constants (spec docs/superpowers/specs/2026-11-03-new-scores-design.md).

Controller-owned: tickets import from here and never edit it. A ticket's own constants live in its own module. Ask
the controller (in the hand-back note) for an addition here.
"""
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

ROUND = Path(__file__).resolve().parent
TEST = ROUND.parent
REPO = ROUND.parents[2]
MAIN = Path('/project/CoSiR')            # frozen, gitignored inputs and records of older rounds live only here
MAIN_TEST = MAIN / 'src/test'
RECORDS = ROUND / 'records'              # committed records: status, stage 1 ranges and picks, cells, verdicts

# Run outputs (spec §9, D13). NS_OUTPUTS points a smoke run at a scratch folder; smoke never writes RECORDS.
DEFAULT_OUT = Path('/data/SSD2/cosir_r17/20261103_new_scores/outputs')
OUT = Path(os.environ.get('NS_OUTPUTS', DEFAULT_OUT))
SMOKE = int(os.environ.get('NS_SMOKE') or 0)
if SMOKE:
    assert (OUT / 'records').resolve() != RECORDS.resolve(), 'smoke OUT/records must differ from committed RECORDS'

# Older rounds (frozen, read only; spec §3.7). Their code is imported from the MAIN checkout, never edited.
ROUNDS = {
    'r10': MAIN_TEST / '20261027_recipe_sweep',      # rs_*
    'r11': MAIN_TEST / '20261029_retrieval',         # rt_*
    'r12': MAIN_TEST / '20261030_affection_confirmation',  # ac_*
    'r14': MAIN_TEST / '20261102_backbones_sweep',   # bs_*
    'r15prep': MAIN_TEST / '20261102_mmart_prep',    # mp_*
    'r15': MAIN_TEST / '20261102_mmart_test',        # mt_*
}
R10_OUT = Path('/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs')
R11_OUT = Path('/data/SSD2/cosir_r11/20261029_retrieval/outputs')
R12_OUT = Path('/data/SSD2/cosir_r12/20261030_affection_confirmation/outputs')
R13_OUT = Path('/data/SSD2/cosir_r13/20261031_photo_finetune/outputs')
R14_OUT = Path('/data/SSD2/cosir_r14/20261102_backbones_sweep/outputs')
R15_OUT = Path('/data/SSD2/cosir_r15/20261102_mmart_test/outputs')
PRE = Path('/data/SSD2/pre_extract')


def add_rounds():
    """Put the older rounds' folders (MAIN checkout) and the repo root on sys.path, in the order r15's modules use
    (r15 and r14 first; `ac_paths` then adds r11, r10 and the older ones). Idempotent."""
    for p in (MAIN, ROUNDS['r10'], ROUNDS['r11'], ROUNDS['r12'], ROUNDS['r15prep'], ROUNDS['r14'], ROUNDS['r15']):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import ac_paths  # noqa: F401  (adds r11's, r10's and the older rounds' folders)


# Encoders, aspects, arms (§3.2)
ENCODERS = ('qwen', 'siglip2', 'siglip1', 'clip_l14')
ASPECTS = ('emotion', 'style', 'genre')
SEEDS = (0, 1, 2, 3, 4)
KMEANS_STATES = (0, 1, 2)
WIDTH = {'qwen': 2048, 'siglip2': 1152, 'siglip1': 1152, 'clip_l14': 768}
CSTAR_WIDTH = 2048
UNTOLD_WIDTH = 3 * CSTAR_WIDTH           # D8: three unit heads joined, divided by sqrt(3)
N_TOPICS = 40                            # matched PercepT probabilities
N_CRL_VALUES = 50                        # CRL: 50 value texts per aspect
MATCHED = {'qwen': 'P3', 'siglip2': 'M2', 'siglip1': 'M8', 'clip_l14': 'M7'}   # matched PercepT arm names
ENCODER_NAME = {'qwen': 'Qwen3-VL-Embedding-2B', 'siglip2': 'SigLIP 2', 'siglip1': 'SigLIP 1',
                'clip_l14': 'CLIP L/14'}

# Sets and splits (§3.3). Painting sets carry three aspects; photo sets emotion only.
PAINTING_SETS = ('artelingo', 'mmart')
PHOTO_SETS = ('affection', 'emoset')
HELD_SPLIT = {'artelingo': 'held', 'mmart': 'test', 'affection': 'test', 'emoset': 'test'}
DEV_SPLIT = 'val'
N_ROWS = {('artelingo', 'held'): 12281, ('artelingo', 'val'): 6152,
          ('mmart', 'test'): 16384, ('mmart', 'val'): 5769,
          ('affection', 'test'): 8501, ('affection', 'val'): 4251,
          ('emoset', 'test'): 17348, ('emoset', 'val'): 5770}
SET_ASPECTS = {'artelingo': ASPECTS, 'mmart': ASPECTS, 'affection': ('emotion',), 'emoset': ('emotion',)}
PHOTO_ENCODERS = {'affection': ENCODERS, 'emoset': ('qwen',)}     # SC4 scope

# Class counts K (§3.3; labelled classes present; the same on each validation split). Gate (iii) checks them.
CLASS_COUNT = {('artelingo', 'emotion'): 9, ('artelingo', 'style'): 27, ('artelingo', 'genre'): 10,
               ('mmart', 'emotion'): 9, ('mmart', 'style'): 20, ('mmart', 'genre'): 10,
               ('affection', 'emotion'): 9, ('emoset', 'emotion'): 8}

# k-means (§3.4, D20): r10's settings; threads per set as its earlier round, at every K.
K20 = 20
KMEANS_THREADS = {'artelingo': 8, 'mmart': 8, 'affection': 1, 'emoset': 1}
# Count sweep (§3.6, D2): point values only, no verdict word.
SWEEP_COUNTS = {'emotion': (5, 8, 9, 16, 20), 'genre': (5, 10, 16, 20), 'style': (10, 20, 27, 40)}

# Metrics (§3.4, D6)
CLASS_METRICS = ('ami', 'nmi', 'ari', 'acc')
SWEEP_METRICS = ('ami', 'ari')

# Protocol (§3.4, D12)
N_DRAWS = {'clu': 2000, 'ret': 10000}
SMOKE_DRAWS = {'clu': 40, 'ret': 100}
BOOT_SEED = 0                            # a fresh default_rng(0) per cell

# Tolerances (§3.5)
TOL_REPLAY = 1e-9
TOL_METRIC = 1e-12
TOL_MAPPER = 1e-5
TOL_C13 = 1e-6
TOL_UNIT = 1e-3

# D15: teacher cells whose bar is WIN (r15 D7's MMArt emotion and style clustering cells, four encoders; EmoSet r14
# SC9, Qwen). Every other teacher cell (SC3, SC4): LEVEL or WIN.
TEACHER_WIN_BAR = frozenset({('mmart', e, a) for e in ENCODERS for a in ('emotion', 'style')}
                            | {('emoset', 'qwen', 'emotion')})

# Candidate whose validation ranges give r (D7, D17): c* (told) on painting sets, the photo model on photo sets.
CRL_ENCODER = 'clip_l14'                 # SC5, SC6: c* on CLIP L/14 against CRL on CLIP L/14


def record_path(name):
    """Path of a record: RECORDS/<name>, or OUT/records/<name> under smoke (smoke never writes RECORDS)."""
    return (OUT / 'records' / name) if SMOKE else (RECORDS / name)


def read_committed_record(name):
    """The JSON of record <name>. Outside smoke, refuses unless the file equals its content in git HEAD of this
    checkout (§3.5: stage 1 is committed before any test number)."""
    path = record_path(name)
    text = path.read_text()
    if not SMOKE:
        rel = path.resolve().relative_to(REPO)
        head = subprocess.run(['git', '-C', str(REPO), 'show', f'HEAD:{rel}'], capture_output=True, text=True)
        if head.returncode != 0 or head.stdout != text:
            raise RuntimeError(f'record {rel} is not committed unchanged in HEAD; commit stage 1 before stage 2')
    return json.loads(text)


def write_json(path, obj):
    """Atomically write JSON (indent 1, sorted keys) to path, creating parents."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    stream = tmp.open('x')                       # normal 0666 & umask mode, rather than mkstemp's 0600
    try:
        with stream:
            stream.write(json.dumps(obj, indent=1, sort_keys=True) + '\n')
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
