"""Round-local paths and constants of r13 (spec docs/superpowers/specs/2026-10-31-photo-finetune-design.md).

r13 reads r12's real outputs and stores (never writes them) and writes only under OUTPUTS. Every pf module imports this
module first: it puts r12's round (and, through ac_paths, r11, r10 and every older round) on sys.path.
"""
import os
import sys
from pathlib import Path

ROUND = Path(__file__).resolve().parent
TEST = ROUND.parent
_r12 = str(TEST / '20261030_affection_confirmation')
if _r12 not in sys.path:
    sys.path.insert(0, _r12)
import ac_paths  # noqa: E402

# r13 always reads r12's real outputs and full val/test keys: r12's own smoke switches must stay off.
assert ac_paths.SMOKE == 0, 'AC_SMOKE must be unset for r13 (r13 smoke subsets train only: PF_SMOKE)'
assert ac_paths.OUTPUTS.resolve() == ac_paths.DEFAULT_OUTPUTS.resolve(), 'AC_OUTPUTS must be unset for r13'

MAIN = ac_paths.MAIN                                   # frozen, gitignored inputs live only in the main checkout
R10_OUT = Path('/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs')
CSTAR_WEIGHTS = R10_OUT / 'full/weights/k11=40'        # emotion__s{0..4}.pt (+ .json): the start points; read only
R10_GRAPHS = R10_OUT / 'graphs'                        # full/aspect__k40, full/uncorrected__k40, sample/aspect__k10
R12_OUT = ac_paths.DEFAULT_OUTPUTS                     # r12's outputs: comparators, picks, ranges, exports; read only

DEFAULT_OUTPUTS = Path('/data/SSD2/cosir_r13/20261031_photo_finetune/outputs')
OUTPUTS = Path(os.environ.get('PF_OUTPUTS', DEFAULT_OUTPUTS))
assert OUTPUTS.resolve() != R12_OUT.resolve(), 'PF_OUTPUTS must not be r12 outputs'
# Smoke mode: PF_SMOKE=N uses the first N sorted train keys (validation and test stay full). Read at call time.
SMOKE = int(os.environ.get('PF_SMOKE') or 0)
assert SMOKE >= 0, 'PF_SMOKE must be 0 (full) or a positive train-key count'
assert SMOKE == 0 or OUTPUTS.resolve() != DEFAULT_OUTPUTS.resolve(), \
    'PF_SMOKE needs PF_OUTPUTS set to a scratch folder, never the default outputs'

# Affection stores (spec §3.3). The default store holds all 85,007 images and 526,749 explanation rows.
AFF = ac_paths.AFF
STORE = ac_paths.STORE                                  # index.json, img.npy, txt.npy (float16, 2048)
VIEWS = ac_paths.VIEWS                                  # r12's val+test instructed image views live here
VAL_TEST_TEACHER_VIEW = VIEWS / 'emotion_focus'         # r12's artwork-wording teacher, val + test rows
TRAIN_IMG_VIEW = VIEWS / 'emotion_focus_train'          # new: instructed emotion image, train keys
TXT_VIEW = VIEWS / 'emotion_focus_txt'                  # new: instructed emotion explanations, train + val photos
if SMOKE > 0:
    TRAIN_IMG_VIEW = OUTPUTS / 'smoke_views' / TRAIN_IMG_VIEW.name
    TXT_VIEW = OUTPUTS / 'smoke_views' / TXT_VIEW.name
ANNOTATIONS = ac_paths.ANNOTATIONS
ART_TXT_STORE = ac_paths.PRE / 'artelingo_full/qwen3vl_emb_2b_aspect_prompts/emotion_focus'   # gate (i) text parity

SPLITS = ('train', 'val', 'test')
N = {'train': 72255, 'val': 4251, 'test': 8501}
N_TEXTS = {'train': 447680, 'val': 26371}             # explanation rows of train / val photos, "ignored" included
N_TEXTS_ELIGIBLE = {'train': 443258, 'val': 26113}    # rows whose own split is not "ignored"
SEEDS = (0, 1, 2, 3, 4)
EMOTIONS = ac_paths.EMOTIONS
ARTWORK_WORDING = ac_paths.WORDINGS['artwork']         # 'emotion_focus': c*'s own instruction (D8)


def train_seed(s):
    """Stage 2 run s starts from c*'s seed s and uses this seed for its training randomness (D10)."""
    return 100 + s


SWEEP_SEED = 0                                          # stage 1: c*'s seed-0 start, training seed 0

# Spec §3.4: the sweep score's constants (r12 stage1/scores.json and stage0b/ranges.json, unrounded).
TEACHER_AMI = 0.17189868120941146                       # teacher_wording.clustering.artwork
TEACHER_MAP = 0.304397579447216                         # teacher_wording.retrieval.artwork
R12_R_CLU = 0.003430790752866536
R12_R_MAP = 0.0010103233538674705
SWEEP_READOUT = 'center+k10'                            # D13
TIE_MARGIN = 0.5                                        # D9, in r units, inclusive (score >= best - 0.5)
NEAR_COPY_CUT = 0.85                                    # D12, inclusive (cosine >= 0.85 leaves training)

# Spec §3.5 gate references.
FROZEN_AMI_S0 = 0.14268969071601648                     # gate (ii): frozen c* seed-0 validation AMI
CSTAR_MAP_5SEED = 0.28584595252536343                   # gate (ii): r12's c* center+k10 validation mAP (5-seed mean)
BEST_DEV_LOSS_S0 = 6.161110545314756                    # gate (vi): c*'s seed-0 emotion best_dev_loss
TOL = {'ii': 1e-9, 'iii': 1e-3, 'vi': 1e-4, 'parity_cos': 0.999, 'unit_norm': 1e-3, 'c13': 1e-6}

# Spec §3.4 training recipe (c*'s emotion head; facts in .scratch/r13-photo-finetune/notes/code_map.md A1).
RECIPE = dict(weight_decay=0.05, batch_pairs=512, max_epochs=200, patience=15)

# Spec §3.5 sweep grid. The five grid columns, in D9's order.
COLUMNS = ('start', 'lr', 'k', 'correction', 'teacher_weight')
BASE = dict(start='cstar', lr=0.001, k=40, correction=True, teacher_weight=0.25)
GRID = {
    1: dict(BASE),
    2: dict(BASE, lr=0.0003),
    3: dict(BASE, lr=0.0001),
    4: dict(BASE, k=20),
    5: dict(BASE, k=80),
    6: dict(BASE, correction=False),
    7: dict(BASE, teacher_weight=1.0),
    8: dict(BASE, start='random'),
}
ROW9_SOURCES = (2, 3, 4, 5, 6, 7)                       # rows that may set row 9's values (spec §3.5)

# Git-tracked records of this round, written by the runner and committed by the queue (spec §3.5).
RECORDS_DIR = ROUND if SMOKE == 0 else OUTPUTS / 'records'   # smoke never writes the committed records
NEARCOPY_FILE = RECORDS_DIR / 'stage0a_nearcopy.json'  # near-copy count, histogram and the removed train keys
SHEET_FILE = RECORDS_DIR / 'stage1_sheet.json'         # rows 0 to 9, scores, row 9's settings, the pick
STAGE2_FILE = RECORDS_DIR / 'stage2_records.json'      # r' and the photo model's read-out pick
RECORDS = (NEARCOPY_FILE, SHEET_FILE, STAGE2_FILE)
