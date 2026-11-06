"""r23 shared paths and constants (spec docs/superpowers/specs/2026-11-06-text-claim-design.md).

Controller-owned: tickets import from here and never edit it. A ticket's own constants live in its own module; ask the
controller, in the hand-back note, for an addition here.

Older rounds are read from the main checkout, since their frozen, gitignored inputs live only there.
TCL_OUTPUTS points a smoke run at a scratch folder; TCL_SMOKE=N limits each set to its first N paintings. Smoke never
writes the committed records.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROUND = Path(__file__).resolve().parent
REPO = ROUND.parents[2]
MAIN = Path('/project/CoSiR')
RECORDS = ROUND / 'records'              # committed: stage 1 ranges and picks, status, verdicts, C13 record

R22_ROUND = MAIN / 'src/test/20261105_text_retrieval'
R15_ROUND = MAIN / 'src/test/20261102_mmart_test'
for folder in (R15_ROUND, R22_ROUND):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
if str(ROUND) not in sys.path:
    sys.path.insert(0, str(ROUND))

DEFAULT_OUT = Path('/data/SSD2/cosir_r23/20261106_text_claim')
OUT = Path(os.environ.get('TCL_OUTPUTS', DEFAULT_OUT))
SMOKE = int(os.environ.get('TCL_SMOKE') or 0)
if SMOKE:
    assert OUT.resolve() != DEFAULT_OUT.resolve(), 'smoke needs TCL_OUTPUTS on a scratch folder'
    assert (OUT / 'records').resolve() != RECORDS.resolve(), 'smoke OUT/records must differ from committed RECORDS'

# Frozen outputs of older rounds (read only)
R22_OUT = Path('/data/SSD2/cosir_r22/20261105_text_retrieval')
R15_OUT = Path('/data/SSD2/cosir_r15/20261102_mmart_test/outputs')
R10_OUT = Path('/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs')

# New stores in the shared cache (D17). float16 `txt.npy`, `done.json`, and `ids.npz` (ArtELingo) or `keys.json` (MMArt).
ART_TEACHER_STORE = Path('/data/SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b_aspect_prompts')  # <instruction>_heldtxt
ART_ROBERTA_STORE = Path('/data/SSD2/pre_extract/artelingo_full/roberta_go_emotions_s2_heldtxt')
MMART_TEXT_STORE = Path('/data/SSD2/pre_extract/mmart/qwen3vl_emb_2b_text')   # <kind>/<instruction>/<split>
STORE_ROOT = Path(os.environ.get('TCL_STORES')) if os.environ.get('TCL_STORES') else None  # smoke: scratch folder

# Sets, rows, cells (spec §3.2 to §3.4)
SETS = ('art_val', 'art_held', 'mm_val', 'mm_test')
VAL_OF = {'art_held': 'art_val', 'mm_test': 'mm_val', 'art_val': 'art_val', 'mm_val': 'mm_val'}
GROUP = {'art_val': 'art', 'art_held': 'art', 'mm_val': 'mm', 'mm_test': 'mm'}
N_PAINTINGS = {'art_val': 6152, 'art_held': 12281, 'mm_val': 5769, 'mm_test': 16384}
N_CAPTIONS = {'art_val': 30872, 'art_held': 61744}
CAPTION = 'caption'                                  # ArtELingo's one text kind
KINDS = ('narrative', 'formal', 'emotional', 'historical', 'unified')    # MMArt; column e_<kind>
BAR_KIND = {'art': CAPTION, 'mm': 'narrative'}        # D9
ASPECTS = ('emotion', 'style', 'genre')
SEEDS = (0, 1, 2, 3, 4)
DIRECTIONS = ('t2i', 'i2t', 't2t')
PAIR_DIRECTIONS = ('t2i', 'i2t')
READOUTS = ('r0', 'center', 'center_k10')             # the menu, in tie order
INSTRUCTIONS = {'emotion': 'emotion_focus', 'style': 'style_word', 'genre': 'genre_word'}
A = 0.7                                               # D8
SEEDED_ROWS = ('cstar', 'late', 'free')
HEAD_ROWS = ('cstar', 'late', 'free', 'teacher')      # one vector per head; the other rows have one vector, head `all`
BAR_ROWS = ('cstar', 'late', 'plain', 'teacher')
SIDE_ROWS = ('free', 'clip_l14', 'siglip2_so400m', 'siglip_so400m', 'roberta')   # ArtELingo only, no bar (D12)
N_DRAWS = 10000
PAIR_LIMIT = -0.02                                    # Q: lower bound of late fusion minus plain Qwen pair R@10 (D5)
TEACHER_HOLDS, TEACHER_QUALIFIED = 5, 3               # D3
SC_OF = {('cstar', 'art'): 'SC1', ('cstar', 'mm'): 'SC2', ('late', 'art'): 'SC3', ('late', 'mm'): 'SC4'}


def kinds(group):
    """Text kinds of a set group: 'art' or 'mm'."""
    return (CAPTION,) if group == 'art' else KINDS


def score_path(set_name, kind, row, head, seed, direction, readout, out=None):
    """One ranking's per-query file. head: an aspect, or 'all' for one-vector rows; seed: 0..4, or 'x' when unseeded."""
    assert set_name in SETS and direction in DIRECTIONS and readout in READOUTS
    assert (head in ASPECTS) == (row in HEAD_ROWS) and (seed in SEEDS) == (row in SEEDED_ROWS)
    return Path(OUT if out is None else out) / 'scores' / set_name / kind / row / f'{head}__s{seed}__{direction}__{readout}.npz'


def means_path(set_name, kind, row, head, seed, out=None):
    """A row's validation means: arrays `t`, `i` (float32 [d]) and `n_t`, `n_i`. Only validation sets have one."""
    assert set_name in ('art_val', 'mm_val')
    return Path(OUT if out is None else out) / 'means' / set_name / kind / row / f'{head}__s{seed}.npz'


def record_path(name):
    """RECORDS/<name>, or OUT/records/<name> under smoke."""
    return (OUT / 'records' / name) if SMOKE else (RECORDS / name)


def read_committed_record(name):
    """The JSON of record <name>. Outside smoke, refuses unless the file equals its content in git HEAD (§3.5: stage 1
    is committed before any held number)."""
    path = record_path(name)
    text = path.read_text()
    if not SMOKE:
        rel = path.resolve().relative_to(REPO.resolve())
        head = subprocess.run(['git', '-C', str(REPO), 'show', f'HEAD:{rel}'], capture_output=True, text=True)
        if head.returncode != 0 or head.stdout != text:
            raise RuntimeError(f'record {rel} is not committed unchanged in HEAD; commit stage 1 before stage 2')
    return json.loads(text)
