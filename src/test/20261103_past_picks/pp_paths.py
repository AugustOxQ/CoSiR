"""r19 shared paths and constants (spec docs/superpowers/specs/2026-11-03-past-picks-design.md).

Controller-owned: tickets import from here and never edit it. A ticket's own constants live in its own module. Ask
the controller (in the hand-back note) for an addition here.

Import this module FIRST in every pp_ module: it points the older rounds' output roots (BS_OUTPUTS, MT_OUTPUTS,
EG_OUTPUTS, RB_OUTPUTS) at r19's own OUT before any of their modules is imported, so a default write of r14's, r15's or 10-26's
code lands in r19's tree, never in a frozen round. Code that reads an older round's stored outputs passes that
round's real root (R14_OUT, R15_OUT, RB_PART2, EG_PART2, ...) explicitly.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROUND = Path(__file__).resolve().parent
TEST = ROUND.parent
REPO = ROUND.parents[2]
MAIN = Path('/project/CoSiR')            # frozen, gitignored inputs of older rounds live only in the main checkout
RECORDS = ROUND / 'records'              # committed records: SemArt set, stage 1 ranges and picks, status, verdicts

# Run outputs (spec §9, D25). PP_OUTPUTS points a smoke run at a scratch folder; smoke never writes RECORDS.
DEFAULT_OUT = Path('/data/SSD2/cosir_r19/20261103_past_picks/outputs')
OUT = Path(os.environ.get('PP_OUTPUTS', DEFAULT_OUT))
SMOKE = int(os.environ.get('PP_SMOKE') or 0)
if SMOKE:
    assert OUT.resolve() != DEFAULT_OUT.resolve(), 'smoke runs need their own PP_OUTPUTS'

# Older rounds' output roots follow r19's OUT (see the docstring). They must be set before those modules load.
_ROOT_ENV = {'BS_OUTPUTS': OUT, 'MT_OUTPUTS': OUT / 'mmart', 'EG_OUTPUTS': OUT / 'matched', 'RB_OUTPUTS': OUT / 'rb'}
for _name, _mod in (('BS_OUTPUTS', 'bs_paths'), ('MT_OUTPUTS', 'mt_paths'), ('EG_OUTPUTS', 'eg_paths'),
                    ('RB_OUTPUTS', 'rb_paths')):
    assert _mod not in sys.modules or os.environ.get(_name) == str(_ROOT_ENV[_name]), \
        f'{_mod} was imported before pp_paths: import pp_paths first'
    os.environ[_name] = str(_ROOT_ENV[_name])
for _flag in ('BS_SMOKE', 'MT_SMOKE'):         # r19 smoke runs use PP_SMOKE only; older rounds' smoke modes stay off
    assert not int(os.environ.get(_flag) or 0), f'{_flag} must be unset in r19 (use PP_SMOKE)'

# Older rounds (read only, spec §3.7)
BS_ROUND = MAIN / 'src/test/20261102_backbones_sweep'     # r14 (bs_*)
MT_ROUND = MAIN / 'src/test/20261102_mmart_test'          # r15 (mt_*)
MP_ROUND = MAIN / 'src/test/20261102_mmart_prep'          # MMArt prep (mp_*)
RB_ROUND = MAIN / 'src/test/20261025_recipe_backbones'    # 10-25 round (rb_*): M2, M3
EG_ROUND = MAIN / 'src/test/20261025_encoders_genre'      # 10-26 round (eg_*): M5 (sample), M6, M7, M8
SC_ROUND = MAIN / 'src/test/20261103_semart_check'        # SemArt check (sc_check.py, the set's rule)
R10_OUT = Path('/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs')
R11_OUT = Path('/data/SSD2/cosir_r11/20261029_retrieval/outputs')
R12_OUT = Path('/data/SSD2/cosir_r12/20261030_affection_confirmation/outputs')
R13_OUT = Path('/data/SSD2/cosir_r13/20261031_photo_finetune/outputs')
R14_OUT = Path('/data/SSD2/cosir_r14/20261102_backbones_sweep/outputs')
R15_OUT = Path('/data/SSD2/cosir_r15/20261102_mmart_test/outputs')
R14_RECORDS = BS_ROUND / 'records'
R15_RECORDS = MT_ROUND / 'records'
RB_PART2 = RB_ROUND / 'outputs/part2'                     # M3 (and M2): full/{held,matched}/M3, partitions/full/M3
EG_PART2 = Path('/data/SSD2/cosir_r10/20261025_encoders_genre/outputs/part2')   # M6, M7, M8 full; M5 sample
M5_ROOT = OUT / 'matched'                                 # r19's full-scale DINOv2 M5: part2/full/{held,matched}/M5

for _p in (ROUND, BS_ROUND, MT_ROUND, MP_ROUND, RB_ROUND, EG_ROUND, SC_ROUND, MAIN / 'src', MAIN):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))       # appended: r19's own modules (ROUND) win; older rounds add their own deps

# Feature stores
PRE = Path('/data/SSD2/pre_extract')
ART = PRE / 'artelingo_full'
AFF = PRE / 'affection_local'
MMART_PRE = PRE / 'mmart'
SEMART_PRE = PRE / 'semart'
MMART_INDEX = Path('/data/SSD/mmart/mmart_index.parquet')
SEMART_DATA = Path('/data/SSD/semart/SemArt')             # semart_{train,val,test}.csv, Images/
SC_NEARCOPY = SC_ROUND / 'sc_nearcopy.parquet'
SC_CHECK = SC_ROUND / 'sc_check.json'

# Encoders (§3.2). Keys are the ArtELingo store folder names for the new three.
ENCODERS = ('qwen', 'siglip2', 'siglip1', 'clip_l14', 'dinov2_large', 'clip_b32', 'clip_b16')
OLD_ENCODERS = ENCODERS[:4]              # c* checkpoints exist (r10, r14); reused frozen
NEW_ENCODERS = ENCODERS[4:]              # this round trains X-c* and the X photo model (SC1 to SC10)
WIDTH = {'qwen': 2048, 'siglip2': 1152, 'siglip1': 1152, 'clip_l14': 768,
         'dinov2_large': 1024, 'clip_b32': 512, 'clip_b16': 512}
ART_STORE = {'qwen': 'qwen3vl_emb_2b', 'siglip2': 'siglip2_so400m', 'siglip1': 'siglip_so400m',
             'clip_l14': 'clip_l14', 'dinov2_large': 'dinov2_large', 'clip_b32': 'clip_b32', 'clip_b16': 'clip_b16'}
CSTAR_ARM = 'k11=40'
CSTAR_ROOT = {'qwen': R10_OUT / 'full',
              'siglip2': R14_OUT / 'cstar/siglip2', 'siglip1': R14_OUT / 'cstar/siglip1',
              'clip_l14': R14_OUT / 'cstar/clip_l14',
              **{e: OUT / 'cstar' / e for e in NEW_ENCODERS}}       # r14's layout under r19's OUT
PHOTO_ROOT = {e: OUT / 'photo' / e for e in NEW_ENCODERS}          # r14's layout: <root>/s<seed>/
MATCHED = {'qwen': 'P3', 'siglip2': 'M2', 'siglip1': 'M8', 'clip_l14': 'M7',
           'clip_b32': 'M3', 'clip_b16': 'M6', 'dinov2_large': 'M5'}
N_TOPICS = 40
TEACHER_VIEW = {'emotion': 'emotion_focus', 'style': 'style_word', 'genre': 'genre_word'}

# New stores (§9): written by pp_extract, read by pp_stores. Format: img.npy float16 [n, d] unit rows, keys.json
# (keys in row order), meta.json (model id, commit hash, n, keys_sha256, img_sha256, writer).
NEW_STORE = {
    **{('mmart', e): MMART_PRE / e for e in NEW_ENCODERS},
    **{('affection', e): AFF / e for e in NEW_ENCODERS},
    ('semart', 'siglip1'): SEMART_PRE / 'siglip_so400m',
    ('semart', 'clip_l14'): SEMART_PRE / 'clip_l14',
    ('semart', 'clip_b16'): SEMART_PRE / 'clip_b16',
    ('semart', 'dinov2_large'): SEMART_PRE / 'dinov2_large',
    ('semart', 'genre_word'): SEMART_PRE / 'qwen3vl_emb_2b_aspect_prompts/genre_word',
}
STORE_ROWS = {'mmart': 16782, 'affection': 85007, 'semart': 21382}
PARITY_COS = 0.999                       # gate (i): by key, cosine >= 0.999
PARITY_ROWS = 100
UNIT_TOL = 1e-3

# The SemArt set (§3.3, D5, D10, D14, D19)
SEMART_CUT = 0.95                        # strict: kept when nearest cosine < 0.95 (sc_check.py)
SEMART_GATE = {'paintings': 11162, 'artists': 2717}
N_SEMART = 11159                         # after the md5 step (recorded, expected)
SEMART_CLOSE = 0.93                      # D14's context row: paintings at nearest cosine >= 0.93 left out
SEMART_TYPES = ('genre', 'historical', 'interior', 'landscape', 'mythological', 'other', 'portrait', 'religious',
                'still-life', 'study')   # class index = position; all 10 kept (D10)

# Protocol (§3.4, inherited from r14 part a and r15)
ASPECTS = ('emotion', 'style', 'genre')
SEEDS = (0, 1, 2, 3, 4)
PHOTO_SEED_OFFSET = 100
K = 20
KMEANS_STATES = (0, 1, 2)
N_DRAWS = {'clu': 2000, 'ret': 10000}
SMOKE_DRAWS = {'clu': 40, 'ret': 100}

# D17: the cells where a WIN licenses "better than": SemArt genre mAP of the four encoders with r15 records.
D17 = {(e, 'genre', 'ret', 'semart_new') for e in OLD_ENCODERS}

# The cost probe (§3.4 Cost, D22)
PROBE_N = 2000                           # the first 2,000 keys of r15's test cut set in sha256 order
PROBE_BATCH = 32
PROBE_WARMUP = 3                         # batches, not timed
PROBE_REPEATS = 3                        # model time; end to end once
CSTAR_NET_PARAMS = 4197376               # one trunk-and-head network (paper §3.6)
QWEN_E2E_REF_MS = 86.3                   # gate (viii): MMArt prep log, Qwen default, end to end
QWEN_E2E_TOL = 0.20

ENCODER_NAME = {'qwen': 'Qwen3-VL-Embedding-2B', 'siglip2': 'SigLIP 2', 'siglip1': 'SigLIP 1',
                'clip_l14': 'CLIP L/14', 'dinov2_large': 'DINOv2', 'clip_b32': 'CLIP B/32', 'clip_b16': 'CLIP B/16'}
DATA = ('artelingo_held', 'affection_test', 'mmart_test', 'semart_new')   # the `data` field of verdict cells


def order(keys):
    """Evaluation order on SemArt and MMArt (r15 D9): ascending sha256 hex of the key string."""
    return sorted(keys, key=lambda k: hashlib.sha256(k.encode('utf-8')).hexdigest())


def record_path(name):
    """Path of a record: RECORDS/<name>, or OUT/records/<name> under smoke (smoke never writes RECORDS)."""
    return (OUT / 'records' / name) if SMOKE else (RECORDS / name)


def read_committed_record(name):
    """The JSON of record <name>. Outside smoke, refuses unless the file equals its content in git HEAD (§3.5: stage 1
    is committed before any test number). `*.json` is gitignored: records are committed with `git add -f`."""
    path = record_path(name)
    text = path.read_text()
    if not SMOKE:
        rel = path.resolve().relative_to(REPO)
        head = subprocess.run(['git', '-C', str(REPO), 'show', f'HEAD:{rel}'], capture_output=True, text=True)
        if head.returncode != 0 or head.stdout != text:
            raise RuntimeError(f'record {rel} is not committed unchanged in HEAD; commit stage 1 before stage 2')
    return json.loads(text)
