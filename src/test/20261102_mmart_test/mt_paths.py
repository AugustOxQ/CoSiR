"""r15 shared paths and constants (spec docs/superpowers/specs/2026-11-02-mmart-test-design.md).

Controller-owned: tickets import from here and never edit it. A ticket's own constants live in its own module. Ask
the controller (in the hand-back note) for an addition here.
"""
import json
import os
import subprocess
from pathlib import Path

ROUND = Path(__file__).resolve().parent
TEST = ROUND.parent
REPO = ROUND.parents[2]
MAIN = Path('/project/CoSiR')            # frozen, gitignored inputs of older rounds live only in the main checkout
RECORDS = ROUND / 'records'              # committed records: cut sets, stage 1 ranges and picks, status, verdicts

# Run outputs (spec §9, D13). MT_OUTPUTS points a smoke run at a scratch folder; smoke never writes RECORDS.
DEFAULT_OUT = Path('/data/SSD2/cosir_r15/20261102_mmart_test/outputs')
OUT = Path(os.environ.get('MT_OUTPUTS', DEFAULT_OUT))
SMOKE = int(os.environ.get('MT_SMOKE') or 0)

# Older rounds (read only, spec §3.7). r14's own constants are importable from its bs_paths.
BS_ROUND = MAIN / 'src/test/20261102_backbones_sweep'
MP_ROUND = MAIN / 'src/test/20261102_mmart_prep'
R10_OUT = Path('/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs')
R11_OUT = Path('/data/SSD2/cosir_r11/20261029_retrieval/outputs')
R12_OUT = Path('/data/SSD2/cosir_r12/20261030_affection_confirmation/outputs')
R14_OUT = Path('/data/SSD2/cosir_r14/20261102_backbones_sweep/outputs')

# MMArt (prep round, 2026-11-02)
MMART_INDEX = Path('/data/SSD/mmart/mmart_index.parquet')       # one row per painting; column `split`
MMART_NEARCOPY = Path('/data/SSD/mmart/mmart_nearcopy.parquet')  # key, max_cos, nearest_painting; one row per new key
MMART_PRE = Path('/data/SSD2/pre_extract/mmart')
PREP_PARITY = MP_ROUND / 'mp_parity.json'
PREP_STORES_CHECK = MP_ROUND / 'mp_stores_check.json'
PREP_NEARCOPY = MP_ROUND / 'mp_nearcopy.json'
ARTELINGO_PRE = Path('/data/SSD2/pre_extract/artelingo_full')

# Sets (§3.3, D19). Counts are what the real data give; gate (i) asserts the cut counts.
SPLIT_COUNTS = {'new': 16782, 'val': 5769, 'held': 11528, 'train': 40155}
CUT_TEST = 0.95                          # D19, inclusive: cosine >= 0.95 leaves the test set
CUT_FAR = 0.85                           # the far-subset diagnostic (D19), inclusive
N_TEST = 16384
N_FAR = 10403
N_EXACT = 34
NEARCOPY_REFERENCE = 49121               # ArtELingo train and validation paintings

# Encoders, aspects, arms (§3.2)
ENCODERS = ('qwen', 'siglip2', 'siglip1', 'clip_l14')
ASPECTS = ('emotion', 'style', 'genre')
SEEDS = (0, 1, 2, 3, 4)
DEFAULT_VIEW = {'qwen': 'qwen_default', 'siglip2': 'siglip2', 'siglip1': 'siglip1', 'clip_l14': 'clip_l14'}
TEACHER_VIEW = {'emotion': 'emotion_focus', 'style': 'style_word', 'genre': 'genre_word'}
VIEWS = tuple(DEFAULT_VIEW.values()) + tuple(TEACHER_VIEW.values())
WIDTH = {'qwen': 2048, 'siglip2': 1152, 'siglip1': 1152, 'clip_l14': 768}
# c* checkpoints: <root>/weights/k11=40/<aspect>__s<seed>.pt (Qwen: r10's c*; the others: r14's X-c*)
CSTAR_ARM = 'k11=40'
CSTAR_ROOT = {'qwen': R10_OUT / 'full',
              'siglip2': R14_OUT / 'cstar/siglip2',
              'siglip1': R14_OUT / 'cstar/siglip1',
              'clip_l14': R14_OUT / 'cstar/clip_l14'}
MATCHED = {'qwen': 'P3', 'siglip2': 'M2', 'siglip1': 'M8', 'clip_l14': 'M7'}   # PercepT on X's image (§3.2)
N_TOPICS = 40

# Labels (§3.3): MMArt's 9 emotion classes as given ("something else" included), its 20 style classes, the 10 WikiArt
# genre classes. Index order = the class's position here; -1 = no label.
CLASSES = {
    'emotion': ('amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness',
                'something else'),
    'style': ('abstract_expressionism', 'art_nouveau_modern', 'baroque', 'color_field_painting', 'cubism',
              'early_renaissance', 'expressionism', 'fauvism', 'high_renaissance', 'impressionism',
              'mannerism_late_renaissance', 'minimalism', 'naive_art_primitivism', 'northern_renaissance', 'pop_art',
              'post_impressionism', 'realism', 'rococo', 'romanticism', 'ukiyo_e'),
    'genre': ('abstract_painting', 'cityscape', 'genre_painting', 'illustration', 'landscape', 'nude_painting',
              'portrait', 'religious_painting', 'sketch_and_study', 'still_life'),
}

# Protocol (§3.4, inherited from r14 part a)
K = 20
KMEANS_STATES = (0, 1, 2)
N_DRAWS = {'clu': 2000, 'ret': 10000}
SMOKE_DRAWS = {'clu': 40, 'ret': 100}

# D7: the SC4 cells where a WIN licenses "better than" (cells c* won against the teacher on ArtELingo held).
D7 = {
    'qwen': {('emotion', 'clu'), ('style', 'clu'), ('genre', 'ret')},
    'siglip2': {('emotion', 'clu'), ('style', 'clu'), ('style', 'ret'), ('genre', 'ret')},
    'siglip1': {('emotion', 'clu'), ('style', 'clu'), ('style', 'ret'), ('genre', 'ret')},
    'clip_l14': {('emotion', 'clu'), ('style', 'clu'), ('style', 'ret'), ('genre', 'ret')},
}

ENCODER_NAME = {'qwen': 'Qwen3-VL-Embedding-2B', 'siglip2': 'SigLIP 2', 'siglip1': 'SigLIP 1',
                'clip_l14': 'CLIP L/14'}


def record_path(name):
    """Path of a record: RECORDS/<name>, or OUT/records/<name> under smoke (smoke never writes RECORDS)."""
    return (OUT / 'records' / name) if SMOKE else (RECORDS / name)


def read_committed_record(name):
    """The JSON of record <name>. Outside smoke, refuses unless the file equals its content in git HEAD (§3.5: stage 1
    is committed before any test number)."""
    path = record_path(name)
    text = path.read_text()
    if not SMOKE:
        rel = path.resolve().relative_to(REPO)
        head = subprocess.run(['git', '-C', str(REPO), 'show', f'HEAD:{rel}'], capture_output=True, text=True)
        if head.returncode != 0 or head.stdout != text:
            raise RuntimeError(f'record {rel} is not committed unchanged in HEAD; commit stage 1 before stage 2')
    return json.loads(text)
