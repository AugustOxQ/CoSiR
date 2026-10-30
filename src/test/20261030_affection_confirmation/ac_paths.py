"""Round-local code, SSD outputs, and read-only frozen inputs from the main checkout (r12, spec 2026-10-30)."""
import os
import sys
from pathlib import Path

ROUND = Path(__file__).resolve().parent
TEST = ROUND.parent
_r11 = str(TEST / '20261029_retrieval')
if _r11 not in sys.path:
    sys.path.insert(0, _r11)
import rt_paths  # noqa: E402  (puts r10's round and, through rs_paths, every older round on sys.path)

MAIN = Path('/project/CoSiR')                       # frozen, gitignored inputs live only here, never in a worktree
MAIN_TEST, CT_OUT, PRE = rt_paths.MAIN_TEST, rt_paths.CT_OUT, rt_paths.PRE
R11_ROUND = MAIN / 'src/test/20261029_retrieval'
R11_OUT = Path('/data/SSD2/cosir_r11/20261029_retrieval/outputs')
R11_RANGES_FILE = R11_ROUND / 'stage0a_ranges.json'  # r11's r_a (mAP per aspect): SC5 to SC7 and the ArtELingo picks
OUTPUTS = Path(os.environ.get('AC_OUTPUTS', '/data/SSD2/cosir_r12/20261030_affection_confirmation/outputs'))

# Affection (spec §3.3). The store holds 85,007 images after ticket 01's merge (63,199 before).
AFF = Path('/data/SSD2/pre_extract/affection_local')
STORE = AFF / 'qwen3vl_emb_2b'
STORE_FI = AFF / 'qwen3vl_emb_2b_fi'                 # FI-only extraction, merged into STORE by ac_store.py
VIEWS = AFF / 'qwen3vl_emb_2b_aspect_prompts'
CLIP_VIEW = AFF / 'clip_l14_valtest'
ANNOTATIONS = Path('/data/SSD/affection/language/version_0/affection_preprocessed_public_version_0.csv')
FI_ROOT = Path('/data/SSD/affection/fi_t4sa/dataset/benchmark/FI/images')   # <emotion>/<emotion>_NNNN.jpg
FI_SOURCE = 'you_aaai_affective'
IMPRESSIONS_STORE = PRE / 'impressions/qwen3vl_emb_2b'
IMPRESSIONS_DIR = Path('/project/Impressions')

SPLITS = ('val', 'test')
N = {'val': 4251, 'test': 8501}
N_OLD = {'images': 63199, 'texts': 392331}
N_FI = {'images': 21808, 'texts': 134418}
SEEDS = (0, 1, 2, 3, 4)
ASPECTS = ('emotion', 'style', 'genre')
EMOTIONS = ('amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness',
            'something else')                       # r10's rs_vh.EMOTIONS: code = position (alphabetical)

# The teacher's two wordings (spec §3.2, D3). The artwork text is r10's emotion_focus prompt, unchanged.
WORDINGS = {'artwork': 'emotion_focus', 'photo': 'emotion_focus_photo'}
PHOTO_INSTRUCTION = ('Represent the emotion that this photo, or this description of a photo, conveys. '
                     'Focus only on the emotion.')
CRL_VERSIONS = ('clip_painting', 'clip_photo', 'qwen_painting', 'qwen_photo')   # spec §6 tie order

# Git-tracked records of this round (spec §3.5): written by the runner, committed by the queue.
STORE_HASHES_FILE = ROUND / 'stage0a_store_hashes.json'
RANGES_FILE = ROUND / 'stage0b_ranges.json'
PICKS_FILE = ROUND / 'stage1_picks.json'
