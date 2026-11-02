"""r14 shared paths and constants (spec docs/superpowers/specs/2026-11-02-backbones-sweep-design.md).

Controller-owned: tickets import from here and never edit it. A ticket's own constants live in its own module. Ask
the controller (in the hand-back note) for an addition here.
"""
import os
from pathlib import Path

ROUND = Path(__file__).resolve().parent
TEST = ROUND.parent
REPO = ROUND.parents[2]
MAIN = Path('/project/CoSiR')            # frozen, gitignored inputs of older rounds live only in the main checkout
RECORDS = ROUND / 'records'              # committed records: seed ranges, picks, near-copy sets, verdict files

# Run outputs (spec §9). BS_OUTPUTS points a smoke run at the scratchpad; smoke never writes RECORDS.
DEFAULT_OUT = Path('/data/SSD2/cosir_r14/20261102_backbones_sweep/outputs')
OUT = Path(os.environ.get('BS_OUTPUTS', DEFAULT_OUT))
SMOKE = int(os.environ.get('BS_SMOKE') or 0)

# Older rounds' stored outputs (read only, spec §3.7)
R10_OUT = Path('/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs')
R11_OUT = Path('/data/SSD2/cosir_r11/20261029_retrieval/outputs')
R12_OUT = Path('/data/SSD2/cosir_r12/20261030_affection_confirmation/outputs')
R13_OUT = Path('/data/SSD2/cosir_r13/20261031_photo_finetune/outputs')
R10_ROUND = MAIN / 'src/test/20261027_recipe_sweep'
R11_ROUND = MAIN / 'src/test/20261029_retrieval'
R12_ROUND = MAIN / 'src/test/20261030_affection_confirmation'
R13_ROUND = MAIN / 'src/test/20261031_photo_finetune'
RB_ROUND = MAIN / 'src/test/20261025_recipe_backbones'   # 10-25: SigLIP 2 matched arm M2
EG_ROUND = MAIN / 'src/test/20261025_encoders_genre'     # 10-26: SigLIP 1 (M8) and CLIP L/14 (M7) matched arms

# Feature stores
PRE = Path('/data/SSD2/pre_extract')
ART = PRE / 'artelingo_full'
AFF = PRE / 'affection_local'
EMO = PRE / 'emoset'
EMOSET_ZIP = Path('/data/SSD/emoset/EmoSet-118K.zip')
EMOSET_ROOT = Path('/data/SSD/emoset')   # val/test images, annotations and split files unpacked here (D9)
EMOSET_SPLITS = {'val': 5905, 'test': 17716}
EMOSET_CLASSES = ('amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness')

# Encoders. `art`: the ArtELingo default-image store folder (exists, 61,402 rows). `aff`: the Affection default-image
# store folders (all 85,007 keys together; CLIP L/14 is split in two folders). `emo`: the EmoSet store folder.
ENCODERS = {
    'qwen': dict(width=2048, art='qwen3vl_emb_2b', aff=('qwen3vl_emb_2b',), emo='qwen3vl_emb_2b'),
    'siglip2': dict(width=1152, art='siglip2_so400m', aff=('siglip2_so400m',), emo='siglip2_so400m'),
    'siglip1': dict(width=1152, art='siglip_so400m', aff=('siglip_so400m',), emo='siglip_so400m'),
    'clip_l14': dict(width=768, art='clip_l14', aff=('clip_l14_valtest', 'clip_l14_train'), emo='clip_l14'),
}
NEW_ENCODERS = ('siglip2', 'siglip1', 'clip_l14')        # part a (D2)
MATCHED = {'siglip2': 'M2', 'siglip1': 'M8', 'clip_l14': 'M7'}   # matched PercepT arms, full scale (§3.2)
FROZEN_ARM = {'siglip2': 'F2', 'siglip1': 'F8', 'clip_l14': 'F7'}  # frozen-image names of 10-25/10-26 (0(xii))

# Qwen instructed views on EmoSet (teacher wordings, r12's names) and their store folders
EMO_QWEN_VIEWS = {'default': 'qwen3vl_emb_2b',
                  'emotion_focus': 'qwen3vl_emb_2b_aspect_prompts/emotion_focus',
                  'emotion_focus_photo': 'qwen3vl_emb_2b_aspect_prompts/emotion_focus_photo'}

SEEDS = (0, 1, 2, 3, 4)
PHOTO_SEED_OFFSET = 100                  # photo training seed = 100 + s (r13)
SWEEP_SEEDS_KNOB = (1, 2, 3)             # the knob sweep (D12)
SWEEP_SEEDS_PHOTO = (0, 1, 2)            # the photo sweep (§3.2)
ASPECTS = ('emotion', 'style', 'genre')
NODE_END = '2026-11-04 18:06'            # node402's reservation end, Amsterdam time (D21)
NEAR_COPY_CUT = 0.85                     # D9, inclusive (cosine >= 0.85 leaves the EmoSet set)
