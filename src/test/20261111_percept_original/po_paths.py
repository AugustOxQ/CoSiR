"""r26 PercepT on its own encoders: shared paths and constants (controller-owned; tickets import it and never edit it).

Handoff: docs/superpowers/handoffs/2026-11-11-r26-percept-original-handoff.md. PercepT at paper settings (code tag
`percept_paper_k40`, GLOSSARY row 10; the strict Algorithm 1 cut beside it) on CLIP ViT-L/14 (image and text) and
ModernBERT-base tuned on GoEmotions, trained on the 42,969 ArtELingo train paintings (seeds 0 to 4), scored on the
12,281 test paintings with r17's scorer (AMI at 20 clusters per aspect, the "r20" k-means reading).

This module imports nothing heavy.
"""
import os
from pathlib import Path

ROUND = Path(__file__).resolve().parent
TEST = ROUND.parent
REPO = ROUND.parents[2]
MAIN = Path('/project/CoSiR')                  # frozen, gitignored inputs and records of older rounds live only here

CT_ROUND = TEST / '20261020_claim_test'        # ct_ / ct2_ code (read only, reused by import)
PC_ROUND = TEST / '20261018_percept_core'      # pc_ code (read only)
NS_ROUND = TEST / '20261103_new_scores'        # r17's scorer (read only)

DEFAULT_OUT = Path('/data/SSD2/cosir_r26/20261111_percept_original/outputs')
OUT = Path(os.environ.get('PO_OUTPUTS', DEFAULT_OUT))

# ---------------------------------------------------------------------------------------------- data
ANNOTATIONS = Path('/data/PDD/artelingo/artelingo_train.json')   # 308,723 English caption rows; sample_id = row index
IMAGE_ROOT = Path('/data/PDD/wikiart_proj/wikiart')               # image file = IMAGE_ROOT / images[i]['image']
N_PAINTINGS = 61402                                               # all ArtELingo paintings (train + val + test)
N_CAPTIONS = 308723
N_TRAIN, N_VAL, N_TEST = 42969, 6152, 12281
SEEDS = (0, 1, 2, 3, 4)
ASPECTS = ('emotion', 'style', 'genre')
K_SCORE = 20                                                      # clusters for the r20 reading (the paper's K)

# ---------------------------------------------------------------------------------------------- encoders
HF_HOME = Path('/data/SSD2/HF_home')
CLIP_L14_MODEL = 'openai/clip-vit-large-patch14'
CLIP_L14_REVISION = '32bd64288804d66eefd0ccbe215aa642df71cc41'
MODERNBERT_MODEL = 'cirimus/modernbert-base-go-emotions'           # the audit's C2 (report 2026-12-04)
MODERNBERT_REVISION = '808f9dbabf006d17d68cbbf232580e2580e83931'

# Existing global store (scripts/extract_features.py, backbone clip_l14): img.npy [61402, 768] fp16 in index.json
# `images` order (painting names); txt.npy [308723, 768] fp16 in index.json `texts` order (each with its sample_id).
CLIP_L14_GLOBAL = Path('/data/SSD2/pre_extract/artelingo_full/clip_l14')

# New stores (ticket A writes them; ticket B reads them). Join by the IDs they store, never by position (C6).
#   CLIP_L14_PATCH/tokens.npy   float16 [61402, 257, 768]: visual_projection(vision_model(x).last_hidden_state), all
#                               257 tokens (CLS first), no post-layernorm on the tokens, no normalisation; the form of
#                               the B/32 store (artelingo_percept_patch_features, 50 x 512).
#   CLIP_L14_PATCH/ids.npz      painting_names <U [61402], sorted ascending (the B/32 store's row order).
#   CLIP_L14_PATCH/done.json    model, revision, form, preprocessing, count, shape, dtype, parity gate, seconds.
#   MODERNBERT/txt.npy          float32 [308723, 768]: attention-mask mean of last_hidden_state, special tokens
#                               included, max_length 64 (pc_roberta.py's pooling for RoBERTa).
#   MODERNBERT/ids.npz          sample_ids int64 [308723] (= 0..308722, annotation row order), painting_ids <U.
#   MODERNBERT/done.json        model, revision, pooling, max_length, truncated count, batch, count, parity, seconds.
CLIP_L14_PATCH = Path('/data/SSD2/pre_extract/artelingo_full/clip_l14_patch')
MODERNBERT = Path('/data/SSD2/pre_extract/artelingo_full/modernbert_go_emotions')
PATCH_TOKENS, PATCH_WIDTH = 257, 768
TEXT_WIDTH = 768

# ---------------------------------------------------------------------------------------------- PercepT
VERSION = 'clipl14_modernbert'                 # Stage 1: (image, text, emotion, width)
VERSION_KINDS = ('clipl14_image', 'clipl14_text', 'modernbert', 768)
ARM = 'L1'                                     # Stage 2: attention pooling on the L/14 patch tokens
CUTS = ('k40', 'strict')                       # k40 fills the paper row; strict is reported with its collapse check

# Reference for a scorer check: arm B4 (paper K = 40, CLIP B/32 + RoBERTa, P1) held emotion AMI, r20 reading,
# report 2026-12-19 line 86.
B4_R20_EMOTION = 0.0699
