"""r29 shared paths and constants (spec docs/superpowers/specs/2026-11-12-siglip-only-adapter-design.md).

Controller-owned: tickets import from here and never edit it. A ticket's own constants live in its own module. Ask
the controller (in the hand-back note) for an addition here.
"""
import os
from pathlib import Path

ROUND = Path(__file__).resolve().parent
TEST = ROUND.parent
REPO = ROUND.parents[2]
MAIN = Path('/project/CoSiR')            # frozen, gitignored inputs of older rounds live only in the main checkout
RECORDS = ROUND / 'records'              # committed records: gates, picks, seed ranges, pass rule, cells
GATES = RECORDS / 'gates'                # one file per gate check, `<gate>.json` (Shared interfaces, ticket index)

# Run outputs (spec §9). R29_OUTPUTS points a smoke run at a scratch folder; smoke never writes RECORDS.
DEFAULT_OUT = Path('/data/SSD2/cosir_r29/20261112_siglip_only')
OUT = Path(os.environ.get('R29_OUTPUTS', DEFAULT_OUT))
SMOKE_OUT = DEFAULT_OUT / 'smoke'

# Frozen inputs (spec §3.7): K0 = r14's c* on SigLIP 1 (node402), and r28's records of it
R14_OUT = Path('/data/SSD2/cosir_r14/20261102_backbones_sweep/outputs')
K0_ROOT = R14_OUT / 'cstar/siglip1'      # weights/k11=40/<aspect>__s<seed>.pt, train/k11=40/, held/k11=40/
K0_ARM = 'k11=40'
R28_ROUND = TEST / '20261112_siglip_main'
R28_RECORDS = R28_ROUND / 'records'
R14_ROUND = TEST / '20261102_backbones_sweep'
R23_ROUND = TEST / '20261106_text_claim'

# Feature stores (D15: new stores only in new folders)
PRE = Path('/data/SSD2/pre_extract')
ART = PRE / 'artelingo_full'
AFFECTION = PRE / 'affection_local'
PREFIX = 'Represent the caption: '       # K2's caption form (D5); the store manifest records it after lowercasing

# Caption sources of the caption tower: name -> (store folder or None for the trainer's own Qwen3 path, input width).
# `qwen` is K0's input (Qwen3 plain caption embedding, read by the CT's loaders); the others read per-caption text rows
# of a store in scripts/extract_features.py's format (index.json, progress.json, txt.npy).
CAPTION_SOURCES = {
    'qwen': (None, 2048),
    'siglip1': (ART / 'siglip_so400m', 1152),                 # K1
    'siglip1_prefix': (ART / 'siglip_so400m_capprefix', 1152),  # K2 (ticket 02 writes it)
    'siglip2': (ART / 'siglip2_so400m', 1152),                # E2 if K1 wins (D13)
    'siglip2_prefix': (ART / 'siglip2_so400m_capprefix', 1152),  # E2 if K2 wins
    'clipl14': (ART / 'clip_l14', 768),                       # E3 if K1 wins
    'clipl14_prefix': (ART / 'clip_l14_capprefix', 768),      # E3 if K2 wins
}
# Caption sources each image encoder may take (bs_paths' encoder keys): Qwen3 (K0's input, gate runs) or that encoder's own
# text store, plain or prefixed (D13). The trainer asserts the pair.
ENCODER_CAPTION_SOURCES = {
    'siglip1': ('qwen', 'siglip1', 'siglip1_prefix'),
    'siglip2': ('qwen', 'siglip2', 'siglip2_prefix'),
    'clip_l14': ('qwen', 'clipl14', 'clipl14_prefix'),
}
OUTPUT_WIDTH = 2048                      # heads, teacher and graph embeddings, loss targets: unchanged (facts §3)

SEEDS = (0, 1, 2, 3, 4)
ASPECTS = ('emotion', 'style', 'genre')

# K0's pass-rule values as the spec prints them (§3.4, 10 decimals). The rule itself uses the unrounded values read from
# r28's records (records/valpick.json, records/text.json at r0), never recomputed, and asserts they round to these.
K0_IMAGE_MEAN = 0.3862917081
K0_IMAGE_SEEDS = (0.3848908314, 0.3878696916)   # (min, max) of the five per-seed image group means
K0_TEXT_MEAN = 0.2593512884
K0_TEXT_SEEDS = (0.2586239413, 0.2606346548)    # (min, max) of the five per-seed text group means
