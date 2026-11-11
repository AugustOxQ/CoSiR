"""r25 paper baselines: shared paths and constants (controller-owned; tickets import from here and never edit it).

Handoff: docs/superpowers/handoffs/2026-11-11-r25-paper-baselines-handoff.md. Two clustering comparators for the paper's
main table, scored beside Qwen c* on the ArtELingo held (test) split, AMI at 20 clusters per aspect:
    VLM2Vec-V2  image embeddings under each aspect instruction, k-means K=20 as c* (r10's MiniBatchKMeans, states 0-2)
    BERTopic    one document per held painting (its captions joined), default pipeline reduced to 20 topics, three
                UMAP seeds; its own topics are the partition

This module imports nothing heavy, so the two side envs (VLM2Vec, BERTopic) can import it. The scorer (CoSiR env) adds
r17's folder and the older rounds to sys.path through `add_scorer_paths()`.
"""
import os
import sys
from pathlib import Path

ROUND = Path(__file__).resolve().parent
TEST = ROUND.parent
REPO = ROUND.parents[2]
MAIN = Path('/project/CoSiR')                  # frozen, gitignored inputs and records of older rounds live only here

NS_ROUND = TEST / '20261103_new_scores'        # r17's scorer, read only
OM_ROUND = TEST / '20261103_outside_methods'   # r18's records, read only
RECORDS = ROUND / 'records'                    # committed records of this round

DEFAULT_OUT = Path('/data/SSD2/cosir_r25/20261111_paper_baselines/outputs')
OUT = Path(os.environ.get('PB_OUTPUTS', DEFAULT_OUT))
SMOKE = int(os.environ.get('PB_SMOKE') or 0)

# ---------------------------------------------------------------------------------------------- data
SET = 'artelingo'
SPLIT = 'held'                                 # ArtELingo's held split is the paper's test split (12,281 paintings)
N_HELD = 12281
ASPECTS = ('emotion', 'style', 'genre')
K = 20                                         # the paper's cluster count for every aspect
CSTAR_ENC = 'qwen'
ANNOTATIONS = Path('/data/PDD/artelingo/artelingo_train.json')   # 308,723 English caption rows
IMAGE_ROOT = Path('/data/PDD/wikiart_proj/wikiart')               # image file = IMAGE_ROOT / row['image']

# Inputs written once by the CoSiR env (pb_inputs.py) and read by the side envs, so they never import r17's code:
#   OUT/inputs/held_images.json  {names, images, names_sha256}   names in r17's `ns_data.keys('artelingo','held')` order
#   OUT/inputs/held_docs.json    {names, docs, n_captions, names_sha256}
INPUTS = OUT / 'inputs'

# ---------------------------------------------------------------------------------------------- VLM2Vec-V2
VLM2VEC_MODEL = 'VLM2Vec/VLM2Vec-V2.0'         # 2B, on Qwen2-VL-2B-Instruct
VLM2VEC_CODE = Path('/data/SSD2/cosir_r25/ext/VLM2Vec')   # code checkout outside the repo (its top-level package is `src`)
VLM2VEC_STORE = Path('/data/SSD2/pre_extract/artelingo/vlm2vec_v2')   # <aspect>/{emb.npy, keys.json, meta.json}
VLM2VEC_ENV = Path('/root/venvs/vlm2vec')

# The aspect instructions of the paper's Table 1 (Tables/instructions.tex), used verbatim; the extraction places them
# in VLM2Vec's own image-query format and records the exact final strings in each store's meta.json.
INSTRUCTIONS = {
    'emotion': 'Represent the emotion that this artwork, or this description of an artwork, conveys. Focus only on the emotion.',
    'style': 'Describe the artistic style of this artwork in one word.',
    'genre': 'Describe the genre of this artwork in one word.',
}

# ---------------------------------------------------------------------------------------------- BERTopic
BERTOPIC_ENV = Path('/root/venvs/bertopic')
MINILM = 'sentence-transformers/all-MiniLM-L6-v2'
UMAP_SEEDS = (0, 1, 2)
N_TOPICS = K
BERTOPIC_OUT = OUT / 'bertopic'                # topics_seed<s>.npz (names, topics int32) + .json sidecar

# ---------------------------------------------------------------------------------------------- scoring
# r: c*'s own validation AMI-at-20 seed range (Qwen, seeds 0-4, mean over k-means states), r18's stage 1 record.
RANGES = OM_ROUND / 'records' / 'stage1_om_ranges.json'   # key 'artelingo/qwen/<aspect>/ami/20'
# Gate: the candidate's AMI must reproduce r18's SC1 cells (same c* partitions, same labels).
R18_LABELS = OM_ROUND / 'records' / 'labels.json'          # ids 'SC1/artelingo/qwen/<aspect>/ami/20/tpips'
R18_CAND_TOL = 1e-9


def add_scorer_paths():
    """Put r17's round folder first on sys.path and add the older rounds (r17's order). CoSiR env only."""
    if str(NS_ROUND) not in sys.path:
        sys.path.insert(0, str(NS_ROUND))
    import ns_paths
    ns_paths.add_rounds()
    return ns_paths
