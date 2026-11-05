"""Round-local imports and output paths; frozen inputs are read only."""
import os
import sys
from pathlib import Path

ROUND = Path(__file__).resolve().parent
TEST = ROUND.parent
ROOT = TEST.parent.parent
for folder in ('20261029_retrieval', '20261027_recipe_sweep',
               '20261020_claim_test', '20261019_model_sheet',
               '20261017_aspect_premise_checks', '20261017_prompted_aspect_views',
               '20261017_emotion_word_masking'):
    path = str(TEST / folder)
    if path not in sys.path:
        sys.path.insert(0, path)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import rt_paths  # noqa: E402; installs the older rounds' dependencies

OUT = Path(os.environ.get('TX_OUTPUTS', '/data/SSD2/cosir_r22/20261105_text_retrieval'))
ASPECTS = ('emotion', 'style', 'genre')
SEED = 0
N_PAINTINGS = 6152
N_CAPTIONS = 30872
