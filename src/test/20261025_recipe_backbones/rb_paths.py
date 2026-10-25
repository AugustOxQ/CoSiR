"""Round paths and imports of the frozen rounds: code from this checkout, artifacts from the main checkout."""
import os
import sys
from pathlib import Path

ROUND = Path(__file__).resolve().parent
TEST = ROUND.parent
MAIN_TEST = Path('/project/CoSiR/src/test')
OUTPUTS = Path(os.environ.get('RB_OUTPUTS', MAIN_TEST / '20261025_recipe_backbones/outputs'))
RULE = ROUND / 'DECISION_RULE.md'

# Frozen artifacts are gitignored and exist only in the main checkout: always read them from these paths,
# never from a path relative to a worktree's copy of the round code.
AB_OUT = MAIN_TEST / '20261021_d2_ablation/outputs'
AB_GRAPHS = MAIN_TEST / '20261021_d2_ablation/graphs'
CT_OUT = MAIN_TEST / '20261020_claim_test/outputs'
CT_FULL = CT_OUT / 'stage2/full'
SAMPLE = MAIN_TEST / '20261017_prompted_aspect_views/sample.npz'
PRE = Path('/data/SSD2/pre_extract')
HELD_SHA256 = 'a0b1139c6056fd858ddbd2ee1dd4bafd1453e590818c6aa59883364c0288a876'
RB_ASPECTS = ('emotion', 'style', 'genre')

_FOLDERS = {
    'apc_': '20261017_aspect_premise_checks',
    'pav_': '20261017_prompted_aspect_views',
    'itp_': '20261017_image_transfer_pilot',
    'pbp_': '20261018_purified_buddy_pilot',
    'ags_': '20261018_aspect_graph_sheet',
    'pc_': '20261018_percept_core',
    'ms_': '20261019_model_sheet',
    'ct_': '20261020_claim_test',
    'ct2_': '20261020_claim_test',
    'ab_': '20261021_d2_ablation',
    'rb_': '20261025_recipe_backbones',
}
for _folder in dict.fromkeys(_FOLDERS.values()):
    _path = str(TEST / _folder)
    if _path not in sys.path:
        sys.path.insert(0, _path)


def check_imports():
    """Reject a prefixed module imported from another checkout or folder."""
    for name, module in tuple(sys.modules.items()):
        prefix = name.split('_', 1)[0] + '_'
        if prefix in _FOLDERS and '.' not in name and getattr(module, '__file__', None):
            expected = (TEST / _FOLDERS[prefix] / f'{name}.py').resolve()
            actual = Path(module.__file__).resolve()
            assert actual == expected, f'{name}: imported {actual}; expected {expected}'
