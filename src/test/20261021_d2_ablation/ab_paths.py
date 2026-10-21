"""Round paths and checked imports of the frozen worktree code."""
import importlib
import sys
from pathlib import Path

ROUND = Path(__file__).resolve().parent
MAIN_ROUND = Path('/project/CoSiR/src/test/20261021_d2_ablation')
OUTPUTS = MAIN_ROUND / 'outputs'
MAIN_TEST = MAIN_ROUND.parent
SAMPLE = MAIN_TEST / '20261017_prompted_aspect_views/sample.npz'
ASPECT_BASE = MAIN_TEST / '20261018_aspect_graph_sheet'

_FOLDERS = {
    'apc_': '20261017_aspect_premise_checks',
    'pav_': '20261017_prompted_aspect_views',
    'itp_': '20261017_image_transfer_pilot',
    'pbp_': '20261018_purified_buddy_pilot',
    'ags_': '20261018_aspect_graph_sheet',
    'ms_': '20261019_model_sheet',
}
for _folder in _FOLDERS.values():
    sys.path.insert(0, str(ROUND.parent / _folder))

ms_train = importlib.import_module('ms_train')
ms_rows = importlib.import_module('ms_rows')
pav_sample = importlib.import_module('pav_sample')


def check_imports():
    """Reject cached bare imports from another checkout, including transitive helpers."""
    for name, module in tuple(sys.modules.items()):
        prefix = name.split('_', 1)[0] + '_'
        if prefix in _FOLDERS and '.' not in name:
            expected = ROUND.parent / _FOLDERS[prefix] / f'{name}.py'
            actual = Path(module.__file__).resolve()
            assert actual == expected.resolve(), f'{name}: imported {actual}; expected {expected}'
    assert ms_train.head is sys.modules['ags_head']
    assert ms_train.ags_trainimg is sys.modules['ags_trainimg']


check_imports()
