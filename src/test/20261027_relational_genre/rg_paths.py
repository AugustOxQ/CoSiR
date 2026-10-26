"""Round paths: code from this checkout, this round's outputs on /data/SSD2, frozen artifacts from the main checkout."""
import os
import sys
from pathlib import Path

ROUND = Path(__file__).resolve().parent
TEST = ROUND.parent
for _path in (str(ROUND), str(TEST / '20261025_encoders_genre'), str(TEST / '20261025_recipe_backbones')):
    if _path not in sys.path:
        sys.path.insert(0, _path)
import eg_paths  # noqa: E402  (puts 12-25's, 12-24's and the older rounds' folders on sys.path)

MAIN_TEST = eg_paths.MAIN_TEST
RULE = ROUND / 'DECISION_RULE.md'
RULE_COMMIT = '2e9529a81b6670b0411b61ce5c4f3cb7c4bbf2a8'
# Outputs of this round. Every runner takes --outputs; this is only the default.
OUTPUTS = Path(os.environ.get('RG_OUTPUTS', '/data/SSD2/cosir_r10/20261027_relational_genre/outputs'))
SMOKE_ROOT = Path('/tmp/claude-0')

# Frozen, read only (rule §5). Gitignored, so they exist only in the main checkout: never read them through a
# worktree's copy, and never pass one of them as an output folder to an eg_ or rb_ runner.
EG_OUT = MAIN_TEST / '20261025_encoders_genre/outputs'      # 12-25 ("EG/outputs")
EG_III = EG_OUT / 'stage_iii'
RB_OUT = eg_paths.RB_OUT                                    # 12-24 ("RB/outputs")
RB_PART1 = eg_paths.RB_PART1
RB_PART2 = eg_paths.RB_PART2
CT_OUT = eg_paths.CT_OUT
CT_FULL = eg_paths.CT_FULL
PRE = eg_paths.PRE
ASPECTS = eg_paths.ASPECTS                                  # ('emotion', 'style', 'genre')
P_SHA256 = eg_paths.P_SHA256                                # rule VA3: 12-25's p_sha256
HELD_SHA256 = eg_paths.HELD_SHA256                          # rule §3.2, VC3
N_P, N_OLD, N_HELD, WIDTH = 29232, 5000, 12281, 2048


def run_dir(outputs, arm, seed):
    """A sample-scale run of this round (stage A at seed 0, stage B at seeds 1 to 3): best states, exports, run.json,
    as 12-25's `eg_paths.iii_run`."""
    return Path(outputs) / 'train' / arm / f's{seed}'


def frozen_run(arm, seed):
    """12-25's frozen run folder of GR-8 (seeds 0 to 3) or R6-5 (seeds 1 to 3)."""
    assert (arm, seed) in {('GR-8', s) for s in range(4)} | {('R6-5', s) for s in (1, 2, 3)}, (arm, seed)
    return eg_paths.iii_run(EG_OUT, arm, seed)


def full_root(outputs):
    """Stage C's full-scale tree: weights/<arm>/, train/<arm>/, held/<arm>/ as 12-24's part2/full layout."""
    return Path(outputs) / 'stage_c/full'


def smoke_ok(outputs):
    """A smoke writes only under /tmp/claude-0."""
    return Path(outputs).resolve().is_relative_to(SMOKE_ROOT)


def check_imports():
    """Reject an rg_, eg_ or older prefixed module imported from another checkout or folder."""
    eg_paths.check_imports()
    for name, module in tuple(sys.modules.items()):
        if name.startswith('rg_') and '.' not in name and getattr(module, '__file__', None):
            expected = (ROUND / f'{name}.py').resolve()
            actual = Path(module.__file__).resolve()
            assert actual == expected, f'{name}: imported {actual}; expected {expected}'
