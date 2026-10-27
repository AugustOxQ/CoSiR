"""Round paths: code from this checkout, this round's outputs on /data/SSD2, frozen artifacts from the main checkout."""
import os
import sys
from pathlib import Path

ROUND = Path(__file__).resolve().parent
TEST = ROUND.parent
for _path in (str(ROUND), str(TEST / '20261027_relational_genre')):
    if _path not in sys.path:
        sys.path.insert(0, _path)
import rg_paths  # noqa: E402  (puts 12-27's, 12-25's, 12-24's and the older rounds' folders on sys.path)

MAIN_TEST = rg_paths.MAIN_TEST
RULE = ROUND / 'DECISION_RULE.md'
RULE_COMMIT = '35fcf684e32f2a154c3f3eb72235bafe7b015542'     # the rule's first commit (after its review)
FROZEN_COMMIT = '68e808167fda05350dfc510ea38c627f8a0afc89'   # V4a: the ms_, ct2_, rg_loss and rg_full hashes
# Outputs of this round. Every runner takes --outputs; this is only the default.
OUTPUTS = Path(os.environ.get('RS_OUTPUTS', '/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs'))
SMOKE_ROOT = Path('/tmp/claude-0')

# Frozen, read only (rule §5). Gitignored, so they exist only in the main checkout: never read them through a
# worktree's copy, and never pass one of them as an output folder.
RG_OUT = MAIN_TEST / '20261027_relational_genre/outputs'       # 12-27 ("RG/outputs")
RG_TRAIN = RG_OUT / 'train'                                    # GS-0.02/s{0..3}, R6-5/s0, ...
RG_STAGE_A, RG_STAGE_B, RG_STAGE_C = RG_OUT / 'stage_a', RG_OUT / 'stage_b', RG_OUT / 'stage_c'
RG_FULL = RG_STAGE_C / 'full'                                  # weights/, train/, held/ of GS-0.02
EG_OUT = rg_paths.EG_OUT                                       # 12-25 ("EG/outputs")
RB_OUT = rg_paths.RB_OUT                                       # 12-24 ("RB/outputs")
RB_PART2 = rg_paths.RB_PART2
CT_OUT = rg_paths.CT_OUT
CT_FULL = rg_paths.CT_FULL
PRE = rg_paths.PRE
FROZEN = (RG_OUT, EG_OUT, RB_OUT, CT_OUT, PRE)

G_ARM = 'GS-0.02'                                              # model G's 12-27 arm name
ASPECTS = rg_paths.ASPECTS                                     # ('emotion', 'style', 'genre')
P_SHA256 = rg_paths.P_SHA256                                   # V1b: 12-25's p_sha256
HELD_SHA256 = rg_paths.HELD_SHA256                             # V0c, V4c
N_P, N_OLD, N_HELD, WIDTH = 29232, 5000, 12281, 2048


def run_dir(outputs, name, seed):
    """A sample-scale run (stages 1 to 3, the development ablation, stage 0's seed 4): best states, exports on P,
    run.json. `name` is `rs_sheet.name(config)`."""
    return Path(outputs) / 'train' / name / f's{seed}'


def full_root(outputs):
    """Full-scale tree (stage 4, the held ablation): weights/<name>/, train/<name>/, held/<name>/ as 12-27's
    stage_c/full layout."""
    return Path(outputs) / 'full'


def smoke_ok(outputs):
    """A smoke writes only under /tmp/claude-0."""
    return Path(outputs).resolve().is_relative_to(SMOKE_ROOT)


def check_outputs(outputs, smoke=False):
    """An output folder is never inside a frozen input, and a smoke writes only to scratch."""
    outputs = Path(outputs).resolve()
    assert not smoke or smoke_ok(outputs), '--smoke outputs must be under /tmp/claude-0'
    for frozen in FROZEN:
        assert not outputs.is_relative_to(frozen.resolve()), f'outputs inside frozen input {frozen}'
    return outputs


def check_imports():
    """Reject an rs_, rg_, eg_ or older prefixed module imported from another checkout or folder."""
    rg_paths.check_imports()
    for name, module in tuple(sys.modules.items()):
        if name.startswith('rs_') and '.' not in name and getattr(module, '__file__', None):
            expected = (ROUND / f'{name}.py').resolve()
            actual = Path(module.__file__).resolve()
            assert actual == expected, f'{name}: imported {actual}; expected {expected}'
