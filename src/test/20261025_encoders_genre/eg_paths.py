"""Round paths: code from this checkout, this round's outputs on /data/SSD2, frozen artifacts from the main checkout."""
import os
import sys
from pathlib import Path

ROUND = Path(__file__).resolve().parent
TEST = ROUND.parent
for _path in (str(ROUND), str(TEST / '20261025_recipe_backbones')):
    if _path not in sys.path:
        sys.path.insert(0, _path)
import rb_paths  # noqa: E402  (puts the older rounds' folders on sys.path)

MAIN_TEST = rb_paths.MAIN_TEST
RULE = ROUND / 'DECISION_RULE.md'
RULE_COMMIT = 'f0f7c9133bc535825cc16cce0f0c0051ffefcf6c'
# Outputs of this round (spec D31). Every runner takes --outputs; this is only the default.
OUTPUTS = Path(os.environ.get('EG_OUTPUTS', '/data/SSD2/cosir_r10/20261025_encoders_genre/outputs'))
SMOKE_ROOT = Path('/tmp/claude-0')

# Frozen, read only (rule §5). Gitignored, so they exist only in the main checkout: never read them through a
# worktree's copy, and never pass RB_OUT as an output folder to an rb_ runner.
RB_OUT = MAIN_TEST / '20261025_recipe_backbones/outputs'
RB_PART1 = RB_OUT / 'part1'
RB_PART2 = RB_OUT / 'part2'
CT_OUT = rb_paths.CT_OUT
CT_FULL = rb_paths.CT_FULL
PRE = rb_paths.PRE
DINO_ARCHIVE = PRE / 'artelingo/r8_enc_dinov2l'   # archived DINOv2-large store for V5 (rule §4.1)
ASPECTS = rb_paths.RB_ASPECTS                       # ('emotion', 'style', 'genre')
SEEDS = (0, 1, 2, 3, 4)
P_SHA256 = '169f137e9088577a3c2f5ad352e430c5ef4d1ecb58e74dffacd4d0ea71181ef7'   # rule §3.2, 12-24's P
HELD_SHA256 = rb_paths.HELD_SHA256
# The extractor before this round's change (X0): main when the integration branch was cut.
X0_OLD_COMMIT = '45721894ade9ab553b92e561aeda9348c09cb399'


def p_student(outputs, arm, aspect, seed):
    """Stage ii: a student's P export (key 'img', float16 [29232, 2048]) with a .json sidecar."""
    return Path(outputs) / 'stage_ii/p' / arm / f'{aspect}__s{seed}.npz'


def p_own(outputs, arm, seed):
    """Stage ii: an own PercepT arm's P export (key 'prob', float32 sigmoid [29232, 40]) with a .json sidecar."""
    return Path(outputs) / 'stage_ii/p' / arm / f'P3__s{seed}.npz'


def iii_run(outputs, variant, seed):
    """Stage iii: one run's folder. Holds {aspect}.pt/.json (best state), {aspect}__P.npz and {aspect}__old5000.npz
    (key 'img', float16 [n, 2048]), genre__P_adaptor.npz (GA only, key 'adaptor'), run.json (status, best epochs)."""
    return Path(outputs) / 'stage_iii/train' / variant / f's{seed}'


def smoke_ok(outputs):
    """A smoke writes only under /tmp/claude-0."""
    return Path(outputs).resolve().is_relative_to(SMOKE_ROOT)


def check_imports():
    """Reject an eg_ or older prefixed module imported from another checkout or folder."""
    rb_paths.check_imports()
    for name, module in tuple(sys.modules.items()):
        if name.startswith('eg_') and '.' not in name and getattr(module, '__file__', None):
            expected = (ROUND / f'{name}.py').resolve()
            actual = Path(module.__file__).resolve()
            assert actual == expected, f'{name}: imported {actual}; expected {expected}'
