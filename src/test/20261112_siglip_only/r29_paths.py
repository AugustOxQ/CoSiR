"""r29 shared paths and constants (spec docs/superpowers/specs/2026-11-12-siglip-only-adapter-design.md).

Controller-owned: tickets import from here and never edit it. A ticket's own constants live in its own module. Ask
the controller (in the hand-back note) for an addition here.
"""
import json
import os
import subprocess
import sys
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
# Older rounds (r10 to r28) are imported from the main checkout only: their frozen modules hard-code its paths and its
# gitignored inputs live only there. After the merge into main it equals TEST. r29's own folder stays ROUND.
OLDER = MAIN / 'src' / 'test'
R28_ROUND = OLDER / '20261112_siglip_main'
R28_RECORDS = R28_ROUND / 'records'
R14_ROUND = OLDER / '20261102_backbones_sweep'
R23_ROUND = OLDER / '20261106_text_claim'
R17_ROUND = OLDER / '20261103_new_scores'
APC_ROUND = OLDER / '20261017_aspect_premise_checks'

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


# One place sets where older rounds are imported from (every r29 module calls these; none inserts older folders itself).
def use_older_rounds():
    """Put the older-round folders the r29 modules need on sys.path, in one fixed order (later = searched first)."""
    for folder in (APC_ROUND, R17_ROUND, R28_ROUND):
        if str(folder) in sys.path:
            sys.path.remove(str(folder))
        sys.path.insert(0, str(folder))


def check_older_imports():
    """Fail loudly if a loaded module of an older round (any checkout's src/test/<round>, other than r29's own) is not
    the main checkout's copy: frozen modules assert where their siblings came from, so a mixed import breaks later."""
    own, main = ROUND.name, OLDER.resolve()
    bad = []
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, '__file__', None)
        if not f:
            continue
        parts = Path(f).resolve().parts
        for i in range(len(parts) - 3):
            if parts[i] == 'src' and parts[i + 1] == 'test' and parts[i + 3:] and parts[i + 2] != own:
                if Path(*parts[:i + 2]) != main:
                    bad.append(f'{name}: {f}')
                break
    assert not bad, f'older-round modules not from {main} (mixed import): {bad[:5]}'


def require_committed(path, why):
    """The order rule's refusal (spec §3.4, §3.5 row 1b): `path` exists and is committed unchanged in git HEAD of its own
    repository (presence and commit, not its content or status), else RuntimeError naming `why`."""
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f'no {path.name} at {path}: {why}')
    git = ['git', '-C', str(path.parent)]
    tracked = subprocess.run(git + ['ls-files', '--error-unmatch', '--', path.name], capture_output=True)
    unchanged = subprocess.run(git + ['diff', '--quiet', 'HEAD', '--', path.name], capture_output=True)
    if tracked.returncode != 0 or unchanged.returncode != 0:
        raise RuntimeError(f'{path} is not committed unchanged in HEAD: {why}')


# ------------------------------------------------------------------------------------------------ the stage 2 gate (D3, A1)
A1_LINE = 'stage 2 ran on the fail under A1'     # carried by every record and sheet that says what stage 2 is, when the gate said "A1"
A1_ENV = 'R29_AMENDMENT'                         # the runner's `--amendment A1` sets it for every Python step it runs
A1_RECORD = 'amendment_A1.json'                  # in the records folder, committed (written by the controller)


def stage2_gate(records):
    """D3 as amended by A1 (spec §4): the one check every stage 2 entry point makes. Returns "pass" when `records/pass_rule.json`
    says pass (a status check only, as before), or "A1" when it says exactly "fail", R29_AMENDMENT=A1 is set and
    `records/amendment_A1.json` exists, is committed unchanged in HEAD, says amendment A1 and applies to this pass rule's status,
    level and arm. Any other case (no record, parked, a fail without the flag, a bad A1 record) raises RuntimeError."""
    records = Path(records)
    path = records / 'pass_rule.json'
    if not path.is_file():
        raise RuntimeError(f'no {path}: stage 2 runs only after a pass of SC2 (D3)')
    rule = json.loads(path.read_text())
    status = rule.get('status')
    if status == 'pass':
        return 'pass'
    if status != 'fail':
        raise RuntimeError(f'{path}: status {status!r}, not "pass" (D3): stage 2 does not run (A1 covers only a fail)')
    flag = os.environ.get(A1_ENV)
    if flag is None:
        raise RuntimeError(f'{path}: status "fail", not "pass": stage 2 does not run (D3); it runs on a fail only with the runner '
                           f'flag --amendment A1 ({A1_ENV}=A1) and a committed {A1_RECORD}')
    if flag != 'A1':
        raise RuntimeError(f'{A1_ENV}={flag!r}: only "A1" is an amendment this code knows')
    a1 = records / A1_RECORD
    require_committed(path, 'A1 names the committed pass rule it applies to')
    require_committed(a1, 'stage 2 runs on the fail only under a committed record of A1')
    doc = json.loads(a1.read_text())
    if doc.get('amendment') != 'A1' or doc.get('note') != A1_LINE:
        raise RuntimeError(f'{a1}: not the record of A1 (amendment {doc.get("amendment")!r}, note {doc.get("note")!r})')
    want = dict(pass_rule_status=status, pass_rule_level=rule.get('level'), arm=rule.get('arm'))
    if doc.get('applies_to') != want:
        raise RuntimeError(f'{a1}: applies_to {doc.get("applies_to")} does not match {path} ({want})')
    return 'A1'


def a1_marks(gate):
    """The key a record carries when `stage2_gate` answered "A1" (nothing otherwise: a record written on a pass is unchanged):
    `amendment` "A1" and `amendment_line` = A1_LINE."""
    assert gate in ('pass', 'A1', None), gate
    return dict(amendment='A1', amendment_line=A1_LINE) if gate == 'A1' else {}
