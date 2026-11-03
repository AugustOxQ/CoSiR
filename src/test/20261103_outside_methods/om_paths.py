"""r18 shared paths and constants (spec docs/superpowers/specs/2026-11-03-outside-methods-design.md).

Controller-owned: tickets import from here and never edit it. A ticket's own constants live in its own module. Ask
the controller (in the hand-back note) for an addition here.

r17's protocol constants (encoders, class counts, cluster counts, draws, metrics) are re-exported from r17's
`ns_paths`, imported read-only from this checkout (spec §3.7): one source for both rounds.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROUND = Path(__file__).resolve().parent
TEST = ROUND.parent
REPO = ROUND.parents[2]
MAIN = Path('/project/CoSiR')            # frozen, gitignored inputs and records of older rounds live only here
RECORDS = ROUND / 'records'              # committed records: gates, stage 1 ranges and picks, status, cells, labels

# Run outputs (spec D18). OM_OUTPUTS points a smoke run at a scratch folder; smoke never writes RECORDS.
DEFAULT_OUT = Path('/data/SSD2/cosir_r18/20261103_outside_methods/outputs')
OUT = Path(os.environ.get('OM_OUTPUTS', DEFAULT_OUT))
SMOKE = int(os.environ.get('OM_SMOKE') or 0)
if SMOKE:
    assert (OUT / 'records').resolve() != RECORDS.resolve(), 'smoke OUT/records must differ from committed RECORDS'

# r17 (the scorer, read only; spec §3.7, D15). Its round folder from this checkout, so the merged interface is used.
NS_ROUND = TEST / '20261103_new_scores'
if str(NS_ROUND) not in sys.path:
    sys.path.insert(0, str(NS_ROUND))
import ns_paths as ns  # noqa: E402

ns.add_rounds()                          # older rounds' folders (MAIN checkout) on sys.path, r17's order
ROUNDS = dict(ns.ROUNDS, r17=NS_ROUND)
R10_OUT, R11_OUT, R14_OUT, R15_OUT = ns.R10_OUT, ns.R11_OUT, ns.R14_OUT, ns.R15_OUT
R17_OUT = ns.DEFAULT_OUT                 # r17's real outputs (never its smoke folder)
PRE = ns.PRE

# Encoders, aspects, seeds (§3.2), from r17
ENCODERS = ns.ENCODERS                   # ('qwen', 'siglip2', 'siglip1', 'clip_l14')
ASPECTS = ns.ASPECTS                     # ('emotion', 'style', 'genre')
SEEDS = ns.SEEDS                         # c* seeds 0 to 4
KMEANS_STATES = ns.KMEANS_STATES         # 0, 1, 2
WIDTH = ns.WIDTH
CSTAR_WIDTH = ns.CSTAR_WIDTH
ENCODER_NAME = ns.ENCODER_NAME

# Sets (§3.3): the two painting sets only. Development split 'val'; held split per set.
SETS = ('artelingo', 'mmart')
HELD_SPLIT = {s: ns.HELD_SPLIT[s] for s in SETS}          # artelingo: held, mmart: test
DEV_SPLIT = ns.DEV_SPLIT                                   # 'val'
N_ROWS = {k: v for k, v in ns.N_ROWS.items() if k[0] in SETS}
N_GENRE_LABELLED = {('artelingo', 'val'): 4960, ('artelingo', 'held'): 9949}

# Class counts and cluster counts (§3.3), from r17
CLASS_COUNT = {k: v for k, v in ns.CLASS_COUNT.items() if k[0] in SETS}
K20 = ns.K20
SWEEP_COUNTS = ns.SWEEP_COUNTS           # emotion 5, 8, 9, 16, 20; genre 5, 10, 16, 20; style 10, 20, 27, 40
KMEANS_THREADS = {s: ns.KMEANS_THREADS[s] for s in SETS}
CLASS_METRICS = ns.CLASS_METRICS         # ami, nmi, ari, acc
SWEEP_METRICS = ns.SWEEP_METRICS         # ami, ari
N_DRAWS = ns.N_DRAWS                     # clu 2000, ret 10000
SMOKE_DRAWS = ns.SMOKE_DRAWS
BOOT_SEED = ns.BOOT_SEED
TOL_C13 = ns.TOL_C13


def label_ks(dataset, aspect):
    """The K of the labelled cells (§6): the class count and 20 (one K when they are equal, MMArt style)."""
    return tuple(sorted({CLASS_COUNT[(dataset, aspect)], K20}))


def plain_ks(dataset):
    """The K of the plain-image TURTLE fits on a set: every r17 count of any aspect (one partition serves all three
    aspects, §3.3). 8 values on each set."""
    return tuple(sorted({k for a in ASPECTS for k in SWEEP_COUNTS[a]} | {CLASS_COUNT[(dataset, a)] for a in ASPECTS}))


# TURTLE (§3.4, D9, D21): the official code's defaults; starts 0 to 2 on the plain image, start 0 per c* seed.
TURTLE = {'gamma': 10.0, 'outer_steps': 6000, 'inner_steps': 10, 'outer_lr': 1e-3, 'inner_lr': 1e-3,
          'batch_size': 10000, 'warm_start': False}
TURTLE_PLAIN_STARTS = (0, 1, 2)
TURTLE_CSTAR_START = 0
N_TURTLE_FITS = 412                      # 192 plain + 220 on c*'s heads (§3.3); ticket 09 asserts its list

# TPIPS (§3.2, D2, D18, D20)
TPIPS_REPO = 'sywang/TPIPS-Embed-Qwen3VL-8B'
TPIPS_SNAPSHOT = '86a1493c60121e656a1c075bc7aed88f25fcd4d4'
TPIPS_BACKBONE_REPO = 'Qwen/Qwen3-VL-Embedding-8B'
TPIPS_BACKBONE_SNAPSHOT = '2c4565515e0f265c6511776e7193b22c0968ddc7'
TPIPS_ENV = Path('/root/venvs/tpips')    # local venv: CoSiR packages plus tpips 0.1.2
TPIPS_FACTORS = {'emotion': 'emotion', 'style': 'style', 'genre': 'genre'}   # aspect -> factor text (D2)
TPIPS_WIDTH = 4096
TPIPS_STORE = {'artelingo': PRE / 'artelingo_full' / 'tpips_8b', 'mmart': PRE / 'mmart' / 'tpips_8b'}
TPIPS_PROBE_N = 48                       # gate (iv): the first 48 MMArt 'new' keys in sorted order (as om_tpips_probe)
N_TPIPS_IMAGES = 34817                   # ArtELingo held 12,281 + validation 6,152 + MMArt test 16,384

# CLAY (§3.4, D4 to D8): CLIP L/14; text set A = r11's CRL lists, set B = ticket 05's generated lists.
CLAY_ENCODER = 'clip_l14'
CLAY_TEXT_A = {a: R11_OUT / 'crl' / f'{a}_text.npy' for a in ASPECTS}      # 50 unit rows each, rank 49
CLAY_LISTS_A = R11_OUT / 'crl' / 'lists.json'
CLAY_RANK_A = 49
CLAY_B_N = 100                           # D5: values asked for and kept (first 100 after dedup)
CLAY_B_PROMPT = 'a painting of {value}'
CLAY_B_MODEL = 'Qwen/Qwen3-VL-8B-Instruct'
CLAY_B_DIR = OUT / 'textset_b'           # lists.json and <aspect>_text.npy (unit rows); frozen once generated
CLIP_L14_REPO = 'openai/clip-vit-large-patch14'


def clay_rank(n_texts):
    """D5: rank min(50, n - 1); set A (50 texts) gives 49."""
    return min(50, n_texts - 1)


# Parity references (gates (i), (ii)), made once by the official code (round log, spec chat). Read only.
EXT = Path('/data/SSD2/cosir_r18/ext')
PARITY = EXT / 'parity'
PARITY_FILES = {'inputs': PARITY / 'inputs.npz', 'clay': PARITY / 'clay_ref.npz', 'turtle': PARITY / 'turtle_ref.npz'}
PARITY_SHA256 = {'inputs': '7b10bf69d6fe0358b7b00d98ef974a2d486a73a74f4faa384cc75dd219e4876c',
                 'clay': '3bd542a01e9898dceb4f198c99a4240fc87f81693f089f91c6d0554104c8771a',
                 'turtle': 'd73efb37269c50711356e642c0fb78fd65167d588636e2d704cb2353f327a4b1'}
PARITY_N = 2000
PARITY_TURTLE = {'K': 9, 'outer_steps': 100, 'seed': 0, 'threads': 8}
BATCH_DRAW_CHECK = {'n': (12281, 16384), 'draws': 3, 'seed': 0}   # gate (ii)'s numpy check of batch indices

# Gate numbers (D23)
TOL_CLAY = 1e-5
TOL_TURTLE = 1e-5
TOL_REPLAY = 1e-9
GPU_CPU_AMI = 0.99
TPIPS_COS = 0.999
TEMI_TOY_ACC = 0.95


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def parity_file(name):
    """Path of a parity reference after checking its sha256 against the round log's."""
    path = PARITY_FILES[name]
    got = sha256_file(path)
    assert got == PARITY_SHA256[name], f'{path}: sha256 {got} != round log {PARITY_SHA256[name]}'
    return path


def record_path(name):
    """Path of a record: RECORDS/<name>, or OUT/records/<name> under smoke (smoke never writes RECORDS)."""
    return (OUT / 'records' / name) if SMOKE else (RECORDS / name)


def read_committed_record(name):
    """The JSON of record <name>. Outside smoke, refuses unless the file equals its content in git HEAD of this
    checkout (§3.5: stage 1 is committed before any stage 2 number)."""
    path = record_path(name)
    text = path.read_text()
    if not SMOKE:
        rel = path.resolve().relative_to(REPO)
        head = subprocess.run(['git', '-C', str(REPO), 'show', f'HEAD:{rel}'], capture_output=True, text=True)
        if head.returncode != 0 or head.stdout != text:
            raise RuntimeError(f'record {rel} is not committed unchanged in HEAD; commit stage 1 before stage 2')
    return json.loads(text)


write_json = ns.write_json
