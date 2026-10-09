"""Round 6 foundation (DECISION_RULE.md of this folder: section 5 items 1 and 2, section 6 item 1, section 10 items 1
and 2): paths, the earlier rounds' modules, the input SHA-256s, the held split and the development value sets.

Paths. HERE is this folder wherever it is checked out (the main checkout or a git worktree). MAIN is the main checkout,
found through git's common directory, so a worktree reads the gitignored inputs (earlier rounds' results, caches and
checkpoints) and imports the earlier round folders from MAIN, while this round's own outputs go under HERE/results.

Imports. At import MAIN and the earlier round folders are moved to the front of sys.path (MAIN first, each once, even
when already present: the env's editable install lists MAIN last, behind the cwd). Round 3's r3_common (which brings
round 1's common, rc_core, rb_build, rb_eval, rb_features and round 2's r2_fusion), r3_fusion and r3_stats are then
imported under their usual names; run_checks, run_n6 and run_told_oracle are loaded by path through
r3_bundle.modules() (one instance each). Every one of them must resolve to its folder under MAIN, `src` and every
`src.*` module must resolve under MAIN/src, and in a worktree no loaded module may come from the worktree's src/
outside HERE; otherwise the import stops (ImportError). r3_bundle.build_bundle is never used by round 6 (its seed guard
and EvalContext admit selection rows only).

Held rows. load_split() reads only the four index keys of prepare.npz and only `held_rows` of held_codes.npz (round 5
read whole files once; never `dict(np.load(...))` here). No function of this module reads a feature or label of a held
row beyond what the split itself touches; assert_held_eligible is called only by the held runner.

Guards carry a `# guard:<name>` marker; the tests delete each on a copy and show that its scenario then passes.
"""
import ast
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np


def _require(cond, msg, exc=AssertionError):
    if not cond:
        raise exc(msg)


def main_checkout(start) -> Path:
    """The main checkout of the git repository holding ``start``: the parent of git's common directory (a worktree
    and the main checkout give the same answer)."""
    out = subprocess.run(["git", "-C", str(start), "rev-parse", "--path-format=absolute", "--git-common-dir"],
                         capture_output=True, text=True)
    if out.returncode != 0 or not out.stdout.strip():
        raise RuntimeError(f"{start} is not inside a git checkout: {out.stderr.strip()}")
    return Path(out.stdout.strip()).resolve().parent


# ---------------------------------------------------------------- paths

HERE = Path(__file__).resolve().parent
CHECKOUT = HERE.parents[2]                     # the checkout (main or worktree) that holds HERE
MAIN = main_checkout(HERE)
TEST = MAIN / "src/test"
_require(TEST.is_dir(), f"{TEST} does not exist: MAIN={MAIN} is not CoSiR's main checkout")  # guard:main_src_test
RESULTS = HERE / "results"
SMOKE = RESULTS / "smoke"

R1D = TEST / "20261117_reader_fix_csd"
R2D = TEST / "20261118_reader_fix_round2"
R3D = TEST / "20261121_round3_affect_gate"
R4D = TEST / "20261122_round4_aff_vetoes"
QC = TEST / "20261108_new_method_quick_checks"
TO = TEST / "20261111_community_told_oracle"
GG = TEST / "20261101_aspect_factor_gonogo"
FIX = TEST / "20261109_fix_diagnostics"
AB = TEST / "20261030_aspect_baselines"
S1 = TEST / "20261116_grouping_step1_style"
STAGE_D_SEL = TEST / "20261013_stage_d_selection"
STAGE_D_FINAL = TEST / "20261014_stage_d_final"



def to_front(paths):
    """Put each of ``paths`` exactly once at the front of sys.path, in the given order, removing any earlier entry
    for the same folder. The CoSiR env's editable install lists MAIN last, behind the cwd, so in a worktree a bare
    `import src` would otherwise find the worktree's own copy."""
    paths = [str(p) for p in paths]
    real = {os.path.realpath(p) for p in paths}
    sys.path[:] = paths + [e for e in sys.path if not (e and os.path.realpath(e) in real)]


to_front((MAIN, R3D, R2D, R1D))


def check_src_loaded_early():
    """A `src` package already imported (before r6_common) must be MAIN's; otherwise later imports would silently
    use it, so stop here with a clear message."""
    mod = sys.modules.get("src")
    if mod is None:
        return
    f = getattr(mod, "__file__", None)
    if not f or not Path(f).resolve().is_relative_to((MAIN / "src").resolve()):
        raise ImportError(f"package 'src' was imported from {f} before r6_common; it must come from {MAIN / 'src'}"
                          " (import r6_common before anything from src)")


check_src_loaded_early()  # guard:src_early

import r3_common as R3  # noqa: E402  (puts round 1, round 2 and MAIN on sys.path; imports round 1 and round 2)
import r3_fusion as RF  # noqa: E402
import r3_stats as RS  # noqa: E402
import r3_bundle as _R3B  # noqa: E402  only for modules(); its build_bundle is never used by round 6

_MODS = _R3B.modules()
C = R3.C                 # round 1's common
RC = R3.K                # round 1's rc_core
rb = R3.rb               # round 1's rb_build
rbe = R3.rbe             # round 1's rb_eval
rf = R3.rf               # round 1's rb_features
F2 = R3.F                # round 2's r2_fusion
run_checks = _MODS.rc    # QC's run_checks (by path)
run_n6 = _MODS.n6
rto = _MODS.rto          # run_told_oracle
rg = _MODS.rg            # run_gonogo (imported by run_checks)

# module name -> the folder under MAIN it must come from
EXPECTED_HOME = {
    "r3_common": R3D,
    "r3_fusion": R3D,
    "r3_stats": R3D,
    "r3_bundle": R3D,
    "common": R1D,
    "rc_core": R1D,
    "rb_build": R1D,
    "rb_eval": R1D,
    "rb_features": R1D,
    "r2_fusion": R2D,
    "run_checks": QC,
    "run_n6": QC,
    "run_n6c": QC,
    "run_told_oracle": TO,
    "run_gonogo": GG,
    "diagnose_counterparts": FIX,
    "diagnose_fixes": FIX,
}


def check_homes(expected=None):
    """Each module of ``expected`` (default EXPECTED_HOME) is loaded and its file lies in its expected folder."""
    for name, folder in (expected or EXPECTED_HOME).items():
        mod = sys.modules.get(name)
        if mod is None:
            raise ImportError(f"module {name!r} is not loaded")
        got = Path(mod.__file__).resolve().parent
        if got != Path(folder).resolve():
            raise ImportError(f"module {name!r} resolved to {mod.__file__}, not to {folder}")




def check_sources():
    """`src` and every `src.*` module come from MAIN/src, and (in a worktree) no loaded module comes from this
    checkout's src/ tree outside HERE (the worktree's copies of src and of the earlier round folders)."""
    main_src = (MAIN / "src").resolve()
    here, own = HERE.resolve(), (CHECKOUT / "src").resolve()
    if "src" not in sys.modules:
        raise ImportError("package 'src' is not loaded")
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        f = Path(f).resolve()
        if (name == "src" or name.startswith("src.")) and not f.is_relative_to(main_src):
            raise ImportError(f"module {name!r} resolved to {f}, not under {main_src}")
        if own != main_src and f.is_relative_to(own) and not f.is_relative_to(here):
            raise ImportError(f"module {name!r} resolved to {f}, in this checkout's src/ instead of MAIN's")


check_homes(EXPECTED_HOME)  # guard:homes
for _m, _name in ((R3, "r3_common"), (RF, "r3_fusion"), (RS, "r3_stats"), (C, "common"), (RC, "rc_core"),
                  (rb, "rb_build"), (rbe, "rb_eval"), (rf, "rb_features"), (F2, "r2_fusion"),
                  (run_checks, "run_checks"), (run_n6, "run_n6"), (rto, "run_told_oracle"), (rg, "run_gonogo")):
    _require(sys.modules[_name] is _m, f"{_name} is loaded twice", ImportError)

from src.data.artelingo_splits import (EMOTION_CATCH_ALL, EXPECTED_SIZES, SPLIT_SEED,  # noqa: E402
                                       artelingo_splits, encode_labels)
from src.data.splits import grouped_split  # noqa: E402
from src.data.wikiart_genre import GENRE_NAMES  # noqa: E402
from src.eval.aspect_episodes import eligible_values  # noqa: E402

to_front((MAIN,))           # the earlier modules insert their own folders at the front; MAIN goes back to first
check_sources()  # guard:sources

# ---------------------------------------------------------------- constants (contracts section 1)

RULE_SHA256 = "7444a5e338838d837673b82b047c82b033eb1e2d1e0c3ed4e388dd8673070724"
R3_RULE_SHA256 = "2d311dbeac1a561fc2a719b730075c30819f4f7f3a592041cb64b65821dc5925"
PAIRS = (("emotion", "style", "genre"), ("emotion", "genre", "style"), ("style", "genre", "emotion"))
PAIR_NAMES = ("emotion__style", "emotion__genre", "style__genre")
ASPECTS = ("emotion", "style", "genre")
DEV_SEED = 42
HELD_SEEDS = (52, 53, 54)
SMOKE_SEEDS = (9001, 9002, 9003)
N_PER_PAIR = 4096
N_SMOKE = 64
AFF_HITS_SEED42 = 9406
N_RANKINGS_SEED42 = 49152
N_ROWS = 308_723
MIN_PAINTINGS = 30
VALUE_COUNTS = {"emotion": 8, "style": 23, "genre": 10}    # rule section 5 item 2

_require(PAIR_NAMES == tuple(f"{a}__{b}" for a, b, _ in PAIRS), "PAIR_NAMES do not follow PAIRS", ImportError)
_require(all(set(p) == set(ASPECTS) for p in PAIRS), "every pair must cover the three aspects", ImportError)
_require(R3.RULE_SHA == R3_RULE_SHA256, "r3_common's RULE_SHA is not the R3 rule's", ImportError)


def sha256_bytes(b) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def amsterdam_now() -> str:
    """Amsterdam local time, plain: YYYY-MM-DD HH:MM:SS."""
    return datetime.now(ZoneInfo("Europe/Amsterdam")).strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------- inputs (rule section 6 item 1)

RULE_REL = "20261125_artelingo_held_test/DECISION_RULE.md"       # read from HERE, keys relative to src/test/
R3_RULE_REL = "20261121_round3_affect_gate/DECISION_RULE.md"
PREPARE_REL = "20261013_stage_d_selection/cache/prepare.npz"
HELD_CODES_REL = "20261014_stage_d_final/cache/held_codes.npz"
PREPARE_KEYS = ("groups", "split_train", "scorer_train", "selection")

_require(len(R3.INPUTS) == 36, f"r3_common.INPUTS has {len(R3.INPUTS)} entries, not R3 rule D15's 36", ImportError)
EXTRA_INPUTS = {
    # rule section 6 item 1: the three extra files
    "20261116_grouping_step1_style/results/step1_group_style.npz":
        "b04d96b4798acdfdbc9ca87436012755a681c931ac4f7612fa5683350b7a20e2",
    "20261116_grouping_step1_style/results/step1_heads_style.npz":
        "898a37017d82d3e130b20155e90f69e51f82f30892e04316dcb56d7aaaf8df8b",
    "20261122_round4_aff_vetoes/results/dev_seed42.json":
        "fd7b3f480d5997284f9d8cfe29fe25275ff3edd38c1ada0a9392d8396ba9e01f",
    # rule section 5 item 1 (prepare.npz) and contracts section 1 (held_codes.npz: observed, stricter than the rule)
    PREPARE_REL: "d30f0281bb521d18c7a4d4adca1a7938689e2789db7fd528ce0322b90ac44c8b",
    HELD_CODES_REL: "2901f875e7ad59f9f54509b61352a9cecb0eba310b31357f872219f25e2a12e9",
    # rule section 6 item 4: round 3's per-anchor arrays
    "20261121_round3_affect_gate/results/seed42_arrays.npz":
        "5ea4b09a4161a5eac6ca78942cf0f4b99b9c634edca651fba4c2689c7c24ab8a",
    # rule section 6 item 4: AB's smoke baselines (episode SHAs of seeds 9001 to 9003); observed 2026-10-09, the rule
    # states no file SHA
    "20261030_aspect_baselines/results/smoke/baselines_seed9001.json":
        "729cb5653cbcf2ea97da6dad65da9eaddf5d1b4bbace5c9d551f9c1d4fe8c517",
    "20261030_aspect_baselines/results/smoke/baselines_seed9002.json":
        "8e86b7e6da2411743d09c28154c2189ad50b11fd4a04f2bd76e53e80808ba352",
    "20261030_aspect_baselines/results/smoke/baselines_seed9003.json":
        "e7e0ac0ed30cd170d0de1ccda56d9e6a2f6a663576725dcd6306b46e3429aada",
    # round 3's own modules, imported above (stricter than the rule): the SHAs round 4's rule D11 recorded
    "20261121_round3_affect_gate/r3_common.py": "b1a60b1fd801bf9147d0bd58ae6da806d12fbcc982796135454f31dce3836f86",
    "20261121_round3_affect_gate/r3_bundle.py": "bef50cbbef17f6c40d53c705a060967950d0690168c4b046b1cc3f0dd9fa0e9d",
    "20261121_round3_affect_gate/r3_fusion.py": "ce51a819157842035fdfde488f042c25fa4764ed1b1c363728d5e3837e314637",
    "20261121_round3_affect_gate/r3_stats.py": "846a4f5b3175302280c20fa415cfff4b0652522d06aa4fc46290f1a1db5de54d",
}
INPUT_SHA256 = {RULE_REL: RULE_SHA256, R3_RULE_REL: R3_RULE_SHA256, **R3.INPUTS}
for _k, _v in EXTRA_INPUTS.items():
    _require(INPUT_SHA256.setdefault(_k, _v) == _v, f"{_k}: two different SHA-256s", ImportError)
# where each input is read (patched by the tests to point at a copy)
INPUT_PATHS = {k: (HERE / "DECISION_RULE.md" if k == RULE_REL else TEST / k) for k in INPUT_SHA256}
_SHA_CACHE = {}


def _sha_cached(path) -> str:
    p = Path(path).resolve()
    st = p.stat()
    key = (str(p), st.st_size, st.st_mtime_ns)
    if key not in _SHA_CACHE:
        _SHA_CACHE[key] = sha256_file(p)
    return _SHA_CACHE[key]


def assert_input(rel) -> str:
    """SHA-256 of one input (key of INPUT_SHA256); a mismatch stops the work (SystemExit)."""
    got, want = _sha_cached(INPUT_PATHS[rel]), INPUT_SHA256[rel]
    _require(got == want, f"input {rel}: SHA-256 {got} differs from {want}", SystemExit)  # guard:input_sha
    return got


def assert_inputs() -> dict:
    """Every input of rule section 6 item 1 (and contracts section 1) asserted; {path relative to src/test/: sha}."""
    return {rel: assert_input(rel) for rel in INPUT_SHA256}


def r6_module_shas(here=None) -> dict:
    """SHA-256 of every r6_*.py and run_r6_*.py in this folder (tests excluded), dts_settings.json, and every
    scripts/run_r6_*.sh and scripts/das6_sync_r6.py of the checkout that holds this folder. Keys are paths relative
    to that checkout."""
    here = Path(here or HERE).resolve()
    root = here.parents[2]
    files = sorted(here.glob("r6_*.py")) + sorted(here.glob("run_r6_*.py"))
    files += [p for p in (here / "dts_settings.json",) if p.is_file()]
    files += sorted((root / "scripts").glob("run_r6_*.sh"))
    files += [p for p in (root / "scripts/das6_sync_r6.py",) if p.is_file()]
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in files}


# ---------------------------------------------------------------- the held split (rule section 5 item 1)

def _read_npz_keys(path, keys) -> dict:
    """Only ``keys`` of an npz (never the whole file)."""
    out = {}
    with np.load(path) as z:
        for k in keys:
            _require(k in z.files, f"{path} has no key {k!r}")
            out[k] = np.asarray(z[k])
    return out


def check_split(groups, train, val, held, scorer_train, selection, prepared, held_rows):
    """Rule section 5 item 1's assertions on a recomputed split (H3's recompute_split plus round 6's)."""
    for name, mine in (("groups", groups), ("split_train", train), ("scorer_train", scorer_train),
                       ("selection", selection)):
        _require(np.array_equal(mine, prepared[name]),
                 f"recomputed {name} differs from prepare.npz")  # guard:prepared
    _require(np.array_equal(held, held_rows),
             "recomputed held rows differ from held_codes.npz held_rows")  # guard:held_rows
    _require(np.intersect1d(held, train).size == 0 and np.intersect1d(held, val).size == 0,
             "held rows overlap train or val rows")  # guard:held_disjoint
    _require(np.intersect1d(groups[held], groups[train]).size == 0
             and np.intersect1d(groups[held], groups[val]).size == 0,
             "a leakage group spans held and train or val")  # guard:leakage
    _require(np.intersect1d(held, selection).size == 0, "held rows overlap selection rows")  # guard:held_selection
    _require(np.intersect1d(held, scorer_train).size == 0, "held rows overlap scorer-train rows")  # guard:held_scorer
    _require(np.intersect1d(scorer_train, selection).size == 0
             and np.array_equal(np.union1d(scorer_train, selection), np.sort(train)),
             "scorer_train and selection do not partition train")  # guard:subsplit


def load_split(data) -> SimpleNamespace:
    """The stage (d) split recomputed from ``data`` (load_artelingo()), with every assertion of rule section 5 item 1.

    groups, scorer_train, selection, val and held come from artelingo_splits (which asserts the sizes); train is
    grouped_split(groups, seed=42).train (artelingo_splits keeps none). All are int64 row positions (C6)."""
    assert_input(PREPARE_REL)  # guard:split_inputs
    assert_input(HELD_CODES_REL)  # guard:split_inputs
    _require(len(data.paintings) == len(data.sample_ids) == N_ROWS, "data does not have ArtELingo's 308,723 rows")
    sp = artelingo_splits(data)
    gs = grouped_split(sp.groups, seed=SPLIT_SEED)
    _require(len(gs.train) == EXPECTED_SIZES["train"], "train size differs from the stage (d) split")
    _require(np.array_equal(gs.val, sp.val) and np.array_equal(gs.held, sp.held),
             "grouped_split and artelingo_splits disagree on val or held")
    prepared = _read_npz_keys(INPUT_PATHS[PREPARE_REL], PREPARE_KEYS)
    held_rows = _read_npz_keys(INPUT_PATHS[HELD_CODES_REL], ("held_rows",))["held_rows"]
    train = np.asarray(gs.train, dtype=np.int64)
    check_split(sp.groups, train, sp.val, sp.held, sp.scorer_train, sp.selection, prepared, held_rows)
    return SimpleNamespace(groups=np.asarray(sp.groups, dtype=np.int64), train=train,
                           val=np.asarray(sp.val, dtype=np.int64), held=np.asarray(sp.held, dtype=np.int64),
                           scorer_train=np.asarray(sp.scorer_train, dtype=np.int64),
                           selection=np.asarray(sp.selection, dtype=np.int64))


# ---------------------------------------------------------------- value sets (rule section 5 item 2)

def _pair_pool(labels, rows, a, b, third):
    """The pool build_aspect_episodes forms: rows with a, b and the third aspect known."""
    rows = np.asarray(rows, dtype=np.int64)
    la, lb, lt = (np.asarray(labels[x], dtype=np.int64) for x in (a, b, third))
    return rows[(la[rows] >= 0) & (lb[rows] >= 0) & (lt[rows] >= 0)]


def _values_on(labels, groups, rows) -> dict:
    """eligible_values of each aspect on the pool of ``rows``, formed per pair as build_aspect_episodes forms it."""
    pools = [_pair_pool(labels, rows, a, b, t) for a, b, t in PAIRS]
    _require(all(np.array_equal(p, pools[0]) for p in pools), "the pairs' pools differ")  # guard:pair_pools
    pool, groups = pools[0], np.asarray(groups)
    out = {x: eligible_values(np.asarray(labels[x], dtype=np.int64), groups, pool, MIN_PAINTINGS) for x in ASPECTS}
    for a, b, _ in PAIRS:      # per pair, as the rule states it; the same lists because the pool is shared
        _require(eligible_values(np.asarray(labels[a], dtype=np.int64), groups, pool, MIN_PAINTINGS) == out[a]
                 and eligible_values(np.asarray(labels[b], dtype=np.int64), groups, pool, MIN_PAINTINGS) == out[b],
                 f"{a}__{b}: per-pair values differ")
    return out


def development_value_sets(labels, groups, selection) -> dict:
    """Rule section 5 item 2: {aspect: ascending codes} eligible (>= 30 paintings) on the selection pool."""
    out = _values_on(labels, groups, selection)
    got = {x: len(out[x]) for x in ASPECTS}
    _require(got == VALUE_COUNTS,
             f"development value counts {got} differ from the rule's {VALUE_COUNTS}")  # guard:value_counts
    return out


def value_names(data) -> dict:
    """Code -> name per aspect, in artelingo_aspect_labels' coding."""
    emotion = encode_labels(data.emotions, exclude=(EMOTION_CATCH_ALL,))[1]
    style = encode_labels(data.art_styles)[1]
    _require("something else" not in emotion, "the emotion catch-all 'something else' has a code")  # guard:catch_all
    return {"emotion": list(emotion), "style": list(style), "genre": list(GENRE_NAMES)}


def write_value_sets(path, value_sets, data) -> dict:
    """JSON {aspect: [{"code": int, "name": str}]} (rule section 5 item 2's results/value_sets.json)."""
    names = value_names(data)
    rec = {}
    for x in ASPECTS:
        codes = [int(c) for c in value_sets[x]]
        _require(all(0 <= c < len(names[x]) for c in codes), f"{x}: a code outside the name list")
        rec[x] = [{"code": c, "name": str(names[x][c])} for c in codes]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1))
    return rec


def assert_held_eligible(labels, groups, held, value_sets):
    """Rule section 5 item 2: every development value passes eligible_values (>= 30 paintings) on the held pool.
    Called only by the held runner; tests use synthetic labels."""
    ok = _values_on(labels, groups, held)
    bad = {x: [int(v) for v in value_sets[x] if int(v) not in set(ok[x])] for x in ASPECTS}
    _require(not any(bad.values()), f"development values not eligible on held rows: {bad}")  # guard:held_eligible


# ---------------------------------------------------------------- a check used by the tests

def parse_baselines_pairs(path=None):
    """run_baselines.PAIRS read from the file's source (importing it sets torch threads and has side effects)."""
    tree = ast.parse(Path(path or AB / "run_baselines.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "PAIRS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise LookupError("run_baselines.py assigns no PAIRS")
