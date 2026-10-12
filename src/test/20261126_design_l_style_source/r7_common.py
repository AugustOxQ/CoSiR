"""Round 7 foundation (DECISION_RULE.md of this folder: section 9 "Read only" and "Modules", section 10; spec
2026-10-12-r7-design-l-design.md): paths, the SHA-256 tables of section 10, the loader for reused modules, seed roles,
and write-once file writers. Every other round-7 file imports this module first.

Paths. MAIN is /project/CoSiR (absolute). Reused modules and their gitignored results are read from MAIN, never from a
worktree's copy. RESULTS is RF_MAIN/results unless the environment variable R7_RESULTS is set. Smoke output goes under
RESULTS/smoke/chain (chain smoke) and RESULTS/smoke/rule (rule smoke); only files under those two roots may be
overwritten.

Reused code. load(name) imports one module of the CODE table by absolute path under MAIN, once per process, after its
SHA-256 matched. The sys.path set-up is rounds 3 and 6's (MAIN, then round 3, round 2, round 1 at the front), and the
modules are registered under their usual names, so run_checks, run_n6, run_gonogo, common, ... are single instances (as
in r3_bundle.modules()). A name already registered must be this very file, else ImportError.

Rule SHA. RULE_SHA256 is the SHA-256 of DECISION_RULE.md as committed at HEAD of the integration branch (the file the
user approved at 02:34; the spec commit c95b6ea is only named in its header). assert_rule() exits if it changed.
"""
import hashlib
import importlib
import importlib.util
import json
import os
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

# ---------------------------------------------------------------- paths

MAIN = Path("/project/CoSiR")
RF_NAME = "src/test/20261126_design_l_style_source"
RF_MAIN = MAIN / RF_NAME
TEST = MAIN / "src/test"
HERE = Path(__file__).resolve().parent
RESULTS = Path(os.environ["R7_RESULTS"]) if os.environ.get("R7_RESULTS") else RF_MAIN / "results"
SMOKE_CHAIN = RESULTS / "smoke" / "chain"
SMOKE_RULE = RESULTS / "smoke" / "rule"
SMOKE_ROOTS = (SMOKE_CHAIN, SMOKE_RULE)

R1D = TEST / "20261117_reader_fix_csd"
R2D = TEST / "20261118_reader_fix_round2"
R3D = TEST / "20261121_round3_affect_gate"
RULE = RF_MAIN / "DECISION_RULE.md"
RULE_SHA256 = "204a0ac5d0f35afe6cd490662036c35710a8180e02802b3a80f7c45a2c38647d"


def _require(cond, msg, exc=AssertionError):
    if not cond:
        raise exc(msg)


def sha_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def now_ams() -> str:
    """Amsterdam local time, plain (no offset)."""
    return datetime.now(ZoneInfo("Europe/Amsterdam")).strftime("%Y-%m-%d %H:%M")


# ---------------------------------------------------------------- rule section 10 (copied exactly)
# Keys are the paths as the tables write them: absolute, relative to src/test/, or starting with src/ (repo root).

INPUTS = {
    "20261121_round3_affect_gate/results/seed42_arrays.npz":
        "5ea4b09a4161a5eac6ca78942cf0f4b99b9c634edca651fba4c2689c7c24ab8a",
    "20261119_no_caption_csd_spike/results/sp_cand_told_A1_224.npz":
        "08ac1ee86b46375c13c08420e192877eb8c67b54f14304f0a2f0db5e96cd3417",
    "20261119_no_caption_csd_spike/results/sp_cand_R1_A1_224.npz":
        "778a769bbdc55705669bf1f5c0bbd6db7ec5c541dac7abf713301dd5c6478d66",
    "20261117_reader_fix_csd/results/rb_reader_A1.pkl":
        "4e1e4e23c20333b839aa1f92942891bdef51955317640d1014db5d097b7060f6",
    "20261125_artelingo_held_test/results/seed42_per_episode.npz":
        "e93fd7b2b18e788d80605caaa854afa8a25ebda735728f5db956dc322a81b531",
    "20261031_pseudo_partitions/results/partitions.npz":
        "cfd57dbb21216cdc9d1c2e42d1b408648bc18147f62354b52202ade4ab8a9caa",
    "20261116_grouping_step1_style/results/step1_group_style.npz":
        "b04d96b4798acdfdbc9ca87436012755a681c931ac4f7612fa5683350b7a20e2",
    "/data/SSD2/pre_extract/artelingo/style_csd_vitl/embeddings.npy":
        "7eefe522383b753bad32f404651103471f9d5b6c41926ee57bfc7386c84818ad",
    "/data/SSD2/pre_extract/artelingo/style_csd_vitl/meta.json":
        "6013b2f801eb1e6ae104e9fb3c71a377890bb9aab58e4cb27991c269febdf13b",
    "20261030_aspect_baselines/results/codes_provenance.json":
        "8e6a517b73610d4d21b42e2eb39f6fd4118dc4132049364c8e89eecb7716bfaf",
}

CODE = {
    "20261125_artelingo_held_test/r6_common.py":
        "cacbe204add6a6141978af20599f8b2431da78c14040127a373be2a1fd77f3f3",
    "20261125_artelingo_held_test/r6_stats.py":
        "fde578723d7e3e7e8eeacb3a4b0e1902c81cc0e95b4ba154b9ab9ba975ae8aa1",
    "20261121_round3_affect_gate/r3_stats.py":
        "846a4f5b3175302280c20fa415cfff4b0652522d06aa4fc46290f1a1db5de54d",
    "20261125_artelingo_held_test/r6_heads.py":
        "f157e2eda66350047b24c09e06844dc825d3e4061c0f334b76b8d5309fc4e2af",
    "20261125_artelingo_held_test/r6_bundle.py":
        "6b638e5bc2f582358216aea4104ac79de9a78e440560276bae7c42646f030cd9",
    "20261125_artelingo_held_test/r6_score.py":
        "d44bbf150cc8773df9c792baaa232a2825eed4d7605eb95cdf3efe7e3a1f4948",
    "20261125_artelingo_held_test/r6_context.py":
        "0ce2086a2cefa9e75b1c6b8b5733e5052130df0e0cd6f012b3f874de220ba7d7",
    "20261121_round3_affect_gate/r3_common.py":
        "b1a60b1fd801bf9147d0bd58ae6da806d12fbcc982796135454f31dce3836f86",
    "20261121_round3_affect_gate/r3_bundle.py":
        "bef50cbbef17f6c40d53c705a060967950d0690168c4b046b1cc3f0dd9fa0e9d",
    "20261121_round3_affect_gate/r3_fusion.py":
        "ce51a819157842035fdfde488f042c25fa4764ed1b1c363728d5e3837e314637",
    "20261101_aspect_factor_gonogo/run_gonogo.py":
        "8353dbc118cf619494a8e8e5d83bee4f2034ebbaa52946be0bfce92e67ea63dc",
    "20261108_new_method_quick_checks/run_checks.py":
        "5dee3bdf44e526dd6b22580bb5302b41e133e76d4d458224011ca3107dbc4610",
    "20261108_new_method_quick_checks/run_n6.py":
        "57046023af8f4352dc90b563def5e5e27ba35e12489bf7f858a6d0580902f4d0",
    "20261119_no_caption_csd_spike/sp_run.py":
        "d2bf443b21f5af4ad6a343e2b24eac997aeb0e651f6f0d9128919dfd660c995f",
    "20261118_reader_fix_round2/run_r2_fusion.py":
        "fe6fb9ff870c3942bef98327ff23894782522e516fc100cb1635aeec6e7e5b65",
    "20261118_reader_fix_round2/r2_fusion.py":
        "1406e8e71629e74c468f8b009a110e49aa05c481ed52cc1b8d99b59f87016027",
    "20261118_reader_fix_round2/r2_common.py":
        "402c722cc8a94f056ed04b513a1b751f54ac1a8d2f04e91bd0a1226f0b629878",
    "20261117_reader_fix_csd/common.py":
        "99496dcf859ff0b0f746266125c975c5e2f9632e0c2c91d74df17102c65c10a7",
    "20261117_reader_fix_csd/rc_core.py":
        "e649fff5d8253c8ba7aae0dbfff7a68321cef5fe3caa06d7b325a27fe904185c",
    "20261117_reader_fix_csd/rb_build.py":
        "63c7310c890675cc63e49838feb364cdd1c5c68aafdfb0f0f7972128476cba7b",
    "20261117_reader_fix_csd/rb_eval.py":
        "1826e65fd73b8c6cecae33b5df48503bf2cf33e067fc291d5c078c51184e0b14",
    "20261117_reader_fix_csd/rb_features.py":
        "1e758f6d5de1b1dd246592f4045e0ae1dcc520c1d6012540a9e69b3314fb6caf",
    "20261116_grouping_step1_style/run_step1.py":
        "f4bea509a6ca4fbb48e60823b4add1ba89570b626416127aaf57db66d63018c0",
    "20261111_community_told_oracle/run_told_oracle.py":
        "b7ae64175f259bf17edf503a57c5befd47edb938bd4d241aa732abf14ffe464d",
    "20261112_community_sweep/run_sweep.py":
        "13314f6f17edc4fe8665e229608300f6ff38a50025ce321823260f3fab0681b1",
    "20261030_aspect_baselines/run_baselines.py":
        "26508dde35f77850c4a80e98d08b0ea6576638577c84d3e96c39660ef02e63c4",
    "src/train/pseudo_partitions.py":
        "8901cc668fe3fd1834dc2f0a00851030dabb1135bd034a26646629c3621e71f8",
    "src/eval/aspect_metrics.py":
        "ad95485ef887089d74bb685e2cdb1aab7057b87ad62d926f5c252219f4c6c2f1",
    "src/eval/aspect_quick_checks.py":
        "fcd116f529082f81e163092921a91909bd358eb3347695a959c68f814dc7d985",
    "src/eval/aspect_nested.py":
        "fadd1fd5d6fe8920a19ff8ccddd91b0e487f286bf908428c31183c503b4e7fa3",
    "src/model/aspect_rule.py":
        "f1d83bd9a7eb535a054c3221d565b8b38514f1e85590bbde0fd2be8cd502fe34",
}


def path_of(key) -> Path:
    """Absolute path of a table key."""
    if key.startswith("/"):
        return Path(key)
    return MAIN / key if key.startswith("src/") else TEST / key


def rule_sha() -> str:
    """SHA-256 of RF_MAIN/DECISION_RULE.md."""
    return sha_file(RULE)


def assert_rule():
    """Every round-7 script calls this first: the rule must be the committed one."""
    got = rule_sha()
    if got != RULE_SHA256:
        raise SystemExit(f"DECISION_RULE.md differs from the committed version: {got} != {RULE_SHA256}")
    return got


_CHECKED = {}


def _check(table, label):
    out = {}
    for key, want in table.items():
        if key not in _CHECKED:
            p = path_of(key)
            if not p.is_file():
                raise SystemExit(f"{label} {key}: {p} does not exist")
            got = sha_file(p)
            if got != want:
                raise SystemExit(f"{label} {key}: SHA-256 {got} differs from the rule's {want}")
            _CHECKED[key] = got
        out[key] = _CHECKED[key]
    return out


def assert_inputs(fresh=False):
    """Check every INPUTS and CODE file against section 10; stop (SystemExit) on any mismatch. Returns
    {key: sha256} for what it checked. fresh=True re-hashes (the end-of-stage-0 check); otherwise a file already
    verified in this process is not hashed again."""
    if fresh:
        _CHECKED.clear()
    return {**_check(INPUTS, "input"), **_check(CODE, "code")}


# ---------------------------------------------------------------- the reused modules

def to_front(paths):
    """Put each of ``paths`` once at the front of sys.path, in the given order (the env's editable install lists MAIN
    last, so in a worktree a bare `import src` would find the worktree's copy)."""
    paths = [str(p) for p in paths]
    real = {os.path.realpath(p) for p in paths}
    sys.path[:] = paths + [e for e in sys.path if not (e and os.path.realpath(e) in real)]


def _module_name(key) -> str:
    """'src.eval.aspect_metrics' for repo-root files, else the file stem."""
    if key.startswith("src/"):
        return key[:-3].replace("/", ".")
    return Path(key).stem


_NAMES = {_module_name(k): k for k in CODE if k.endswith(".py")}
_LOADED = {}


def load(name):
    """The reused module ``name`` (stem such as 'run_checks', or dotted such as 'src.eval.aspect_metrics'), imported
    by absolute path under MAIN, once per process. Its file SHA-256 is asserted against CODE before first use."""
    if name in _LOADED:
        return _LOADED[name]
    _require(name in _NAMES, f"{name!r} is not a module of rule section 10's code table", KeyError)
    key = _NAMES[name]
    path = path_of(key).resolve()
    _check({key: CODE[key]}, "code")
    to_front((MAIN, R3D, R2D, R1D))
    mod = sys.modules.get(name)
    if mod is None:
        if name.startswith("src."):
            mod = importlib.import_module(name)
        else:
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            try:
                spec.loader.exec_module(mod)
            except BaseException:
                sys.modules.pop(name, None)
                raise
    if Path(mod.__file__).resolve() != path:
        raise ImportError(f"module {name!r} is {mod.__file__}, not {path}")
    src = sys.modules.get("src")
    if src is not None and not Path(src.__file__).resolve().is_relative_to((MAIN / "src").resolve()):
        raise ImportError(f"package 'src' is {src.__file__}, not under {MAIN / 'src'}")
    to_front((MAIN,))           # reused modules insert their own folders at the front; MAIN goes back to first
    _LOADED[name] = mod
    return mod


# ---------------------------------------------------------------- seed roles

DEV_SEED = 42
TEST_SEEDS = (55, 56, 57)
SMOKE_SEEDS = (9001, 9002, 9003)
ALLOWED = (DEV_SEED,) + TEST_SEEDS + SMOKE_SEEDS


@dataclass(frozen=True)
class Roles:
    """One run kind: its development seed, test seeds, smoke flag and the folder its output goes to."""
    name: str
    dev: int
    tests: tuple
    smoke: bool
    root: Path


def roles(kind) -> "Roles":
    """kind: 'real', 'chain' (chain smoke: dev 9001, tests 9002 and 9003) or 'rule' (rule smoke: seeds 9001 to 9003
    on frozen real objects)."""
    if kind == "real":
        return Roles("real", DEV_SEED, TEST_SEEDS, False, RESULTS)
    if kind == "chain":
        return Roles("chain", SMOKE_SEEDS[0], SMOKE_SEEDS[1:], True, SMOKE_CHAIN)
    if kind == "rule":
        return Roles("rule", SMOKE_SEEDS[0], SMOKE_SEEDS[1:], True, SMOKE_RULE)
    raise ValueError(f"unknown run kind {kind!r}")


def check_seed(seed, smoke):
    """A real run uses 42, 55, 56, 57; a smoke run only 9001 to 9003 (never a result, C4)."""
    ok = SMOKE_SEEDS if smoke else (DEV_SEED,) + TEST_SEEDS
    if seed not in ok:
        raise ValueError(f"seed {seed} (smoke {smoke}) is not one of this round's seeds {ok}")
    return seed


def assert_threads():
    """OMP and MKL at 8 threads, numpy 2.2.6, scikit-learn 1.6.1."""
    import sklearn
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        _require(os.environ.get(v) == "8", f"{v} is {os.environ.get(v)!r}, not '8'")
    _require(np.__version__ == "2.2.6", f"numpy {np.__version__}, not 2.2.6")
    _require(sklearn.__version__ == "1.6.1", f"scikit-learn {sklearn.__version__}, not 1.6.1")


# ---------------------------------------------------------------- write-once files

def in_smoke(path) -> bool:
    p = Path(path).resolve()
    return any(p.is_relative_to(r.resolve()) for r in SMOKE_ROOTS)


def fix_path(path) -> Path:
    """The next free ``<name>_fix<n>.<ext>`` beside ``path`` (rule section 8: a corrected run never overwrites)."""
    path = Path(path)
    n = 1
    while True:
        cand = path.with_name(f"{path.stem}_fix{n}{path.suffix}")
        if not cand.exists():
            return cand
        n += 1


def _prepare(path):
    path = Path(path)
    if path.exists() and not in_smoke(path):
        raise SystemExit(f"{path} exists; refusing to overwrite (use fix_path for a corrected run)")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, Path):
        return str(x)
    return x


def write_json_once(path, rec):
    """JSON with top-level ``rule_sha256`` and ``written`` (Amsterdam time) added; returns the record written."""
    path = _prepare(path)
    _require(isinstance(rec, dict), "a JSON record must be a dict")
    _require("rule_sha256" not in rec and "written" not in rec, "record already has rule_sha256 or written")
    rec = {**jsonable(rec), "rule_sha256": rule_sha(), "written": now_ams()}
    path.write_text(json.dumps(rec, indent=1))
    return rec


def write_npz_once(path, **arrays):
    path = _prepare(path)
    with open(path, "wb") as f:
        np.savez(f, **arrays)
    return path


def write_pickle_once(path, obj):
    path = _prepare(path)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    return path


def main():
    """Self-check: the rule, every input and every code file match section 10."""
    assert_rule()
    got = assert_inputs(fresh=True)
    n_in = sum(k in INPUTS for k in got)
    n_code = sum(k in CODE for k in got)
    print(f"r7_common: rule SHA ok; {n_in}/{len(INPUTS)} inputs and {n_code}/{len(CODE)} code files matched")


if __name__ == "__main__":
    main()
