"""Round 8 foundation (DECISION_RULE.md of this folder: M1, M2, M10, section 9, section 10; constitution C6): paths,
the SHA-256 table of section 10, the loader for round-7 modules, the shared row/painting/image joins, seed roles,
the thread check and write-once file writers. Every other r8 file imports this module first and takes every join
from it, never recomputing one.

Paths. MAIN is /project/CoSiR (absolute, as R7). RF, RESULTS, SMOKE and RULE are the main checkout's, so results land
there from any worktree; this file may run from a worktree. R8_RESULTS (environment) moves RESULTS, for tests only.

Rows and paintings (M1). Global rows are positions in load_artelingo(); scorer_train are the 183,694 scorer-train
rows (ascending int64). The painting index p is run_step1.base_data()'s g_st code = position in round 1's
rb_halves.npz paintings_sorted (ascending leakage-group code of the scorer-train rows), NOT painting-id string order;
the two are asserted to agree. A painting's vector in any source is that of its first scorer-train row. Image-level
sources join only through style_csd_vitl/row_to_image.npy, after the C6 checks of r7_groupings.read_source.

Images (M10). nonheld_images() = the ascending CSD image indices of rows outside the held split (49,121), equal to
the 224 cache's image set. The *_pos maps give positions in that array, i.e. rows of a space file (phi[j] belongs to
image nonheld_images()[j]).

Everything returned from a cached function is read-only (shared inside the process); copy before changing.
"""
import hashlib
import importlib.util
import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np

# ---------------------------------------------------------------- paths

MAIN = Path("/project/CoSiR")
RF_NAME = "src/test/20261129_r8_grouping"
RF = MAIN / RF_NAME
TEST = MAIN / "src/test"
HERE = Path(__file__).resolve().parent
RESULTS = Path(os.environ["R8_RESULTS"]) if os.environ.get("R8_RESULTS") else RF / "results"
SMOKE = RESULTS / "smoke"
RULE = RF / "DECISION_RULE.md"
# rule gap: the rule has no SHA of its own, so (as R7) this constant is the committed file's SHA-256 (commit 9fb9d68);
# a deliberate edit of DECISION_RULE.md must update it in the same commit.
RULE_SHA256 = "50eb65addbaeade7cd85b12ea3153da5143fe0f515c1648b8133229088a9d50e"

R7_NAME = "20261126_design_l_style_source"
R7D = TEST / R7_NAME
R1D = TEST / "20261117_reader_fix_csd"
STYLE = Path("/data/SSD2/pre_extract/artelingo/style_csd_vitl")
CACHE224 = Path("/data/SSD2/pre_extract/artelingo_clip224")
N_ROWS = 308_723
N_PAINTINGS = 36_518
N_SCORER_TRAIN = 183_694
N_NONHELD_IMAGES = 49_121
N_CSD_IMAGES = 61_402


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
# Keys: absolute paths, or paths relative to src/test/ (R7 = 20261126_design_l_style_source, written out).

_R7R = f"{R7_NAME}/results"
_HF = Path("/root/.cache/huggingface/hub")
INPUTS = {
    "20261121_round3_affect_gate/results/seed42_arrays.npz":
        "5ea4b09a4161a5eac6ca78942cf0f4b99b9c634edca651fba4c2689c7c24ab8a",
    f"{_R7R}/seed42/arrays.npz": "b8678e21ce2f1b005b6fd0c2d077c05f6b869847dab02082ad8871f3e6a27fa1",
    f"{_R7R}/seed42/reader_R0.npz": "69d688ad72afef5b4c920337c7f6462d2ba23c0faab27b1f2d29ef683a63093c",
    f"{_R7R}/taus.json": "09d1196274b162e2aff8d6b021c570901ff6fe9afbee66e3875b6cf1b83be1da",
    f"{_R7R}/readers/bank_R0_half0.npz": "a499041d94b2396d28c586d1027bfe9fbd39d733334319ca8e2fe1ebec83bac7",
    f"{_R7R}/readers/bank_R0_half1.npz": "887c80658ea295c72e21f2724ac8b289cbf9aee6789ffa5948f892f94ade87fd",
    f"{_R7R}/readers/reader_R0.json": "742311d4013825a6614109130027a7a5ddb5fb95703b8b87b31d987da78384ae",
    f"{_R7R}/heads/post_R0.npz": "0e547fb9a97815da5f7b512652c5c2508d509430341abaf55ac0ed6e9b8bae19",
    f"{_R7R}/heads/xheads_R0.npz": "31246c1e83b2e51a1e21e8192230103856f5f91ee5194440675a6d8af984010a",
    f"{_R7R}/heads/post_affect.npz": "936a28adb461ad2713ff13af89f31310fadc33d3dd1b9b2d9cdadb3e3d70ddfd",
    f"{_R7R}/heads/post_image.npz": "871ed7c577dd42d2d6988ee186cfef5b1a714545b54e02ba32bd7408ab8cd84d",
    f"{_R7R}/heads/post_caption.npz": "116b0b150967972ce7dc75298b9e190ab7787816aa55187b27b53af747533e66",
    f"{_R7R}/heads/xheads_affect.npz": "fb1327205e055040c97a5001505149fd3819d54ae871dd79ade16d35f2e3a838",
    f"{_R7R}/heads/xheads_image.npz": "1a1f35d82b371250952422b49d69f578fc8aa114a5f06645fdd4f8d4fc6caff6",
    f"{_R7R}/heads/xheads_caption.npz": "0d887a4e386ff47d17bb1b95ae8c089553920de9dd8c76442629b176fea84983",
    "20261031_pseudo_partitions/results/partitions.npz":
        "cfd57dbb21216cdc9d1c2e42d1b408648bc18147f62354b52202ade4ab8a9caa",
    "20261116_grouping_step1_style/results/step1_group_style.npz":
        "b04d96b4798acdfdbc9ca87436012755a681c931ac4f7612fa5683350b7a20e2",
    "/data/SSD2/pre_extract/artelingo/style_csd_vitl/embeddings.npy":
        "7eefe522383b753bad32f404651103471f9d5b6c41926ee57bfc7386c84818ad",
    "/data/SSD2/pre_extract/artelingo/style_csd_vitl/meta.json":
        "6013b2f801eb1e6ae104e9fb3c71a377890bb9aab58e4cb27991c269febdf13b",
    "/data/SSD2/pre_extract/artelingo/style_csd_vitl/row_to_image.npy":
        "914f4e621ffecd4f0bcb7bc13465544553f7d48646668f02a0ffe9438050ab10",
    "/data/SSD2/pre_extract/artelingo_clip224/images_uint8.npy":
        "0e56913f818e4604805a6b3a778c73ad717659b63037b632337b5e3cf4fcdb53",
    "/data/SSD2/pre_extract/artelingo_clip224/paintings.json":
        "3ea9522c7e86f14e75b219d66318ceae55d8953a0bb3e2698a1da92c6477a7d8",
    "/data/SSD2/pre_extract/artelingo/features/metadata.json":
        "43963ca92f46ee638ace94f8ff72807de29ec87471aa5b68904bdc4afb7e8011",
    "/data/SSD2/third_party/FCMI/networkExplicitDecoupleShuffle.py":
        "e74c1b65f8d5cb0cdde6df3dcc44725801ea836d581d68b380359659d3d9e6c4",
    "/data/SSD2/third_party/FCMI/main.py":
        "c4cb24540e0211b7ecc7fe7ccc445788cfa42a71bbb058c16df720d5ed45591d",
    "/data/SSD2/third_party/ALADIN/main.py":
        "68b973ef3cba1a426cc25a69fed5466149ba90a4f883871801dbcb50b2f6a44d",
    "/data/SSD2/third_party/ALADIN/vit_networks.py":
        "ff63666fb13e69710d5f3f2bed5f038d386ddd383eb522ee22688ee6abff7ad1",
    "/data/SSD2/third_party/ALADIN_weights/aladinVIT_bamfg_64.47.pt":
        "310835f8b65349f8900ae7f70c95ee58858537939bdd2c7b86ce36a666b2a91d",
    # HF weights: the snapshot of the rule's revision (rule gap: the rule names the revision, not the cache path)
    str(_HF / "models--facebook--dinov2-large/snapshots/47b73eefe95e8d44ec3623f8890bd894b6ea2d6c/model.safetensors"):
        "399fba97a95f22c36834418bc69373364a99af3a1153da1c0fb31db567c92e23",
    str(_HF / "models--google--siglip2-so400m-patch14-384/snapshots/e8e487298228002f3d8a82e0cd5c8ea9c567f57f/"
              "model.safetensors"):
        "9f4f4a49f908ef0c979bce8ff5a5c0e88882dc6c5dc4304387cbbd152558e2c2",
    "/root/miniconda3/envs/CoSiR/lib/python3.11/site-packages/clustpy/alternative/nrkmeans.py":
        "1145bd9f79cc12931ecf91ab38a94bfed7d7df70f98f06056b5e7b0ec855e569",
    "20261117_reader_fix_csd/results/rb_halves.npz":
        "ed2d359e6517b4237f8aa4de5e78f787699b366a975ac993eaef0e7c4491c7da",
    f"{_R7R}/groupings/r7_groupings.npz": "e278fa62093a89fec4c6bea37dc224b15af2fa5464416067611702e5143ff3d2",
}

# CLIP B/32: one of two weight files loads (rule: which one is recorded); assert_inputs(clip=True) needs one to match.
_CLIP = _HF / "models--openai--clip-vit-base-patch32/snapshots"
CLIP_WEIGHTS = {
    str(_CLIP / "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268/pytorch_model.bin"):
        "a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f",
    str(_CLIP / "c237dc49a33fc61debc9276459120b7eac67e7ef/model.safetensors"):
        "99d28a652e6ec46629ab7047a0ac82c69b1fe11e0ce672c43af65d3a9a3fc05d",
}

# round-7 modules reused read only (rule section 10, "Reused code", commit e3776ca), under R7D
R7_CODE = {
    "r7_bundle.py": "051e9922a53c9630c24e782f64044e2409b535e6d970d7a0ecfbcd51962792ab",
    "r7_common.py": "82af92cd71bcb44279d788f2d49db236277449d6d564aa0ba3218ade2575ad12",
    "r7_fusion.py": "f8ac876f1dbf514ddd6985be06de30cddbd613efbb4bb52ac3b3e3d6f654f506",
    "r7_gate.py": "52eb81648d514c5a76c62f94ae61313632da1d59019e3505a61c12c6ab91aa33",
    "r7_groupings.py": "888fbe468c225bd842f6302a6114ce8cad68cfc44b266375994b69fe1faa1521",
    "r7_heads.py": "c221741afb6a37dd63b14b942752992b5b0aba6a761de3b74c7c47e99f299b85",
    "r7_reader.py": "19726c5f92c9fa15186f13476d3da815ca877c2be3a60b6316c61d91f1f9ce00",
    "r7_nrkmeans.py": "6b8b0aa7e1d60b14389abe488616d98323215ba3a853659f6cf4d2afa1074e36",
    "run_r7.py": "639828e74d8f754a36b4ddba017859e42c84bd8392498ea68f20a6ae9d588e66",
    "run_r7_stage0.py": "b3bd38ab96e6b08a511ce5d2b6f6778b5327b2f46c1a0d2ee4b1e40175abc6b4",
}


def path_of(key) -> Path:
    """Absolute path of an INPUTS key."""
    return Path(key) if key.startswith("/") else TEST / key


def assert_rule():
    """Every r8 script calls this first: the rule must be the committed one. Returns its SHA-256 (to record)."""
    got = sha_file(RULE)
    if got != RULE_SHA256:
        raise SystemExit(f"DECISION_RULE.md differs from the committed version: {got} != {RULE_SHA256}")
    return got


_CHECKED = {}


def _check(table, label, keys=None):
    out = {}
    for key in (table if keys is None else keys):
        if key not in table:
            raise KeyError(f"{key!r} is not in the {label} table")
        if key not in _CHECKED:
            p = path_of(key)
            if not p.is_file():
                raise SystemExit(f"{label} {key}: {p} does not exist")
            got = sha_file(p)
            if got != table[key]:
                raise SystemExit(f"{label} {key}: SHA-256 {got} differs from the rule's {table[key]}")
            _CHECKED[key] = got
        out[key] = _CHECKED[key]
    return out


def assert_inputs(keys=None, fresh=False, r7=True, clip=False):
    """Check rule section 10's inputs (all of INPUTS, or only ``keys``); stop (SystemExit) on any mismatch or missing
    file. A file already verified in this process is not hashed again unless fresh=True (the end-of-stage-0 pass).
    With keys=None also: the ten reused R7 modules (R7_CODE), R7's own section-10 tables (r7_common.assert_inputs,
    unless r7=False) and one matching CLIP weight file (clip=True is implied). Returns {key: sha256} for INPUTS and
    R7_CODE keys, plus the CLIP weight actually matched under ``clip_weight``."""
    if fresh:
        _CHECKED.clear()
        _R7_VERIFIED.clear()
    out = _check(INPUTS, "input", keys)
    if keys is None:
        for name in R7_CODE:
            out[name] = _check_r7(name)
        if r7:
            _r7_common().assert_inputs(fresh=fresh)
        clip = True
    if clip:
        out["clip_weight"] = _clip_weight()
    return out


def _clip_weight():
    """The CLIP B/32 weight file that exists and matches (recorded); SystemExit if none does."""
    for key, want in CLIP_WEIGHTS.items():
        if Path(key).is_file():
            if key not in _CHECKED:
                got = sha_file(key)
                if got != want:
                    raise SystemExit(f"clip {key}: SHA-256 {got} differs from the rule's {want}")
                _CHECKED[key] = got
            return key
    raise SystemExit("no CLIP B/32 weight file of rule section 10 exists")


# ---------------------------------------------------------------- the reused modules

_R7_VERIFIED = {}
_R7_LOADED = {}


def _check_r7(name_py):
    if name_py not in _R7_VERIFIED:
        p = R7D / name_py
        if not p.is_file():
            raise SystemExit(f"R7 module {p} does not exist")
        got = sha_file(p)
        if got != R7_CODE[name_py]:
            raise SystemExit(f"R7 module {name_py}: SHA-256 {got} differs from the rule's {R7_CODE[name_py]}")
        _R7_VERIFIED[name_py] = got
    return _R7_VERIFIED[name_py]


def _front(path):
    p = str(path)
    sys.path[:] = [p] + [e for e in sys.path if e != p]


def load_r7(name):
    """Round-7 module ``name`` (stem, e.g. 'r7_heads'), imported by absolute path from the main checkout's R7 folder,
    once per process, after its SHA-256 matched rule section 10. Read only: R7's folder is never written. Only the
    ten modules of the rule's table; earlier rounds' modules come through load_prior."""
    if name in _R7_LOADED:
        return _R7_LOADED[name]
    if f"{name}.py" not in R7_CODE:
        raise KeyError(f"{name!r} is not an R7 module of rule section 10's table")
    _check_r7(f"{name}.py")
    _front(R7D)                       # the R7 modules import each other (r7_common, ...) by bare name
    path = (R7D / f"{name}.py").resolve()
    mod = sys.modules.get(name)
    if mod is None:
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
    _R7_LOADED[name] = mod
    return mod


def _r7_common():
    return load_r7("r7_common")


def load_prior(name):
    """An earlier-round module (run_step1, run_checks, rb_build, ..., or 'src.eval.aspect_metrics') through R7's
    absolute-path loader and its CODE table (SHA-256 asserted); R7's loader does the sys.path set-up."""
    return _r7_common().load(name)


# ---------------------------------------------------------------- seeds, threads

DEV_SEED = 42
SMOKE_SEEDS = (9001, 9002, 9003)


def assert_threads():
    """OMP and MKL at 8 threads, numpy 2.2.6, scikit-learn 1.6.1 (as R7)."""
    import sklearn
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(v) != "8":
            raise AssertionError(f"{v} is {os.environ.get(v)!r}, not '8'")
    if np.__version__ != "2.2.6":
        raise AssertionError(f"numpy {np.__version__}, not 2.2.6")
    if sklearn.__version__ != "1.6.1":
        raise AssertionError(f"scikit-learn {sklearn.__version__}, not 1.6.1")


# ---------------------------------------------------------------- write-once files

def in_smoke(path) -> bool:
    return Path(path).resolve().is_relative_to(SMOKE.resolve())


def fix_path(path) -> Path:
    """The next free ``<name>_fix<n>.<ext>`` beside ``path`` (a corrected run never overwrites)."""
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
    if not isinstance(rec, dict):
        raise AssertionError("a JSON record must be a dict")
    if "rule_sha256" in rec or "written" in rec:
        raise AssertionError("record already has rule_sha256 or written")
    rec = {**jsonable(rec), "rule_sha256": sha_file(RULE), "written": now_ams()}
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


# ---------------------------------------------------------------- the shared joins

def _ro(a):
    a = np.asarray(a)
    a.setflags(write=False)
    return a


_CACHE = {}


def _once(key, build):
    if key not in _CACHE:
        _CACHE[key] = build()
    return _CACHE[key]


def ctx():
    """Rows and paintings of the round, built once per process (rule M1, M2; C6). Fields:
    scorer_train (int64 global rows, 183,694), g_st (int64 painting index p per scorer-train row), n_paintings
    (36,518), painting_ids (str per p), first_rows (int64: position in scorer_train of p's first row), selection
    (int64 global rows, ascending), halves (int64 per p, 0/1: round 1's painting halves), content (int64 per p: c(p),
    64 groups). Additions: held (int64 global rows), nonheld (int64 global rows outside held, ascending), data
    (load_artelingo()), split (artelingo_splits()), halves_dict (r7_heads.load_halves' dict, for R7's reader/bank
    code), groups (leakage-group code per global row)."""
    return _once("ctx", _build_ctx)


def _build_ctx():
    assert_rule()
    assert_inputs(keys=("20261031_pseudo_partitions/results/partitions.npz",
                        "20261117_reader_fix_csd/results/rb_halves.npz"), r7=False)
    rs = load_prior("run_step1")
    data, sp, st, groups, g_st = rs.base_data()
    st, groups, g_st = (np.asarray(a) for a in (st, groups, g_st))
    n_rows = len(groups)
    assert n_rows == N_ROWS == len(data.paintings), "row count"
    assert st.dtype == np.int64 and st.shape == (N_SCORER_TRAIN,) and (np.diff(st) > 0).all(), "scorer-train rows"
    assert g_st.dtype == np.int64 and g_st.shape == st.shape, "painting codes"
    first = np.unique(g_st, return_index=True)[1].astype(np.int64)
    assert len(first) == N_PAINTINGS and np.array_equal(g_st[first], np.arange(N_PAINTINGS)), "painting count"
    assert np.array_equal(first, rs.first_rows(g_st)), "first rows differ from run_step1.first_rows"
    paintings = np.asarray(data.paintings)
    painting_ids = np.asarray([str(s) for s in paintings[st[first]]])
    assert len(set(painting_ids.tolist())) == N_PAINTINGS and (paintings[st] == painting_ids[g_st]).all(), \
        "painting codes are not one-to-one with painting-id strings"
    # M1: p = position in round 1's paintings_sorted; load_halves asserts it, g_st, the draw and the halves
    h = load_r7("r7_heads").load_halves(st, groups)
    assert np.array_equal(h["paintings_sorted"], np.unique(groups[st])) and np.array_equal(h["g_st"], g_st), \
        "rb_halves.npz paintings_sorted / g_st disagree with the painting index"
    half_row = h["half_of_local_row"].astype(np.int64)
    halves = half_row[first]
    assert np.array_equal(halves[g_st], half_row), "round 1's half is not constant within painting"
    assert np.bincount(halves).tolist() == [18_259, 18_259], "halves sizes"
    # selection and held
    selection = np.asarray(sp.selection)
    held = np.asarray(sp.held)
    assert selection.dtype == np.int64 and (np.diff(selection) > 0).all() and held.dtype == np.int64, "selection/held"
    assert len(selection) == 32_413 and len(held) == 61_744, "selection/held sizes"
    assert not np.intersect1d(selection, st).size and not np.intersect1d(selection, held).size, "split overlap"
    # M2 content grouping
    with np.load(path_of("20261031_pseudo_partitions/results/partitions.npz"), allow_pickle=False) as z:
        assert {"image", "local_groups"} <= set(z.files), "partitions keys"
        assert np.array_equal(z["local_groups"], g_st), "partitions row alignment"
        img = z["image"]
    assert img.dtype == np.int64 and img.shape == st.shape, "content shape/dtype"
    content = img[first]
    assert np.array_equal(content[g_st], img), "content is not constant within painting"
    assert np.array_equal(np.unique(content), np.arange(64)), "content group count"
    sizes = np.bincount(content)
    assert sizes.min() >= 129, "content: smallest group under 129 paintings"
    nonheld = np.setdiff1d(np.arange(n_rows, dtype=np.int64), held)
    return SimpleNamespace(
        scorer_train=_ro(st), g_st=_ro(g_st), n_paintings=N_PAINTINGS, painting_ids=_ro(painting_ids),
        first_rows=_ro(first), selection=_ro(selection), halves=_ro(halves), content=_ro(content),
        held=_ro(held), nonheld=_ro(nonheld), data=data, split=sp, halves_dict=h, groups=_ro(groups))


# ---------------------------------------------------------------- CLIP vectors x (M1)

def unit(x):
    """L2-normalised float32 rows (the earlier rounds' recipe, run_checks.unit)."""
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _unit_features():
    def build():
        f = ctx().data.img_features
        assert f.shape == (N_ROWS, 512) and f.dtype == np.float32, "img_features shape/dtype"
        u = unit(f)
        assert np.isfinite(u).all(), "img_features: non-finite after normalising"
        return _ro(u)
    return _once("unit_features", build)


def x_rows(rows):
    """Unit CLIP B/32 ``img_features`` of global ``rows`` (float32 [n, 512])."""
    rows = np.asarray(rows)
    assert rows.ndim == 1 and rows.dtype.kind in "iu", "x_rows: 1-d integer rows required"
    assert rows.size == 0 or (rows.min() >= 0 and rows.max() < N_ROWS), "x_rows: row out of range"
    return _unit_features()[rows.astype(np.int64, copy=False)]


def x_paintings():
    """x_p: unit CLIP of painting p's first scorer-train row, float32 [36,518, 512], in the painting index order."""
    def build():
        c = ctx()
        return _ro(x_rows(c.scorer_train[c.first_rows]))
    return _once("x_paintings", build)


# ---------------------------------------------------------------- images (M10, C6)

def row_to_image():
    """int32 CSD image index per global row, from style_csd_vitl/row_to_image.npy after R7's C6 checks
    (r7_groupings.read_source: SHA of the file and of the array against meta.json, sample-ID join, image-path join,
    painting/image bijection)."""
    def build():
        c = ctx()
        assert_inputs(keys=tuple(f"/data/SSD2/pre_extract/artelingo/style_csd_vitl/{n}"
                                 for n in ("meta.json", "row_to_image.npy", "embeddings.npy")), r7=False)
        rs = load_prior("run_step1")
        _, r2i, prov = load_r7("r7_groupings").read_source(STYLE, c.data, rs)
        assert prov["C6_checks"] and r2i.dtype == np.int32 and r2i.shape == (N_ROWS,), "row_to_image"
        assert r2i.min() == 0 and r2i.max() == N_CSD_IMAGES - 1, "row_to_image range"
        return _ro(r2i)
    return _once("row_to_image", build)


def painting_image():
    """int32 CSD image index per painting p (of its first scorer-train row); one image per painting and one painting
    per image are asserted (R7 painting_vectors)."""
    def build():
        c, r2i = ctx(), row_to_image()
        img = r2i[c.scorer_train]
        pi = img[c.first_rows]
        assert np.array_equal(pi[c.g_st], img), "a painting maps to several images"
        assert len(np.unique(pi)) == N_PAINTINGS, "an image maps to several paintings"
        return _ro(pi)
    return _once("painting_image", build)


def nonheld_images():
    """int32 [49,121]: the ascending CSD image indices referenced by non-held rows (M10), asserted to number 49,121
    and to equal the 224 cache's image set (paintings.json, compared by image path)."""
    def build():
        c, r2i = ctx(), row_to_image()
        img = np.unique(r2i[c.nonheld]).astype(np.int32)
        assert img.shape == (N_NONHELD_IMAGES,), f"non-held images: {len(img)}, not {N_NONHELD_IMAGES}"
        assert_inputs(keys=("/data/SSD2/pre_extract/artelingo_clip224/paintings.json",
                            "/data/SSD2/pre_extract/artelingo/style_csd_vitl/meta.json"), r7=False)
        cache = json.loads((CACHE224 / "paintings.json").read_text())["paintings"]
        paths = json.loads((STYLE / "meta.json").read_text())["images"]["image_paths"]
        assert len(cache) == N_NONHELD_IMAGES, "224 cache image count"
        assert {paths[i] for i in img.tolist()} == {v["image"] for v in cache.values()}, \
            "non-held images differ from the 224 cache's image set"
        return _ro(img)
    return _once("nonheld_images", build)


def _positions(images):
    """Positions of CSD ``images`` in nonheld_images() (every one must be there)."""
    nh = nonheld_images()
    pos = np.searchsorted(nh, images).astype(np.int64)
    assert (pos < len(nh)).all() and np.array_equal(nh[pos], images), "image outside the non-held set"
    return _ro(pos)


def painting_pos():
    """int64 per painting p: position of its image in nonheld_images() (its row in a space file)."""
    return _once("painting_pos", lambda: _positions(painting_image()))


def selection_image():
    """int32 CSD image per selection row (ctx().selection)."""
    return _once("selection_image", lambda: _ro(row_to_image()[ctx().selection]))


def selection_pos():
    """int64 per selection row: position of its image in nonheld_images()."""
    return _once("selection_pos", lambda: _positions(selection_image()))


def nonheld_first_rows():
    """int64 [49,121]: for each non-held image (nonheld_images() order) its first non-held global row."""
    def build():
        c, r2i = ctx(), row_to_image()
        img_of_row = r2i[c.nonheld]
        nh = nonheld_images()
        u, first = np.unique(img_of_row, return_index=True)         # first occurrence in ascending row order
        assert np.array_equal(u.astype(np.int32), nh), "non-held first rows: image set differs"
        return _ro(c.nonheld[first])
    return _once("nonheld_first_rows", build)


def x_images():
    """float32 [49,121, 512]: x of each non-held image's first non-held row, in nonheld_images() order."""
    return _once("x_images", lambda: _ro(x_rows(nonheld_first_rows())))
