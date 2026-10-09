"""Tests of r6_common (ticket 01): paths, earlier modules, input SHA-256s, the held split, the value sets.

Synthetic data uses the real shapes (308,723 rows). Real-data tests read selection rows only (and the split's index
arrays); no held feature or label is read. Mutation tests load a copy of r6_common from tmp_path with one
`# guard:<name>` statement replaced by `pass`, and show that the guard's scenario then goes through.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider \
        src/test/20261125_artelingo_held_test/test_r6_common.py
"""
import ast
import importlib.util
import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r6_common as R  # noqa: E402

from src.data.splits import grouped_split, grouped_subsplit  # noqa: E402

MAIN_EXPECTED = Path("/project/CoSiR")
ENV = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8",
       "CUDA_VISIBLE_DEVICES": ""}
HERE_LINE = "HERE = Path(__file__).resolve().parent\n"
_COUNT = itertools.count()


# ---------------------------------------------------------------- mutation helpers

def mutant_source(guard=None, here=None, replace=()) -> str:
    """r6_common's source with HERE pinned, the `# guard:<guard>` statements replaced by `pass`, and ``replace``
    (old, new) substitutions applied (each old string must occur exactly once)."""
    src = (HERE / "r6_common.py").read_text()
    assert src.count(HERE_LINE) == 1
    lines = src.splitlines(keepends=True)
    if guard is not None:
        hits = [n for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.Expr) and f"# guard:{guard}" in lines[n.end_lineno - 1]]
        assert hits, f"no statement carries # guard:{guard}"
        for n in hits:
            indent = lines[n.lineno - 1][:len(lines[n.lineno - 1]) - len(lines[n.lineno - 1].lstrip())]
            lines[n.lineno - 1] = f"{indent}pass\n"
            for i in range(n.lineno, n.end_lineno):
                lines[i] = "\n"
    src = "".join(lines).replace(HERE_LINE, f"HERE = Path({str(here or R.HERE)!r})\n")
    for old, new in replace:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    return src


def load_copy(tmp_path, **kw):
    """Import a (possibly mutated) copy of r6_common from tmp_path under a fresh name."""
    path = tmp_path / f"r6_common_copy{next(_COUNT)}.py"
    path.write_text(mutant_source(**kw))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(path.stem, None)
    return mod


def raised_message(fn, *a, **kw):
    """The message of the exception ``fn`` raises, or None."""
    try:
        fn(*a, **kw)
    except BaseException as e:   # SystemExit included
        return str(e)
    return None


def test_every_marked_guard_has_a_mutation_test():
    marked = set(re.findall(r"# guard:(\w+)", (HERE / "r6_common.py").read_text()))
    tested = set(re.findall(r"guard=\"(\w+)\"", Path(__file__).read_text()))
    tested |= {g for g, _ in SPLIT_SCENARIOS}
    assert marked == tested, (marked - tested, tested - marked)


# ---------------------------------------------------------------- MAIN and the earlier modules

def _worktrees():
    out = subprocess.run(["git", "-C", str(MAIN_EXPECTED), "worktree", "list", "--porcelain"],
                         capture_output=True, text=True, check=True).stdout
    paths = [Path(line.split(" ", 1)[1]) for line in out.splitlines() if line.startswith("worktree ")]
    return [p for p in paths if p.resolve() != MAIN_EXPECTED.resolve() and (p / "src/test").is_dir()]


def test_main_from_main_checkout_and_worktrees(tmp_path):
    assert R.MAIN == MAIN_EXPECTED.resolve()
    assert R.TEST == MAIN_EXPECTED / "src/test"
    assert R.main_checkout(MAIN_EXPECTED / "src/test") == MAIN_EXPECTED
    assert R.main_checkout(R.HERE) == MAIN_EXPECTED
    trees = _worktrees()
    if not trees:
        pytest.skip("no worktree of the CoSiR repository exists to test from")
    for wt in trees[:3]:
        assert R.main_checkout(wt / "src/test") == MAIN_EXPECTED, wt
    with pytest.raises(RuntimeError, match="not inside a git checkout"):
        R.main_checkout(tmp_path)


def test_main_guard_fires_when_src_test_is_missing(tmp_path):
    repo = tmp_path / "repo"
    fake_here = repo / "a" / "b" / "F"                    # a git repository without src/test
    fake_here.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    with pytest.raises(AssertionError, match="does not exist"):
        load_copy(tmp_path, here=fake_here)
    msg = raised_message(load_copy, tmp_path, here=fake_here, guard="main_src_test")
    assert msg is None or "does not exist" not in msg


def test_earlier_modules_come_from_main():
    for name, folder in R.EXPECTED_HOME.items():
        f = Path(sys.modules[name].__file__).resolve()
        assert f.parent == folder.resolve() and folder.resolve().parent == (R.MAIN / "src/test").resolve(), name
    for attr, name in (("R3", "r3_common"), ("RF", "r3_fusion"), ("RS", "r3_stats"), ("C", "common"),
                       ("RC", "rc_core"), ("rb", "rb_build"), ("rbe", "rb_eval"), ("rf", "rb_features"),
                       ("F2", "r2_fusion"), ("run_checks", "run_checks"), ("run_n6", "run_n6"),
                       ("rto", "run_told_oracle"), ("rg", "run_gonogo")):
        assert getattr(R, attr) is sys.modules[name], attr
    assert "run_baselines" not in sys.modules


def test_homes_guard_fires_on_a_fake_folder_in_a_copy(tmp_path):
    fake = ('    "r3_stats": R3D,\n', '    "r3_stats": Path("/nonexistent/fake_round3"),\n')
    with pytest.raises(ImportError, match="r3_stats"):
        load_copy(tmp_path, replace=(fake,))
    with pytest.raises(ImportError, match="r3_stats"):
        R.check_homes({"r3_stats": tmp_path})
    R.check_homes()
    load_copy(tmp_path, replace=(fake,), guard="homes")      # guard deleted: the fake folder goes unnoticed


_SHADOW = """
import importlib.util, sys
sys.path.insert(0, {fake!r})
import r3_stats                       # a module of the same name, from another folder, loaded first
spec = importlib.util.spec_from_file_location("r6_common_shadow", {path!r})
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print("IMPORTED")
"""


def _shadow_run(fake, path):
    code = _SHADOW.format(fake=str(fake), path=str(path))
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=ENV, cwd=str(fake))


def test_homes_guard_catches_a_shadowing_module(tmp_path):
    fake = tmp_path / "fake"
    fake.mkdir()
    (fake / "r3_stats.py").write_text("X = 1\n")
    real = _shadow_run(fake, HERE / "r6_common.py")
    assert real.returncode != 0 and "IMPORTED" not in real.stdout
    assert "ImportError" in real.stderr and "r3_stats" in real.stderr.splitlines()[-1]
    copy = tmp_path / "r6_common_noguard.py"
    copy.write_text(mutant_source(guard="homes"))
    mutated = _shadow_run(fake, copy)
    assert mutated.returncode == 0 and "IMPORTED" in mutated.stdout, mutated.stderr[-2000:]


# ---------------------------------------------------------------- constants

def test_constants():
    assert R.RULE_SHA256 == R.sha256_file(R.HERE / "DECISION_RULE.md")
    assert R.R3_RULE_SHA256 == R.sha256_file(R.R3D / "DECISION_RULE.md") == R.R3.RULE_SHA
    assert R.PAIR_NAMES == ("emotion__style", "emotion__genre", "style__genre")
    assert (R.DEV_SEED, R.HELD_SEEDS, R.SMOKE_SEEDS) == (42, (52, 53, 54), (9001, 9002, 9003))
    assert (R.N_PER_PAIR, R.N_SMOKE, R.AFF_HITS_SEED42, R.N_RANKINGS_SEED42) == (4096, 64, 9406, 49152)
    assert R.N_RANKINGS_SEED42 == 3 * R.N_PER_PAIR * 4      # 12,288 episodes x 4 rankings (2 conditions x 2 directions)
    assert R.RESULTS == R.HERE / "results" and R.SMOKE == R.RESULTS / "smoke"
    assert R.CHECKOUT == R.HERE.parents[2]
    assert re.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d", R.amsterdam_now())
    assert R.sha256_bytes(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_pairs_equal_run_baselines_pairs_parsed_not_imported():
    tree = ast.parse((R.AB / "run_baselines.py").read_text())
    found = [ast.literal_eval(n.value) for n in tree.body
             if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "PAIRS" for t in n.targets)]
    assert found == [R.PAIRS]
    assert R.parse_baselines_pairs() == R.PAIRS
    assert "run_baselines" not in sys.modules


# ---------------------------------------------------------------- inputs

def test_assert_inputs_real():
    got = R.assert_inputs()
    assert list(got) == list(R.INPUT_SHA256) and got == R.INPUT_SHA256
    assert len(got) == 2 + 36 + 13
    assert got[R.RULE_REL] == R.RULE_SHA256 and got[R.R3_RULE_REL] == R.R3_RULE_SHA256
    assert set(R.R3.INPUTS) <= set(got) and all(got[k] == v for k, v in R.R3.INPUTS.items())
    want = {
        "20261116_grouping_step1_style/results/step1_group_style.npz": "b04d96b4",
        "20261116_grouping_step1_style/results/step1_heads_style.npz": "898a3701",
        "20261122_round4_aff_vetoes/results/dev_seed42.json": "fd7b3f48",
        "20261013_stage_d_selection/cache/prepare.npz": "d30f0281",
        "20261014_stage_d_final/cache/held_codes.npz": "2901f875",
        "20261121_round3_affect_gate/results/seed42_arrays.npz": "5ea4b09a",
    }
    assert all(got[k].startswith(v) for k, v in want.items())
    for s in R.SMOKE_SEEDS:
        assert f"20261030_aspect_baselines/results/smoke/baselines_seed{s}.json" in got
    assert R.INPUT_PATHS[R.RULE_REL] == R.HERE / "DECISION_RULE.md"
    assert all(R.INPUT_PATHS[k] == R.TEST / k for k in got if k != R.RULE_REL)


@pytest.mark.parametrize("rel", ["20261117_reader_fix_csd/results/rc_tau.json",
                                 "20261030_aspect_baselines/results/smoke/baselines_seed9002.json",
                                 "20261125_artelingo_held_test/DECISION_RULE.md"])
def test_assert_inputs_fails_on_one_changed_byte(tmp_path, monkeypatch, rel):
    copy = tmp_path / Path(rel).name
    data = bytearray(R.INPUT_PATHS[rel].read_bytes())
    copy.write_bytes(bytes(data))
    monkeypatch.setitem(R.INPUT_PATHS, rel, copy)
    assert R.assert_inputs()[rel] == R.INPUT_SHA256[rel]          # an identical copy passes
    data[len(data) // 2] ^= 0x01
    copy.write_bytes(bytes(data))
    with pytest.raises(SystemExit, match=re.escape(rel)):
        R.assert_inputs()
    mut = load_copy(tmp_path, guard="input_sha")
    mut.INPUT_PATHS[rel] = copy
    assert raised_message(mut.assert_inputs) is None              # guard deleted: the changed byte goes through


# ---------------------------------------------------------------- the split, synthetic (real shape)

@pytest.fixture(scope="module")
def synth():
    rng = np.random.default_rng(0)
    groups = np.unique(rng.integers(0, 90_000, R.N_ROWS), return_inverse=True)[1].astype(np.int64)
    sp = grouped_split(groups, seed=42)
    scorer, sel = grouped_subsplit(groups, sp.train, 0.15, seed=42)
    return dict(groups=groups, train=sp.train, val=sp.val, held=sp.held, scorer_train=scorer, selection=sel)


def _args(s):
    a = {k: v.copy() for k, v in s.items()}
    a["prepared"] = {"groups": a["groups"], "split_train": a["train"], "scorer_train": a["scorer_train"],
                     "selection": a["selection"]}
    a["held_rows"] = a["held"].copy()
    return a


def _consistent(a):
    """Rebuild prepared and held_rows from the (changed) arrays, so only the targeted guard can fire."""
    a["prepared"] = {"groups": a["groups"], "split_train": a["train"], "scorer_train": a["scorer_train"],
                     "selection": a["selection"]}
    a["held_rows"] = a["held"].copy()
    return a


def _prepared_off(key):
    def f(a):
        a["prepared"] = dict(a["prepared"])
        x = a["prepared"][key].copy()
        x[len(x) // 2] += 1
        a["prepared"][key] = x
        return a
    return f


def _wrong_held(a):
    a["held_rows"] = np.sort(np.concatenate([a["held_rows"][1:], a["val"][:1]]))
    return a


def _held_in_train(a):
    a["held"] = np.union1d(a["held"], a["train"][:1])
    return _consistent(a)


def _leak(a):
    a["groups"][a["held"][0]] = a["groups"][a["train"][0]]
    return _consistent(a)


def _held_in_selection(a):
    a["selection"] = np.union1d(a["selection"], a["held"][:1])
    return _consistent(a)


def _held_in_scorer(a):
    a["scorer_train"] = np.union1d(a["scorer_train"], a["held"][:1])
    return _consistent(a)


def _selection_short(a):
    a["selection"] = a["selection"][1:]
    return _consistent(a)


SPLIT_SCENARIOS = [
    ("prepared", (_prepared_off("groups"), "recomputed groups differs")),
    ("prepared", (_prepared_off("split_train"), "recomputed split_train differs")),
    ("prepared", (_prepared_off("scorer_train"), "recomputed scorer_train differs")),
    ("prepared", (_prepared_off("selection"), "recomputed selection differs")),
    ("held_rows", (_wrong_held, "held_codes.npz held_rows")),
    ("held_disjoint", (_held_in_train, "held rows overlap train or val")),
    ("leakage", (_leak, "leakage group spans held")),
    ("held_selection", (_held_in_selection, "held rows overlap selection")),
    ("held_scorer", (_held_in_scorer, "held rows overlap scorer-train")),
    ("subsplit", (_selection_short, "do not partition train")),
]


def test_check_split_passes_on_a_consistent_split(synth):
    R.check_split(**_args(synth))


@pytest.fixture(scope="module")
def mutants(tmp_path_factory):
    d = tmp_path_factory.mktemp("mutants")
    return {g: load_copy(d, guard=g) for g in sorted({g for g, _ in SPLIT_SCENARIOS})}


@pytest.mark.parametrize("guard,case", SPLIT_SCENARIOS, ids=[f"{g}-{i}" for i, (g, _) in enumerate(SPLIT_SCENARIOS)])
def test_check_split_guard_fires_and_matters(synth, mutants, guard, case):
    change, msg = case
    a = change(_args(synth))
    with pytest.raises(AssertionError, match=msg):
        R.check_split(**a)
    got = raised_message(mutants[guard].check_split, **change(_args(synth)))
    assert got is None or msg not in got                          # guard deleted: this message no longer fires


# ---------------------------------------------------------------- value sets, synthetic (real shape)

def _synthetic_labels(n_groups_8, held_lo=200_000, held_n=61_744, drop_genre_group=False, outside_8=0):
    """Labels over 308,723 rows, groups of 4 rows. On held rows: emotion (group % 7) plus value 8 on the first
    ``n_groups_8`` held groups, style group % 23, genre group % 10. ``outside_8`` non-held groups also get value 8."""
    n = R.N_ROWS
    groups = np.arange(n, dtype=np.int64) // 4
    held = np.arange(held_lo, held_lo + held_n, dtype=np.int64)
    lab = {x: np.full(n, -1, dtype=np.int64) for x in R.ASPECTS}
    g = groups[held]
    lab["emotion"][held] = g % 7
    lab["style"][held] = g % 23
    lab["genre"][held] = g % 10
    first = np.unique(g)[:n_groups_8]
    lab["emotion"][held[np.isin(g, first)]] = 8
    if drop_genre_group:
        lab["genre"][held[g == first[0]]] = -1
    if outside_8:
        other = np.arange(outside_8, dtype=np.int64)          # groups 0.. lie below held_lo // 4
        rows = np.flatnonzero(np.isin(groups, other))
        for x in R.ASPECTS:
            lab[x][rows] = 0
        lab["emotion"][rows] = 8
    vs = {"emotion": [0, 1, 2, 3, 4, 5, 6, 8], "style": list(range(23)), "genre": list(range(10))}
    return lab, groups, held, vs


def test_assert_held_eligible_29_fails_30_passes(tmp_path):
    lab, groups, held, vs = _synthetic_labels(30)
    R.assert_held_eligible(lab, groups, held, vs)
    lab, groups, held, vs = _synthetic_labels(29)
    with pytest.raises(AssertionError, match=r"not eligible on held rows: \{'emotion': \[8\]"):
        R.assert_held_eligible(lab, groups, held, vs)
    lab, groups, held, vs = _synthetic_labels(29, outside_8=5)            # paintings outside held do not count
    with pytest.raises(AssertionError, match="not eligible on held rows"):
        R.assert_held_eligible(lab, groups, held, vs)
    lab, groups, held, vs = _synthetic_labels(30, drop_genre_group=True)  # a painting outside the pool does not count
    with pytest.raises(AssertionError, match="not eligible on held rows"):
        R.assert_held_eligible(lab, groups, held, vs)
    mut = load_copy(tmp_path, guard="held_eligible")
    lab, groups, held, vs = _synthetic_labels(29)
    mut.assert_held_eligible(lab, groups, held, vs)                        # guard deleted: 29 paintings go through


def test_development_value_counts_guard(tmp_path):
    lab, groups, rows, _ = _synthetic_labels(30)       # 8 emotions (0-6 and 8), 23 styles, 10 genres
    got = R.development_value_sets(lab, groups, rows)
    assert got == {"emotion": [0, 1, 2, 3, 4, 5, 6, 8], "style": list(range(23)), "genre": list(range(10))}
    lab["emotion"][rows[lab["emotion"][rows] == 8]] = 0                    # 7 emotions
    with pytest.raises(AssertionError, match="development value counts"):
        R.development_value_sets(lab, groups, rows)
    mut = load_copy(tmp_path, guard="value_counts")
    assert len(mut.development_value_sets(lab, groups, rows)["emotion"]) == 7


def test_pair_pools_guard(tmp_path, monkeypatch):
    lab, groups, rows, _ = _synthetic_labels(30)
    lab["style"][rows[:400]] = -1
    bad_pairs = (("emotion", "style", "genre"), ("emotion", "genre", "emotion"), ("style", "genre", "emotion"))
    monkeypatch.setattr(R, "PAIRS", bad_pairs)
    with pytest.raises(AssertionError, match="pools differ"):
        R._values_on(lab, groups, rows)
    mut = load_copy(tmp_path, guard="pair_pools")
    mut.PAIRS = bad_pairs
    assert raised_message(mut._values_on, lab, groups, rows) is None


def test_catch_all_guard(tmp_path, monkeypatch):
    class D:
        emotions = np.array(["awe", "something else", "fear"], dtype=object)
        art_styles = np.array(["Baroque", "Cubism", "Baroque"], dtype=object)
    assert R.value_names(D)["emotion"] == ["awe", "fear"]
    monkeypatch.setattr(R, "EMOTION_CATCH_ALL", "not an emotion")
    with pytest.raises(AssertionError, match="catch-all"):
        R.value_names(D)
    mut = load_copy(tmp_path, guard="catch_all")
    mut.EMOTION_CATCH_ALL = "not an emotion"
    assert "something else" in mut.value_names(D)["emotion"]


# ---------------------------------------------------------------- r6_module_shas and the folder .gitignore

def test_r6_module_shas_real():
    got = R.r6_module_shas()
    key = "src/test/20261125_artelingo_held_test/r6_common.py"
    assert got[key] == R.sha256_file(R.HERE / "r6_common.py")
    assert not any(Path(k).name.startswith("test_") for k in got)
    assert all(R.sha256_file(R.CHECKOUT / k) == v for k, v in got.items())


def test_r6_module_shas_coverage(tmp_path):
    f = tmp_path / "src/test/20261125_artelingo_held_test"
    f.mkdir(parents=True)
    for name in ("r6_a.py", "run_r6_b.py", "test_r6_c.py", "helper.py", "other.json"):
        (f / name).write_text(name)
    rel = "src/test/20261125_artelingo_held_test"
    assert set(R.r6_module_shas(f)) == {f"{rel}/{n}" for n in ("r6_a.py", "run_r6_b.py")}
    (f / "dts_settings.json").write_text("{}")
    (tmp_path / "scripts").mkdir()
    for name in ("run_r6_x.sh", "das6_sync_r6.py", "run_other.sh", "das6_sync.py", "run_r6_y.py"):
        (tmp_path / "scripts" / name).write_text(name)
    got = R.r6_module_shas(f)
    assert set(got) == {f"{rel}/r6_a.py", f"{rel}/run_r6_b.py", f"{rel}/dts_settings.json", "scripts/run_r6_x.sh",
                        "scripts/das6_sync_r6.py"}
    assert got["scripts/run_r6_x.sh"] == R.sha256_bytes(b"run_r6_x.sh")


def _ignored(rel):
    p = f"src/test/20261125_artelingo_held_test/{rel}"
    r = subprocess.run(["git", "-C", str(R.CHECKOUT), "check-ignore", "-q", "--no-index", p], capture_output=True)
    assert r.returncode in (0, 1), r.stderr
    return r.returncode == 0


def test_folder_gitignore():
    for rel in ("results/a.json", "results/smoke/b.json", "cache/c.json", "x.npz", "x.npy", "x.pt", "x.log",
                "__pycache__/m.pyc", "results/held_arrays.npz"):
        assert _ignored(rel), rel
    for rel in ("dts_settings.json", "smoke_record.json", "smoke_record_fix1.json", "held_started.json",
                "r6_common.py", "DECISION_RULE.md", "test_r6_common.py"):
        assert not _ignored(rel), rel


# ---------------------------------------------------------------- real data (selection rows only)

class _NpzSpy:
    """np.load wrapper that records (file name, key) of every array read."""

    def __init__(self, real):
        self.real, self.read = real, []

    def __call__(self, path, *a, **kw):
        z, spy, name = self.real(path, *a, **kw), self, Path(path).name

        class W:
            files = z.files

            def __getitem__(self, key):
                spy.read.append((name, key))
                return z[key]

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                z.close()
        return W()


@pytest.fixture(scope="module")
def real():
    from src.data.artelingo import load_artelingo
    from src.data.artelingo_splits import artelingo_aspect_labels
    data = load_artelingo()
    real_load, spy = np.load, _NpzSpy(np.load)
    np.load = spy
    try:
        split = R.load_split(data)
    finally:
        np.load = real_load
    labels = artelingo_aspect_labels(data)
    return dict(data=data, split=split, labels=labels, npz_reads=spy.read)


def test_load_split_reproduces_stage_d(real):
    sp = real["split"]
    assert real["npz_reads"] == [("prepare.npz", k) for k in R.PREPARE_KEYS] + [("held_codes.npz", "held_rows")]
    with np.load(R.INPUT_PATHS[R.PREPARE_REL]) as z:
        for key, mine in (("groups", sp.groups), ("split_train", sp.train), ("scorer_train", sp.scorer_train),
                          ("selection", sp.selection)):
            assert np.array_equal(mine, z[key]), key
    with np.load(R.INPUT_PATHS[R.HELD_CODES_REL]) as z:
        assert np.array_equal(sp.held, z["held_rows"])
    sizes = {k: len(getattr(sp, k)) for k in ("train", "val", "held", "scorer_train", "selection")}
    assert sizes == {"train": 216_107, "val": 30_872, "held": 61_744, "scorer_train": 183_694, "selection": 32_413}
    keys = ("groups", "train", "val", "held", "scorer_train", "selection")
    assert all(getattr(sp, k).dtype == np.int64 for k in keys)
    assert len(sp.groups) == R.N_ROWS
    assert np.intersect1d(sp.held, sp.selection).size == 0 and np.intersect1d(sp.held, sp.scorer_train).size == 0


def test_development_value_sets_real(real):
    sp, lab = real["split"], real["labels"]
    vs = R.development_value_sets(lab, sp.groups, sp.selection)
    assert {x: len(v) for x, v in vs.items()} == {"emotion": 8, "style": 23, "genre": 10}
    assert all(v == sorted(v) and all(isinstance(c, int) and c >= 0 for c in v) for v in vs.values())


def test_write_value_sets_real(real, tmp_path):
    from src.data.wikiart_genre import GENRE_NAMES
    sp, lab, data = real["split"], real["labels"], real["data"]
    vs = R.development_value_sets(lab, sp.groups, sp.selection)
    path = tmp_path / "value_sets.json"
    rec = R.write_value_sets(path, vs, data)
    on_disk = json.loads(path.read_text())
    assert on_disk == rec and list(on_disk) == ["emotion", "style", "genre"]
    for x in R.ASPECTS:
        assert [e["code"] for e in on_disk[x]] == vs[x]
        assert all(set(e) == {"code", "name"} and isinstance(e["name"], str) and e["name"] for e in on_disk[x])
    assert "something else" not in [e["name"] for e in on_disk["emotion"]]
    assert all(e["name"] == GENRE_NAMES[e["code"]] for e in on_disk["genre"])
    sel = sp.selection                                     # the code-name join, checked on selection rows only
    for x, names in (("emotion", np.asarray(data.emotions)), ("style", np.asarray(data.art_styles))):
        for e in on_disk[x]:
            assert set(names[sel][lab[x][sel] == e["code"]].tolist()) == {e["name"]}, (x, e)
