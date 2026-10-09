"""The held read end to end on synthetic data of the real shapes (ticket 08; rule section 5 item 5, section 8 items 1
to 3; contracts section 7): run_r6_held.run_held with the real RowContext, build_bundle_r6, score_seed, assert_distinct,
assert_held_eligible, save_episodes and pass_record, on a fake "held" row set inside synthetic data. No real held row.

Synthetic data (as test_r6_context.py): 308,723 rows of 512-d float32 image and caption features in paintings of four
rows; a held band of 61,744 rows (15,436 whole paintings) and, outside it, scorer-train 183,694, selection 32,413 and
val 30,872 rows; 8 emotions, 23 styles, 10 genres; held seeds 52, 53, 54 at 4,096 episodes per pair (12,288 per seed);
heads with 41 / 64 / 64 / 64 / 17 classes fitted on a 2,000-row draw (the one stand-in), the PM basis fitted on the
synthetic scorer-train draw. The env replaces setup() (the real data, heads and their check against the stored
selection posteriors); its head check is a stand-in marked passed, and refit_check.json in tmp_path records the
synthetic heads' coefficient SHA-256s. The seed-42 records, the smoke record and the ledger (row H5 with this runner's
SHA-256) are written in tmp_path with the real file names. Real files read: the rule's inputs, the A3 checkpoint, the
readers, baselines_seed42.json (the RCA lambdas) and the recorded episode hashes; no held feature, label or episode.

I/O is instrumented (every write-mode open and os.replace, the masked feature copies, the episode builds, the
posterior predictions, the scores), and the test checks the order of rule section 8 items 1 to 3. A few minutes on
CPU:

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider \
        src/test/20261125_artelingo_held_test/test_r6_held_synthetic.py
"""
import builtins
import io
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r6_common as R  # noqa: E402  (first, before anything that imports src)
import r6_bundle as B  # noqa: E402
import r6_context as X  # noqa: E402
import r6_episodes as E  # noqa: E402
import r6_heads as H  # noqa: E402
import r6_score as S  # noqa: E402
import r6_stats as ST  # noqa: E402
import run_r6_held as RH  # noqa: E402
import test_r6_held as TH  # noqa: E402  (records, ledger lines, stand-in picks and sigma)

import numpy as np  # noqa: E402

from src.eval.aspect_metrics import METRICS  # noqa: E402
from src.eval.pair_metric_baselines import fit_pair_scaler, fit_pca_basis  # noqa: E402

N, D = R.N_ROWS, 512
N_SCORER, N_SEL, N_VAL, N_HELD = 183_694, 32_413, 30_872, 61_744
HELD_START = 160_000
N_FIT = 2_000


def _head_labels(rng, img, scorer_train, k):
    W = rng.standard_normal((D, k), dtype=np.float32)
    y = np.argmax(img[scorer_train] @ W + rng.standard_normal((len(scorer_train), k), dtype=np.float32), axis=1)
    lab = np.full(N, -1, dtype=np.int64)
    lab[scorer_train] = y
    return lab


@pytest.fixture(scope="module")
def env():
    """setup()'s namespace on synthetic data of the real shapes."""
    rng = np.random.default_rng(20261008)
    groups = np.arange(N, dtype=np.int64) // 4
    ng = int(groups[-1]) + 1
    style_g, genre_g = rng.integers(0, 23, ng), rng.integers(0, 10, ng)
    genre_g[rng.random(ng) < 0.03] = -1
    emotion = rng.integers(0, 8, N)
    emotion[rng.random(N) < 0.05] = -1
    labels = {"emotion": emotion.astype(np.int64), "style": style_g[groups].astype(np.int64),
              "genre": genre_g[groups].astype(np.int64)}
    held = np.arange(HELD_START, HELD_START + N_HELD, dtype=np.int64)
    rest = np.setdiff1d(np.arange(N, dtype=np.int64), held)
    scorer_train, selection, val = rest[:N_SCORER], rest[N_SCORER:N_SCORER + N_SEL], rest[N_SCORER + N_SEL:]
    assert len(val) == N_VAL
    split = SimpleNamespace(groups=groups, train=np.sort(np.concatenate([scorer_train, selection])), val=val,
                            held=held, scorer_train=scorer_train, selection=selection)
    data = SimpleNamespace(img_features=rng.standard_normal((N, D), dtype=np.float32),
                           txt_features=rng.standard_normal((N, D), dtype=np.float32))
    heads = H.fit_all(data, scorer_train, {h: _head_labels(rng, data.img_features, scorer_train, H.N_CLASSES[h])
                                           for h in H.HEADS}, N_FIT)
    rows = B.fit_rows(scorer_train)
    fi, ft = (np.asarray(x[rows], np.float32) for x in (data.img_features, data.txt_features))
    fi, ft = (x / np.linalg.norm(x, axis=1, keepdims=True) for x in (fi, ft))
    pm = SimpleNamespace(basis=fit_pca_basis(fi, ft), scaler=fit_pair_scaler(fi, ft), fit_rows_sha256="synthetic")
    vs = {"emotion": list(range(8)), "style": list(range(23)), "genre": list(range(10))}
    return SimpleNamespace(inputs={}, data=data, split=split, labels=labels, value_sets=vs, index=None, heads=heads,
                           head_check={"passed": True, "stand_in": "synthetic heads; no stored posteriors"},
                           coef_sha256=H.coef_sha256(heads), pm=pm, readers=B.load_readers())


def instrument(monkeypatch, tmp):
    """Events in call order: ("write", name) for every write-mode open under ``tmp`` (a .partial or the file),
    ("replace", name) for os.replace, ("head_guard",), ("masked", mode), ("episodes", seed, mode), ("predict",
    n rows, held?), ("score", seed, keys)."""
    ev = []
    tmp = str(tmp)
    real_open, real_io_open, real_replace = builtins.open, io.open, os.replace

    def opener(real):
        def f(file, mode="r", *a, **k):
            if isinstance(file, (str, os.PathLike)) and any(c in mode for c in "wax+") and str(file).startswith(tmp):
                ev.append(("write", Path(file).name))
            return real(file, mode, *a, **k)
        return f

    def replace(src, dst, *a, **k):
        if str(dst).startswith(tmp):
            ev.append(("replace", Path(dst).name))
        return real_replace(src, dst, *a, **k)

    monkeypatch.setattr(builtins, "open", opener(real_open))
    monkeypatch.setattr(io, "open", opener(real_io_open))
    monkeypatch.setattr(os, "replace", replace)

    def wrap(obj, name, event):
        real = getattr(obj, name)

        def f(*a, **k):
            out = real(*a, **k)
            event(out, *a, **k)
            return out
        monkeypatch.setattr(obj, name, f)

    def before(obj, name, event):
        real = getattr(obj, name)

        def f(*a, **k):
            event(*a, **k)
            return real(*a, **k)
        monkeypatch.setattr(obj, name, f)

    before(RH, "head_guard", lambda *a, **k: ev.append(("head_guard",)))
    before(X.RowContext, "masked", lambda self, values: ev.append(("masked", self.mode)))
    before(E, "build_seed", lambda labels, groups, rows, index, vs, seed, n: ev.append(
        ("episodes", int(seed), "held" if np.array_equal(rows, np.arange(HELD_START, HELD_START + N_HELD)) else "?")))
    before(H, "predict", lambda heads, data, rows: ev.append(("predict", len(rows), int(np.asarray(rows)[0]))))
    wrap(S, "score_seed", lambda out, bundle, *a, **k: ev.append(("score", int(bundle.seed), tuple(out))))
    return ev


@pytest.fixture(scope="module")
def read(env, tmp_path_factory):
    """One held run of the runner on the synthetic env, everything recorded."""
    tmp = tmp_path_factory.mktemp("read")
    res = TH.write_records(tmp / "results", coef=env.coef_sha256)
    led = TH.write_ledger(tmp / "held_ledger.md", TH.ledger_line())
    mp = pytest.MonkeyPatch()
    try:
        ev = instrument(mp, tmp)
        code = RH.run_held(results=res, ledger=led, here=HERE, folder=tmp / "folder", env=env)
    finally:
        mp.undo()
    return SimpleNamespace(code=code, ev=ev, res=res, tmp=tmp)


def test_the_read_completes_and_writes_every_file_of_contracts_7(read):
    assert read.code == 0
    names = sorted(p.name for p in read.res.iterdir())
    outputs = ["held_started.json", "held_episodes_seed52.npz", "held_episodes_seed53.npz",
               "held_episodes_seed54.npz", "sensitivity_held.json", "held_arrays.npz", "held_pass.json"]
    assert names == sorted(outputs + [RH.REFIT_NAME, RH.PICKS_NAME, RH.REGRESSION_NAME, RH.SENS42_NAME,
                                      RH.DTS_STOP_NAME, RH.DTS_CHOSEN_NAME, "smoke_record.json"])
    assert not any("verdict" in p.name for p in read.tmp.rglob("*"))                # no verdict anywhere
    assert (read.tmp / "folder/held_started.json").read_bytes() == (read.res / "held_started.json").read_bytes()


def test_started_file_and_episode_files(read, env):
    att = json.loads((read.res / "held_started.json").read_text())["attempts"]
    assert len(att) == 1 and att[0]["coef_sha256"] == env.coef_sha256
    hashes = att[0]["episodes_sha256"]
    assert list(hashes) == ["52", "53", "54"]
    held = np.zeros(N, dtype=bool)
    held[env.split.held] = True
    for s in R.HELD_SEEDS:
        eps = E.load_episodes(read.res / f"held_episodes_seed{s}.npz")             # hashes recomputed and asserted
        assert eps.seed == s and eps.sha == hashes[str(s)] and eps.n == 3 * R.N_PER_PAIR
        assert held[eps.pooled.rows()].all()                                          # every member a held row
        with np.load(read.res / f"held_episodes_seed{s}.npz") as z:
            assert set(z.files) == set(E.FILE_KEYS)
    E.assert_distinct({s: hashes[str(s)] for s in R.HELD_SEEDS})


def test_pass_sensitivity_and_arrays_keys(read):
    pr = json.loads((read.res / "held_pass.json").read_text())
    assert set(pr) >= {"rule_sha256", "mode", "seeds", "n_episodes", "n_clusters", "checks", "holm_order",
                       "secondary", "episodes_sha256", "runner_sha256", "module_sha256", "time"}
    assert pr["mode"] == "held" and pr["seeds"] == [52, 53, 54] and pr["n_episodes"] == 36_864
    for c in ST.CHECKS:
        assert set(pr["checks"][c]) == {"quantity", "n", "point", "ci95", "holm_k", "ci_holm", "level_two_sided",
                                        "passes", "own_count_passes", "near_boundary"}
    for c in ST.SECONDARY:
        assert "passes" not in pr["secondary"][c]
    sens = json.loads((read.res / "sensitivity_held.json").read_text())
    assert sens["N"] == 36_864 and set(ST.CHECKS + ST.SECONDARY) <= set(sens)
    for c in ST.CHECKS:
        assert set(sens[c]) == {"quantity", "sigma_a2", "sigma_eps2", "SE", "x", "x95"}
    for c in ST.SECONDARY:
        assert set(sens[c]) == {"quantity", "sigma_a2", "sigma_eps2", "SE", "x2", "x95"}
    with np.load(read.res / "held_arrays.npz") as z:
        assert set(z.files) == ({f"{s}__{m}" for s in RH.READ_SCORERS for m in METRICS}
                                | {"cl", "pair_index", "seed_index"})
        cl = z["cl"]
        assert cl.shape == (36_864,) and len(np.unique(cl)) == pr["n_clusters"] == sens["n_paintings"]
        d = z["aff_fused__r1"] - z["cosine__r1"]                                      # P1 recomputed from the arrays
        assert int(ST.bootstrap_draws(d, cl)[1].sum()) == pr["checks"]["P1"]["n"]
        assert (z["aff_cf__gain"] == 0).all()


def test_only_the_allowed_scores_exist_before_the_verdict(read):
    scores = [e for e in read.ev if e[0] == "score"]
    assert [e[1] for e in scores] == [52, 53, 54]
    for e in scores:
        assert e[2] == RH.READ_KEYS                                                    # no PM, no r1_cf
        assert not set(e[2]) & set(B.PM_NAMES + ("r1_cf",))


def test_order_of_computation(read):
    ev = read.ev
    first_write = next(i for i, e in enumerate(ev) if e[0] == "write")
    assert ev[first_write] == ("write", "held_started.json.partial")
    assert ev[first_write + 1] == ("replace", "held_started.json")
    assert ("head_guard",) in ev[:first_write]                                         # after the head checks
    held_work = [i for i, e in enumerate(ev) if e[0] in ("masked", "episodes", "predict", "score")]
    assert held_work and min(held_work) > first_write                                  # before any held-row array
    assert all(e[1] == "held" for e in ev if e[0] == "masked")                         # held features only
    assert [e[1:] for e in ev if e[0] == "episodes"] == [(s, "held") for s in R.HELD_SEEDS]
    assert all(e[1:] == (N_HELD, HELD_START) for e in ev if e[0] == "predict")         # exactly split.held
    first_score = min(i for i, e in enumerate(ev) if e[0] == "score")
    for s in R.HELD_SEEDS:
        i_eps = ev.index(("episodes", s, "held"))
        i_pred = next(i for i, e in enumerate(ev) if i > i_eps and e[0] == "predict")
        appended = [e for e in ev[i_eps:i_pred] if e[0] == "replace"]
        assert appended[:2] == [("replace", "held_started.json")] * 2, s               # results and copy, at once
        assert ("replace", f"held_episodes_seed{s}.npz") in appended
        assert i_pred < first_score
    assert ev.index(("replace", "sensitivity_held.json")) < first_score
    last_score = max(i for i, e in enumerate(ev) if e[0] == "score")
    assert last_score < ev.index(("replace", "held_arrays.npz")) < ev.index(("replace", "held_pass.json"))
    assert [e for e in ev if e[0] == "replace"][-1] == ("replace", "held_pass.json")
