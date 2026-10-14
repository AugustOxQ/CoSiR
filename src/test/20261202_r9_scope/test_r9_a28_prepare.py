"""run_r9_a28_prepare.main() on stubs (fix wave fx-tests; B1 MEDIUM-1): the runner's own checks (train side = train and
val, the rule's counts, V equal to the rule's V) fire when they should, and the whole run goes through when they hold.

The data, the split, the CSV, the translations and the labels are replaced by small stand-ins; r9_a28's checks and the
runner's checks run for real. No feature, no real file.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider \
        src/test/20261202_r9_scope/test_r9_a28_prepare.py
"""
import builtins
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r9_a28 as A  # noqa: E402
import run_r9_a28_prepare as P  # noqa: E402

LANGS = [f"lang{i:02d}" for i in range(27)]
PAINTINGS = ["h1", "h2", "t1", "t2", "v1", "v2"]          # held 0-1, train 2-3, val 4-5
V = {"emotion": list(range(8)), "style": [9, 12, 17, 20, 21, 23, 24], "genre": [0, 1, 2, 4, 6, 7, 8, 9]}


class World:
    pass


@pytest.fixture
def world(tmp_path, monkeypatch):
    w = World()
    w.seen = {}
    data = SimpleNamespace(paintings=np.asarray(PAINTINGS), art_styles=np.arange(6))
    split = SimpleNamespace(held=np.array([0, 1]), train=np.array([2, 3]), val=np.array([4, 5]),
                            groups=np.arange(6), selection=np.array([2, 3]))
    csv = pd.DataFrame({"painting": PAINTINGS})
    trans = pd.DataFrame({"language": LANGS})
    big = {id(csv): 173_745, id(trans): 167_230}               # the runner asserts the real files' row counts
    monkeypatch.setattr(P, "len", lambda x: big.get(id(x), builtins.len(x)), raising=False)
    monkeypatch.setattr(P.pd, "read_csv", lambda path: csv)
    real_read_parquet = pd.read_parquet
    monkeypatch.setattr(P.pd, "read_parquet", lambda path, *a, **k: trans if "translations" in str(path)
                        else real_read_parquet(path, *a, **k))
    folder = tmp_path / "translations"
    folder.mkdir()
    (folder / "coverage.json").write_text(json.dumps({l: {
        "rows": 1, "code": "xxx_Latn", "proxy": False, "empty_outputs": 0, "outputs_equal_to_source": 0,
        "hit_max_new_tokens": 0, "outputs_still_in_source_script": 0} for l in LANGS}))
    monkeypatch.setattr(P, "TRANSLATIONS", folder)
    monkeypatch.setattr(P, "OUT", tmp_path / "out")
    monkeypatch.setattr(P.K, "load_data", lambda backbone: data)
    monkeypatch.setattr(P.K, "load_split", lambda d: split)
    monkeypatch.setattr(P.K, "value_names", lambda d: {"emotion": [f"e{i}" for i in range(8)],
                                                       "style": [f"s{i}" for i in range(25)],
                                                       "genre": [f"g{i}" for i in range(10)]})
    monkeypatch.setattr(P.K, "development_value_sets", lambda labels, groups, selection: V)
    monkeypatch.setattr(P, "artelingo_aspect_labels", lambda d: {})
    monkeypatch.setattr(P, "load_wikiart_genre", lambda names: np.zeros(len(names), dtype=int))
    # what build_read_set returns: 248 held paintings, the rule's counts as far as the stand-in has them
    df = pd.DataFrame({"row_id": np.arange(248), "painting": [f"p{i:03d}" for i in range(248)],
                       "language": LANGS[0], "emotion": 0, "style": 0, "genre": 0, "held": True, "dropped": False})
    w.counts = {"paintings": 248, "held": 248, "new": 0, "train_side": 4, "rows": 248, "languages": 27,
                "english_dropped": 0, "style_differs": 0}
    monkeypatch.setattr(A, "EXPECTED", dict(w.counts))
    monkeypatch.setattr(A, "EXPECTED_V", V)
    w.v = V
    w.info = {"n_paintings": 248, "n_held": 248, "n_new": 0, "n_train_side": 4, "n_rows": 248, "n_languages": 27,
              "n_english_dropped": 0, "style_differs": [],
              "rows_by_language": {l: 1 for l in LANGS}, "dropped_by_language": {l: 0 for l in LANGS},
              "words_median": {l: 10.0 for l in LANGS}, "english_median_words": 10.0}

    def build_read_set(csv_, trans_, held, train_side, names, genre_codes, pa_style):
        w.seen.update(held=held, train_side=train_side)
        return df, w.info

    monkeypatch.setattr(A, "build_read_set", build_read_set)
    monkeypatch.setattr(A, "value_sets", lambda df_, dev: w.v)
    return w


def test_main_runs_and_hands_the_train_and_val_paintings_to_the_read_set(world, tmp_path):   # prepare :41
    P.main()
    assert world.seen["held"] == {"h1", "h2"}
    assert world.seen["train_side"] == {"t1", "t2", "v1", "v2"}           # val belongs to the train side
    out = tmp_path / "out"
    assert json.loads((out / "a28_counts.json").read_text())["paintings"] == 248
    assert json.loads((out / "value_sets_a28.json").read_text())["style"][0] == {"code": 9, "name": "s9"}


def test_main_stops_when_a_count_differs_from_the_rules(world):          # prepare :63
    world.info["n_new"] = 1
    with pytest.raises(AssertionError, match="differ from the rule's"):
        P.main()


def test_main_stops_when_V_has_the_right_sizes_but_other_values(world):  # prepare :73 (:72 does not see it)
    world.v = {**V, "style": [9, 12, 17, 20, 21, 23, 25]}
    with pytest.raises(AssertionError):
        P.main()
    world.v = {**V, "style": V["style"][:-1]}                             # a wrong size is caught as well
    with pytest.raises(AssertionError):
        P.main()
