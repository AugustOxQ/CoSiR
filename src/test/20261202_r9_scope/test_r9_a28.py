"""Tests of r9_a28 (ticket 14; rule section 3.3 "Read set", "Labels", "Translation", "Language flags", section 5
items 1 and 2) on a small synthetic ArtELingo-28 CSV and translations table. No feature, no score, no real file.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider \
        src/test/20261202_r9_scope/test_r9_a28.py
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r9_a28 as A  # noqa: E402
from src.data.wikiart_genre import GENRE_NAMES  # noqa: E402

NAMES = {"emotion": ["amusement", "anger", "awe"], "style": ["Baroque", "Realism", "Symbolism"],
         "genre": list(GENRE_NAMES)}
COLS = ["image_id", "genre", "emotion", "caption", "art_style", "painting", "language", "image_name", "split"]
TCOLS = ["csv_row", "painting", "language", "nllb_src", "proxy", "source", "translation"]


def make(rows):
    """rows: (painting, language, emotion, style, genre, caption, translation) -> (csv, translations)."""
    csv = pd.DataFrame([[0.0, g, e, c, s, p, lang, f"{s}/{p}.jpg", "train"]
                        for p, lang, e, s, g, c, _ in rows], columns=COLS)
    trans = pd.DataFrame([[i, p, lang, "xxx_Latn", False, c, t]
                          for i, (p, lang, _, _, _, c, t) in enumerate(rows) if lang != "english"], columns=TCOLS)
    return csv, trans


ROWS = [
    ("held1", "english", "anger", "Baroque", "portrait", "a b c d", None),
    ("held1", "Hausa", "anger", "Baroque", "portrait", "x", "one two three"),
    ("held1", "Hausa", "other", "Baroque", "portrait", "x", "one two"),
    ("new1", "english", "something else", "Realism", "landscape", "a b", None),
    ("new1", "Hausa", "something else", "Realism", "landscape", "x", ""),
    ("new1", "chinese", "awe", "Realism", "landscape", "x", "one"),
    ("old1", "english", "awe", "Symbolism", "portrait", "a", None),
    ("old1", "Hausa", "awe", "Symbolism", "portrait", "x", "one two three four"),
]
GENRE = {"held1": GENRE_NAMES.index("portrait"), "new1": GENRE_NAMES.index("landscape"),
         "old1": GENRE_NAMES.index("portrait")}


def build(rows=ROWS, held=("held1",), train_side=("old1",), ours=None, genre=None):
    csv, trans = make(rows)
    return A.build_read_set(csv, trans, set(held), set(train_side), NAMES, genre or GENRE,
                            ours or {"held1": "Baroque"})


def test_painting_both_held_and_train_side_fails():
    with pytest.raises(AssertionError, match="both held and train-side"):
        build(held=("held1", "old1"))


def test_read_set_counts_and_row_space():
    df, info = build()
    assert list(df.columns) == ["row_id", "painting", "language", "emotion", "style", "genre", "held", "dropped"]
    assert (info["n_paintings"], info["n_held"], info["n_new"]) == (2, 1, 1)
    assert info["n_english_dropped"] == 2 and info["n_train_side"] == 1
    assert df.row_id.tolist() == [1, 2, 4, 5]          # CSV data-row positions, ascending, English rows gone
    assert df.painting.tolist() == ["held1", "held1", "new1", "new1"]
    assert df.held.tolist() == [True, True, False, False]
    assert df.dropped.tolist() == [False, False, True, False]    # the empty output of row 4 is kept and flagged
    assert info["dropped_by_language"] == {"Hausa": 1, "chinese": 0}
    assert info["rows_by_language"] == {"Hausa": 3, "chinese": 1}


def test_labels_mapped_by_name_and_other_is_minus_one():
    df, _ = build()
    row = df.set_index("row_id")
    # one emotion, one style, one genre, by name (not by CSV order)
    assert row.loc[1, "emotion"] == NAMES["emotion"].index("anger") == 1
    assert row.loc[5, "style"] == NAMES["style"].index("Realism") == 1
    assert row.loc[2, "genre"] == NAMES["genre"].index("portrait") == 6
    assert row.loc[5, "emotion"] == 2
    # 'other' and 'something else' are unknown
    assert row.loc[2, "emotion"] == -1 and row.loc[4, "emotion"] == -1
    assert A.map_by_name(["other", "something else", "awe"], NAMES["emotion"], unknown=A.EMOTION_UNKNOWN).tolist() \
        == [-1, -1, 2]


def test_unmapped_names_crash():
    with pytest.raises(AssertionError, match="not in"):
        A.map_by_name(["Cubism"], NAMES["style"])
    bad = [r[:4] + ("nude_painting",) + r[5:] if r[0] == "new1" else r for r in ROWS]
    with pytest.raises(AssertionError, match="genre"):
        build(rows=bad)                      # the CSV genre differs from load_wikiart_genre's


def test_held_style_that_differs_keeps_csv_style_and_is_listed():
    df, info = build(ours={"held1": "Realism"})
    assert (df[df.painting == "held1"]["style"] == NAMES["style"].index("Baroque")).all()
    assert info["style_differs"] == [{"painting": "held1", "csv_style": "Baroque", "our_style": "Realism"}]
    assert build()[1]["style_differs"] == []


def test_translation_table_must_cover_the_read_rows_exactly():
    csv, trans = make(ROWS)
    with pytest.raises(AssertionError):
        A.build_read_set(csv, trans.iloc[1:], {"held1"}, {"old1"}, NAMES, GENRE, {"held1": "Baroque"})
    wrong = trans.copy()
    wrong.loc[wrong.index[0], "language"] = "Igbo"
    with pytest.raises(AssertionError):
        A.build_read_set(csv, wrong, {"held1"}, {"old1"}, NAMES, GENRE, {"held1": "Baroque"})


def test_median_words_after_empty_drops():
    _, info = build()
    assert info["words_median"] == {"Hausa": 2.5, "chinese": 1.0}    # Hausa kept rows: 3 and 2 words
    assert info["english_median_words"] == 3.0                        # English rows on read paintings: 4 and 2 words


def test_whitespace_only_translation_is_dropped():
    extra = ("new1", "Hausa", "awe", "Realism", "landscape", "x", "  ")
    df, info = build(rows=ROWS + [extra])
    assert df.dropped.tolist() == [False, False, True, False, True]    # the empty and the blank output
    assert info["dropped_by_language"] == {"Hausa": 2, "chinese": 0}


def test_language_with_every_row_dropped_crashes():
    rows = [r[:6] + ("  ",) if r[1] == "chinese" else r for r in ROWS]
    with pytest.raises(AssertionError, match="not finite"):
        build(rows=rows)
    with pytest.raises(AssertionError, match="not finite"):
        A.language_flags({"Hausa": {"rows": 10, "code": "hau_Latn", "proxy": False, "empty_outputs": 0,
                                    "outputs_equal_to_source": 0, "hit_max_new_tokens": 0,
                                    "outputs_still_in_source_script": 0}}, {"Hausa": float("nan")}, 10.0)


# ---------------------------------------------------------------- language flags

def cov(rows=1000, code="xxx_Latn", proxy=False, empty=0, equal=0, maxtok=0, still=0):
    return {"rows": rows, "code": code, "proxy": proxy, "empty_outputs": empty, "outputs_equal_to_source": equal,
            "hit_max_new_tokens": maxtok, "outputs_still_in_source_script": still}


def flags(coverage, words=10.0, english=10.0):
    return A.language_flags({"L": coverage}, {"L": words}, english)["languages"]["L"]


def test_flags_quiet_language():
    f = flags(cov())
    assert f["flagged"] is False and set(f["criteria"]) == {"proxy", "failed_outputs", "source_script", "length"}
    for c in f["criteria"].values():
        assert c["fired"] is False and "value" in c and "threshold" in c


def test_flag_proxy():
    assert flags(cov(proxy=True))["criteria"]["proxy"]["fired"] is True


def test_flag_failed_outputs_just_above_threshold():
    # failed = empty + equal-to-source + hit-max-new-tokens, over rows; fires above 5%, not at 5%
    at = flags(cov(empty=20, equal=20, maxtok=10))
    above = flags(cov(empty=20, equal=20, maxtok=11))
    assert at["criteria"]["failed_outputs"]["fired"] is False and at["flagged"] is False
    assert above["criteria"]["failed_outputs"]["fired"] is True and above["flagged"] is True
    assert above["criteria"]["failed_outputs"]["value"] == 0.051
    assert above["criteria"]["failed_outputs"]["threshold"] == 0.05


def test_flag_source_script_only_when_not_latin():
    at = flags(cov(code="kir_Cyrl", still=50))
    above = flags(cov(code="kir_Cyrl", still=51))
    assert at["criteria"]["source_script"]["fired"] is False
    assert above["criteria"]["source_script"]["fired"] is True and above["flagged"] is True
    latin = flags(cov(code="hau_Latn", still=900))      # a Latin-script source never fires this criterion
    assert latin["criteria"]["source_script"]["fired"] is False
    assert flags(cov(code="zho_Hans", still=51))["criteria"]["source_script"]["fired"] is True


def test_flag_length_ratio_both_sides():
    # median output words vs the English median: below 0.5x or above 2x
    for words, fired in ((5.0, False), (4.9, True), (20.0, False), (20.5, True)):
        f = flags(cov(), words=words, english=10.0)
        assert f["criteria"]["length"]["fired"] is fired, words
        assert f["flagged"] is fired
    f = flags(cov(), words=20.5, english=10.0)
    assert f["criteria"]["length"]["value"] == 2.05 and f["criteria"]["length"]["threshold"] == [0.5, 2.0]


def test_flags_list_and_thresholds():
    out = A.language_flags({"A": cov(proxy=True), "B": cov()}, {"A": 10.0, "B": 10.0}, 10.0)
    assert out["flagged"] == ["A"] and set(out["languages"]) == {"A", "B"}
    assert out["english_median_words"] == 10.0


# ---------------------------------------------------------------- value sets

def test_value_sets_are_development_values_eligible_on_kept_rows():
    n = 40
    paint = np.repeat([f"p{i}" for i in range(n)], 2)
    emo = np.tile([0, 1], n)
    df = pd.DataFrame({"painting": paint, "emotion": emo, "style": 0, "genre": 3, "dropped": False})
    df.loc[df.index[:20], "emotion"] = 0                   # 10 paintings only with emotion 0 plus others
    dev = {"emotion": [0, 1], "style": [0], "genre": [3]}
    v = A.value_sets(df, dev, min_paintings=30)
    assert v == {"emotion": [0, 1], "style": [0], "genre": [3]}
    # a dropped row leaves the pool; an unknown label does too
    df2 = df.copy()
    df2.loc[df2.emotion == 1, "dropped"] = True
    assert A.value_sets(df2, dev, min_paintings=30)["emotion"] == [0]
    df3 = df.copy()
    df3.loc[df3.index[:], "genre"] = -1
    assert A.value_sets(df3, {**dev, "genre": []}, min_paintings=30)["genre"] == []
    # an eligible value that is not a development value crashes
    with pytest.raises(AssertionError, match="development"):
        A.value_sets(df, {"emotion": [0], "style": [0], "genre": [3]}, min_paintings=30)


def test_value_sets_pool_needs_all_three_labels_known():
    n = 40
    paint = [f"p{i}" for i in range(n)] + [f"q{i}" for i in range(n)]
    # p*: style 0, emotion known; q*: style 1 only on rows whose emotion is unknown (-1)
    df = pd.DataFrame({"painting": paint, "emotion": [0] * n + [-1] * n, "style": [0] * n + [1] * n,
                       "genre": 3, "dropped": False})
    dev = {"emotion": [0], "style": [0, 1], "genre": [3]}
    assert A.value_sets(df, dev, min_paintings=30) == {"emotion": [0], "style": [0], "genre": [3]}


def test_value_sets_record_names():
    rec = A.value_sets_record({"emotion": [1], "style": [2], "genre": [6]}, NAMES)
    assert rec == {"emotion": [{"code": 1, "name": "anger"}], "style": [{"code": 2, "name": "Symbolism"}],
                   "genre": [{"code": 6, "name": "portrait"}]}


def test_module_reads_no_feature_and_no_score():
    src = (HERE / "r9_a28.py").read_text()
    imports = re.findall(r"^\s*(?:import|from)\s+(\S+)", src, flags=re.M)
    assert not [m for m in imports if re.search(r"feature|r9_data|r9_common|r9_score|r9_heads|torch|artelingo$|pre_extract", m)], imports
    assert not re.search(r"img_features|txt_features|np\.load|FeatureManager|\.npy|\.npz|pre_extract", src)


# ---------------------------------------------------------------- fix wave fx-tests (B1 MEDIUM-1, LOW-3)

def test_translation_painting_must_equal_the_csv_painting():            # r9_a28.py :79
    csv, trans = make(ROWS)
    wrong = trans.copy()
    wrong.loc[wrong.index[0], "painting"] = "new1"                      # row and language still agree with the CSV
    with pytest.raises(AssertionError, match="translation painting differs"):
        A.build_read_set(csv, wrong, {"held1"}, {"old1"}, NAMES, GENRE, {"held1": "Baroque"})


@pytest.mark.parametrize("aspect, extra", [
    ("style", ("new1", "Hausa", "awe", "Symbolism", "landscape", "x", "one")),
    ("genre", ("new1", "Hausa", "awe", "Realism", "portrait", "x", "one"))])
def test_a_painting_with_two_styles_or_genres_crashes(aspect, extra):   # r9_a28.py :90
    with pytest.raises(AssertionError, match=f"more than one {aspect}"):
        build(rows=ROWS + [extra])


def _pool(n_single, n_rows_only):
    """Emotion 0 on ``n_single`` paintings (one row each); emotion 1 on 29 paintings, ``n_rows_only`` extra rows on one."""
    paint = [f"a{i}" for i in range(n_single)] + [f"b{i}" for i in range(29)] + ["b0"] * n_rows_only
    emo = [0] * n_single + [1] * (29 + n_rows_only)
    return pd.DataFrame({"painting": paint, "emotion": emo, "style": 0, "genre": 3, "dropped": False})


def test_value_sets_default_cut_is_30_paintings_not_29_and_not_rows():   # r9_a28.py :24 and :167
    assert A.MIN_PAINTINGS == 30                                         # rule section 5 item 2
    dev = {"emotion": [0, 1], "style": [0], "genre": [3]}
    # emotion 0: exactly 30 paintings (in); emotion 1: 29 paintings but 30 rows (out: the groups are paintings)
    v = A.value_sets(_pool(30, 1), dev)
    assert v == {"emotion": [0], "style": [0], "genre": [3]}
    # one more painting for emotion 1 lets it in; one fewer painting takes emotion 0 out
    pool = _pool(30, 1)
    pool.loc[pool.index[-1], "painting"] = "b_new"
    assert A.value_sets(pool, dev)["emotion"] == [0, 1]
    assert A.value_sets(_pool(29, 1), dev)["emotion"] == []


def test_expected_counts_and_value_sets_are_the_rules():                # r9_a28.py :33-36
    assert A.EXPECTED == {"paintings": 746, "held": 248, "new": 498, "train_side": 912, "rows": 74_569,
                          "languages": 27, "english_dropped": 2_000, "style_differs": 6}
    assert A.EXPECTED_V_SIZES == {"emotion": 8, "style": 7, "genre": 8}
    assert A.EXPECTED_V == {"emotion": [0, 1, 2, 3, 4, 5, 6, 7], "style": [9, 12, 17, 20, 21, 23, 24],
                            "genre": [0, 1, 2, 4, 6, 7, 8, 9]}
    rule = (HERE / "DECISION_RULE.md").read_text()                       # the numbers are the rule's, not only ours
    for text in ("746 paintings", "912 paintings", "74,569 non-English rows in 27 languages", "2,000 English rows",
                 "emotion 8 (all)", "style 7 (Expressionism 9,\n   Impressionism 12, Northern_Renaissance 17, "
                 "Post_Impressionism 20, Realism 21, Romanticism 23, Symbolism 24)", "codes 0, 1, 2, 4, 6, 7, 8, 9"):
        assert text in rule, text


def test_importing_r9_a28_first_loads_src_from_main_and_r9_common_still_imports():   # LOW-5
    # r9_backbone (the import guard) comes before the first src import in r9_a28.py; so in a worktree the module can be
    # imported before r9_common, as a whole-folder pytest collection does.
    src = (HERE / "r9_a28.py").read_text()
    assert src.index("import r9_backbone") < src.index("from src.")
    root = HERE.parents[2]                                              # the checkout that holds this round folder
    code = "import r9_a28, r9_common; print('ok')"
    out = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True,
                         env={"PATH": "/usr/bin", "PYTHONPATH": str(HERE), "PYTHONDONTWRITEBYTECODE": "1",
                              "CUDA_VISIBLE_DEVICES": "",
                              "HOME": os.environ.get("HOME", "/root")})   # git reads safe.directory from HOME
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr[-600:]
