"""ArtELingo-28 read set, labels, language flags and value sets (ticket 14; DECISION_RULE.md section 3.3 "Read set",
"Labels", "Translation", "Language flags", section 5 items 1 and 2).

Pure functions on tables the caller passes in: the ArtELingo-28 CSV, the translations table, `coverage.json` and the
names of our label codes. The module reads labels and text only: it opens no feature file and scores nothing (a test
checks the imports). The runner (run_r9_a28_prepare.py) gives it CLIP's split and the development value sets.

Row space: `row_id` is the CSV data-row position (= `csv_row` of the translations table). The read set has one row per
non-English row on a read painting, before the empty-output drops; evaluated rows are those with `dropped` False.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.eval.aspect_episodes import eligible_values

ASPECTS = ("emotion", "style", "genre")
EMOTION_UNKNOWN = ("something else", "other")           # map to -1 (rule section 3.3 "Labels")
CSV_COLUMNS = ("image_id", "genre", "emotion", "caption", "art_style", "painting", "language", "image_name", "split")
TRANSLATION_COLUMNS = ("csv_row", "painting", "language", "nllb_src", "proxy", "source", "translation")
READ_COLUMNS = ["row_id", "painting", "language", "emotion", "style", "genre", "held", "dropped"]
ENGLISH = "english"
MIN_PAINTINGS = 30

# Language flag thresholds (rule section 3.3 "Language flags"); every comparison is strict.
FAILED_SHARE = 0.05
SCRIPT_SHARE = 0.05
LENGTH_BAND = (0.5, 2.0)
LATIN = "Latn"

# What the rule states (section 3.3, section 5 item 2); the runner asserts them.
EXPECTED = {"paintings": 746, "held": 248, "new": 498, "train_side": 912, "rows": 74_569, "languages": 27,
            "english_dropped": 2_000, "style_differs": 6}
EXPECTED_V_SIZES = {"emotion": 8, "style": 7, "genre": 8}
EXPECTED_V = {"emotion": list(range(8)), "style": [9, 12, 17, 20, 21, 23, 24], "genre": [0, 1, 2, 4, 6, 7, 8, 9]}


def map_by_name(values, names, unknown=()) -> np.ndarray:
    """Codes by name (the index of the name in ``names``); -1 for the names in ``unknown``; any other name crashes."""
    lookup = {name: i for i, name in enumerate(names)}
    out = []
    for v in values:
        if v in unknown:
            out.append(-1)
        else:
            assert v in lookup, f"label {v!r} is not in our names {list(names)[:5]}... and is not one of {unknown}"
            out.append(lookup[v])
    return np.asarray(out, dtype=np.int64)


def build_read_set(csv, trans, held, train_side, names, genre_codes, our_style):
    """The read set of rule section 3.3.

    csv: the ArtELingo-28 CSV (columns CSV_COLUMNS, row position = row_id); trans: translations.parquet.
    held / train_side: sets of painting names (ArtELingo's held paintings; all its other paintings, i.e. train and
    val). names: {aspect: [name by code]}. genre_codes: {painting: code} from load_wikiart_genre.
    our_style: {painting: our style name} for the held paintings.
    Returns (read_set DataFrame, info dict)."""
    assert list(csv.columns) == list(CSV_COLUMNS), f"CSV columns {list(csv.columns)}"
    assert list(trans.columns) == list(TRANSLATION_COLUMNS), f"translations columns {list(trans.columns)}"
    assert (csv.index == np.arange(len(csv))).all(), "CSV row positions are not 0..n-1"
    held, train_side = set(held), set(train_side)
    both = held & train_side
    assert not both, f"{len(both)} paintings are both held and train-side: {sorted(both)[:3]}"
    csv_paintings = set(csv.painting)
    read = sorted(csv_paintings - train_side)
    read_set = set(read)
    held_read = read_set & held
    on_read = csv.painting.isin(read_set).to_numpy()
    english = (csv.language == ENGLISH).to_numpy()
    rows = csv[on_read & ~english]
    row_id = rows.index.to_numpy(dtype=np.int64)

    # translations: exactly one entry per read row, same painting and language (file-boundary checks)
    t = trans[trans.csv_row.isin(row_id)].sort_values("csv_row")
    assert t.csv_row.is_unique and len(t) == len(rows) and np.array_equal(t.csv_row.to_numpy(), row_id), \
        f"translations cover {len(t)} of the {len(rows)} read rows"
    assert (t.painting.to_numpy() == rows.painting.to_numpy()).all(), "translation painting differs from the CSV"
    assert (t.language.to_numpy() == rows.language.to_numpy()).all(), "translation language differs from the CSV"
    text = t.translation.to_numpy(dtype=object)
    assert all(isinstance(x, str) for x in text), "a translation is not a string"
    dropped = np.asarray([x.strip() == "" for x in text])

    # labels, by name
    emotion = map_by_name(rows.emotion.to_numpy(), names["emotion"], unknown=EMOTION_UNKNOWN)
    style = map_by_name(rows.art_style.to_numpy(), names["style"])
    for aspect, col in (("style", "art_style"), ("genre", "genre")):
        per = csv[on_read].groupby("painting")[col].nunique()
        assert (per == 1).all(), f"a painting has more than one {aspect}"
    genre = np.asarray([genre_codes[p] for p in rows.painting], dtype=np.int64)
    bad = [p for p, g, csv_g in zip(rows.painting, genre, rows.genre)
           if g < 0 or names["genre"][g] != csv_g]
    assert not bad, f"genre differs between load_wikiart_genre and the CSV on {len(set(bad))} paintings, e.g. {bad[:3]}"

    # held paintings whose style differs from ours keep the CSV style (listed)
    csv_style = csv[on_read].groupby("painting").art_style.first()
    differs = [{"painting": p, "csv_style": csv_style[p], "our_style": our_style[p]}
               for p in sorted(held_read) if csv_style[p] != our_style[p]]

    df = pd.DataFrame({"row_id": row_id, "painting": rows.painting.to_numpy(), "language": rows.language.to_numpy(),
                       "emotion": emotion, "style": style, "genre": genre,
                       "held": rows.painting.isin(held_read).to_numpy(), "dropped": dropped})
    assert list(df.columns) == READ_COLUMNS
    lang = df.language.to_numpy()
    words = np.asarray([len(x.split()) for x in text])
    eng = csv[on_read & english]
    info = {
        "n_paintings": len(read), "n_held": len(held_read), "n_new": len(read_set - held),
        "n_train_side": len(csv_paintings & train_side),
        "n_rows": len(df), "n_languages": len(set(lang)), "n_english_dropped": len(eng),
        "rows_by_language": {k: int((lang == k).sum()) for k in sorted(set(lang))},
        "dropped_by_language": {k: int(dropped[lang == k].sum()) for k in sorted(set(lang))},
        "words_median": {k: float(np.median(words[(lang == k) & ~dropped])) for k in sorted(set(lang))},
        "english_median_words": float(np.median([len(x.split()) for x in eng.caption])),
        "style_differs": differs,
    }
    assert info["n_held"] + info["n_new"] == info["n_paintings"], "held and new do not add up to the read paintings"
    nan = sorted(k for k, v in info["words_median"].items() if not np.isfinite(v))
    assert not nan, f"no kept row left for {nan}: the median word count is not finite"
    return df, info


def _criterion(value, threshold, fired):
    return {"value": value, "threshold": threshold, "fired": bool(fired)}


def language_flags(coverage, words_median, english_median) -> dict:
    """The four criteria of rule section 3.3 per language, each with its value and threshold.

    coverage: coverage.json ({language: {rows, code, proxy, empty_outputs, outputs_equal_to_source,
    hit_max_new_tokens, outputs_still_in_source_script, ...}}); words_median: median output words on kept read rows
    per language; english_median: the same for the English CSV rows on the read paintings. Counts are compared as
    integers: a share fires only strictly above its threshold."""
    assert english_median > 0, "English median word count is not positive"
    out = {}
    for lang in sorted(coverage):
        c = coverage[lang]
        rows = c["rows"]
        failed = c["empty_outputs"] + c["outputs_equal_to_source"] + c["hit_max_new_tokens"]
        script = c["code"].rsplit("_", 1)[1]
        assert np.isfinite(words_median[lang]), f"{lang}: the median word count is not finite"
        ratio = words_median[lang] / english_median
        crit = {
            "proxy": _criterion(bool(c["proxy"]), True, c["proxy"]),
            "failed_outputs": _criterion(failed / rows, FAILED_SHARE, failed * 20 > rows),
            "source_script": _criterion(c["outputs_still_in_source_script"] / rows, SCRIPT_SHARE,
                                        script != LATIN and c["outputs_still_in_source_script"] * 20 > rows),
            "length": _criterion(ratio, list(LENGTH_BAND), ratio < LENGTH_BAND[0] or ratio > LENGTH_BAND[1]),
        }
        out[lang] = {"code": c["code"], "script": script, "rows": rows, "criteria": crit,
                     "flagged": any(x["fired"] for x in crit.values())}
    return {"thresholds": {"failed_outputs": FAILED_SHARE, "source_script": SCRIPT_SHARE,
                           "length_ratio": list(LENGTH_BAND)},
            "english_median_words": english_median,
            "flagged": [k for k, v in out.items() if v["flagged"]], "languages": out}


def value_sets(read_df, dev_sets, min_paintings=MIN_PAINTINGS) -> dict:
    """V (rule section 5 item 2): the development set intersected with `eligible_values` on the pooled kept rows
    (all three labels known; groups = paintings). Every eligible value must be a development value."""
    kept = read_df[~read_df.dropped.to_numpy()]
    known = np.ones(len(kept), dtype=bool)
    for a in ASPECTS:
        known &= kept[a].to_numpy() >= 0
    kept = kept[known]
    groups = np.unique(kept.painting.to_numpy(), return_inverse=True)[1]
    pool = np.arange(len(kept), dtype=np.int64)
    out = {}
    for a in ASPECTS:
        eligible = eligible_values(kept[a].to_numpy(dtype=np.int64), groups, pool, min_paintings)
        outside = sorted(set(eligible) - set(int(v) for v in dev_sets[a]))
        assert not outside, f"{a}: eligible values {outside} are not development values"
        out[a] = sorted(set(eligible) & set(int(v) for v in dev_sets[a]))
    return out


def value_sets_record(v, names) -> dict:
    """{aspect: [{"code", "name"}]}, the format of r9_common.value_sets_record."""
    return {a: [{"code": int(c), "name": str(names[a][c])} for c in v[a]] for a in ASPECTS}
