"""Stage 1, ArtELingo-28 read set (ticket 14; rule section 3.3 and section 5 items 1 and 2).

Reads the ArtELingo-28 CSV, `translations.parquet` and `coverage.json` (labels and text only; no ArtELingo-28 feature,
no score) and CLIP's split (r9_common.load_data / load_split, which read ArtELingo's CLIP features only to recompute the
frozen split). Writes under results/ (not committed; rerun to make them again):
  read_set_a28.parquet, language_flags_a28.json, value_sets_a28.json, a28_style_differs.json, a28_counts.json

    PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= \
    /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/run_r9_a28_prepare.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import r9_a28 as A  # noqa: E402
import r9_common as K  # noqa: E402
from src.data.artelingo_splits import artelingo_aspect_labels  # noqa: E402
from src.data.wikiart_genre import load_wikiart_genre  # noqa: E402

CSV = Path("/data/PDD/artelingo/ArtELingo/semeval-artelingo28/artelingo28_public_noworkerid_split.csv")
TRANSLATIONS = Path("/data/SSD2/pre_extract/artelingo28/nllb200_3.3b_en")
OUT = K.RESULTS


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=1))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = K.load_data("clip_b32")
    split = K.load_split(data)
    paintings = np.asarray(data.paintings)
    held = set(paintings[split.held])
    train_side = set(paintings[np.union1d(split.train, split.val)])      # every ArtELingo painting that is not held
    assert held | train_side == set(paintings) and not held & train_side, "held and train-side do not partition"
    names = K.value_names(data)

    csv = pd.read_csv(CSV)
    trans = pd.read_parquet(TRANSLATIONS / "translations.parquet")
    coverage = json.loads((TRANSLATIONS / "coverage.json").read_text())
    assert len(csv) == 173_745 and len(trans) == 167_230, (len(csv), len(trans))
    assert set(coverage) == set(trans.language), "coverage.json languages differ from translations.parquet"
    for lang, n in trans.language.value_counts().items():
        assert coverage[lang]["rows"] == n, f"{lang}: coverage rows {coverage[lang]['rows']} != {n}"

    csv_paintings = np.asarray(sorted(set(csv.painting)))
    genre = load_wikiart_genre(csv_paintings)
    genre_codes = dict(zip(csv_paintings.tolist(), genre.tolist()))
    pa_style = dict(zip(paintings[split.held].tolist(), np.asarray(data.art_styles)[split.held].tolist()))

    df, info = A.build_read_set(csv, trans, held, train_side, names, genre_codes, pa_style)

    got = {"paintings": info["n_paintings"], "held": info["n_held"], "new": info["n_new"],
           "train_side": info["n_train_side"], "rows": info["n_rows"], "languages": info["n_languages"],
           "english_dropped": info["n_english_dropped"], "style_differs": len(info["style_differs"])}
    assert got == A.EXPECTED, f"counts {got} differ from the rule's {A.EXPECTED}"
    assert df.held.sum() == int(df[df.held].shape[0]) and df[df.held].painting.nunique() == 248
    assert (df.row_id.diff().dropna() > 0).all()
    assert df.drop(columns="painting").notna().all().all()

    # labels of the dev pool, then V
    labels = artelingo_aspect_labels(data)
    dev = K.development_value_sets(labels, split.groups, split.selection)
    v = A.value_sets(df, dev)
    assert {a: len(v[a]) for a in A.ASPECTS} == A.EXPECTED_V_SIZES, {a: len(v[a]) for a in A.ASPECTS}
    assert v == A.EXPECTED_V, v
    vrec = A.value_sets_record(v, names)

    flags = A.language_flags(coverage, info["words_median"], info["english_median_words"])
    assert set(flags["languages"]) == set(info["rows_by_language"]) and len(flags["languages"]) == 27
    for lang, f in flags["languages"].items():
        f["read_rows"] = info["rows_by_language"][lang]
        f["dropped_read_rows"] = info["dropped_by_language"][lang]
        f["median_output_words_read"] = info["words_median"][lang]

    df.to_parquet(OUT / "read_set_a28.parquet", index=False)
    dump(OUT / "language_flags_a28.json", flags)
    dump(OUT / "value_sets_a28.json", vrec)
    dump(OUT / "a28_style_differs.json", info["style_differs"])
    dump(OUT / "a28_counts.json", {**got, "dropped": int(df.dropped.sum()),
                                   "rows_by_language": info["rows_by_language"],
                                   "dropped_by_language": info["dropped_by_language"],
                                   "held_rows": int(df.held.sum()),
                                   "english_median_words": info["english_median_words"]})
    back = pd.read_parquet(OUT / "read_set_a28.parquet")
    assert list(back.columns) == A.READ_COLUMNS and len(back) == A.EXPECTED["rows"]

    print("counts", json.dumps(got))
    print("dropped (empty outputs):", int(df.dropped.sum()), "held rows:", int(df.held.sum()))
    print("V sizes", {a: len(v[a]) for a in A.ASPECTS})
    for a in A.ASPECTS:
        print("V", a, [f"{r['code']}:{r['name']}" for r in vrec[a]])
    print("flagged", flags["flagged"])
    for lang in flags["flagged"]:
        print(" ", lang, {k: (round(c["value"], 4) if not isinstance(c["value"], bool) else c["value"])
                          for k, c in flags["languages"][lang]["criteria"].items() if c["fired"]})
    print("english median words", info["english_median_words"])
    print("style differs", json.dumps(info["style_differs"]))


if __name__ == "__main__":
    main()
