"""Final review B1: ArtELingo-28 read set, labels, flags, V re-derived with own code (no r9_* import).

Inputs: the ArtELingo-28 CSV, the merged translations and coverage.json, ArtELingo's metadata (load_artelingo, frozen
src), round 6's frozen split files (prepare.npz split_train, held_codes.npz held_rows), round 6's dev value_sets.json.
Compares with the records in the main checkout's results/. Labels and text only: no score, no model statistic.
Writes fr_b1_counts.json beside this file.
"""
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/project/CoSiR-wt/r9-fr-b")
sys.path.insert(0, str(ROOT))
from src.data.artelingo import load_artelingo  # noqa: E402  (frozen, pre-r9)
from src.data.wikiart_genre import GENRE_NAMES, load_wikiart_genre  # noqa: E402

HERE = Path(__file__).resolve().parent
REC = Path("/project/CoSiR/src/test/20261202_r9_scope/results")
CSV = Path("/data/PDD/artelingo/ArtELingo/semeval-artelingo28/artelingo28_public_noworkerid_split.csv")
TR = Path("/data/SSD2/pre_extract/artelingo28/nllb200_3.3b_en")
TEST = Path("/project/CoSiR/src/test")
PREP = TEST / "20261013_stage_d_selection/cache/prepare.npz"
HELD = TEST / "20261014_stage_d_final/cache/held_codes.npz"
DEV = TEST / "20261125_artelingo_held_test/results/value_sets.json"

out = {}
csv = pd.read_csv(CSV)
trans = pd.read_parquet(TR / "translations.parquet")
cov = json.loads((TR / "coverage.json").read_text())
out["csv_rows"], out["trans_rows"] = len(csv), len(trans)
out["csv_english_rows_all"] = int((csv.language == "english").sum())

data = load_artelingo()
pa = np.asarray(data.paintings)
with np.load(PREP) as z:
    split_train = z["split_train"]
with np.load(HELD) as z:
    held_rows = z["held_rows"]
held = set(pa[held_rows])
train_rows = set(pa[split_train])
all_pa = set(pa)
train_side = all_pa - held                              # every non-held ArtELingo painting (user decision)
out["artelingo_paintings"] = len(all_pa)
out["held_paintings_artelingo"] = len(held)
out["held_and_train_overlap"] = len(held & train_rows)

a28 = set(csv.painting)
out["a28_paintings"] = len(a28)
out["a28_train_side"] = len(a28 & train_side)
out["a28_in_split_train"] = len(a28 & train_rows)
out["a28_train_side_minus_split_train"] = len(a28 & (train_side - train_rows))
read = sorted(a28 - train_side)
rs = set(read)
out["read_paintings"] = len(read)
out["read_held"] = len(rs & held)
out["read_new"] = len(rs - held)
out["read_new_in_artelingo"] = len((rs - held) & all_pa)
out["held_artelingo_not_in_a28"] = len(held - a28)

on = csv.painting.isin(rs).to_numpy()
eng = (csv.language == "english").to_numpy()
rows = csv[on & ~eng]
out["read_rows"] = len(rows)
out["read_languages"] = int(rows.language.nunique())
out["english_rows_on_read"] = int((on & eng).sum())
row_id = np.flatnonzero(on & ~eng)

# translation join, own
tr = trans.set_index("csv_row")
assert tr.index.is_unique
missing = sorted(set(row_id) - set(tr.index))
out["translation_missing_read_rows"] = len(missing)
t = tr.loc[row_id]
out["translation_painting_mismatch"] = int((t.painting.to_numpy() != rows.painting.to_numpy()).sum())
out["translation_language_mismatch"] = int((t.language.to_numpy() != rows.language.to_numpy()).sum())
text = t.translation.to_numpy(dtype=object)
dropped = np.array([(not isinstance(x, str)) or x.strip() == "" for x in text])
out["dropped"] = int(dropped.sum())
out["translations_cover_nonenglish_csv"] = sorted(set(trans.csv_row)) == sorted(np.flatnonzero(~eng).tolist())

# labels by name (own name lists: sorted unique of ArtELingo's labels, as round 6 codes them)
emo_names = sorted({e for e in data.emotions.tolist() if e not in ("", None, "something else")})
sty_names = sorted({s for s in data.art_styles.tolist() if s not in ("", None)})
emo_lookup = {n: i for i, n in enumerate(emo_names)}
sty_lookup = {n: i for i, n in enumerate(sty_names)}
out["csv_emotion_values"] = sorted(set(rows.emotion))
out["csv_emotion_unmapped"] = sorted(set(rows.emotion) - set(emo_names) - {"something else", "other"})
emotion = np.array([-1 if e in ("something else", "other") else emo_lookup[e] for e in rows.emotion], dtype=np.int64)
out["csv_style_unmapped"] = sorted(set(rows.art_style) - set(sty_names))
style = np.array([sty_lookup[s] for s in rows.art_style], dtype=np.int64)
gl = {n: i for i, n in enumerate(GENRE_NAMES)}
out["csv_genre_unmapped"] = sorted(set(rows.genre) - set(GENRE_NAMES))
genre_csv = np.array([gl.get(g, -1) for g in rows.genre], dtype=np.int64)
rp = np.asarray(read)
wg = dict(zip(read, load_wikiart_genre(rp).tolist()))
genre_w = np.array([wg[p] for p in rows.painting], dtype=np.int64)
per_p = pd.DataFrame({"p": rows.painting.to_numpy(), "c": genre_csv, "w": genre_w}).groupby("p").first()
out["genre_wikiart_eq_csv_paintings"] = int((per_p.c == per_p.w).sum())
out["genre_wikiart_unknown_paintings"] = int((per_p.w < 0).sum())

# held paintings whose CSV style differs from ours
our_style = {}
for p, s in zip(pa[held_rows].tolist(), np.asarray(data.art_styles)[held_rows].tolist()):
    our_style.setdefault(p, set()).add(s)
out["held_with_multiple_our_styles"] = sum(len(v) > 1 for v in our_style.values())
csv_style = rows.groupby("painting").art_style.agg(lambda x: sorted(set(x)))
differs = [{"painting": p, "csv_style": csv_style[p][0], "our_style": sorted(our_style[p])[0]}
           for p in sorted(rs & held) if csv_style[p] != sorted(our_style[p])]
out["style_differs"] = differs
out["paintings_multi_style_csv"] = int((csv_style.map(len) > 1).sum())

mine = pd.DataFrame({"row_id": row_id.astype(np.int64), "painting": rows.painting.to_numpy(),
                     "language": rows.language.to_numpy(), "emotion": emotion, "style": style, "genre": genre_csv,
                     "held": rows.painting.isin(rs & held).to_numpy(), "dropped": dropped})
rec = pd.read_parquet(REC / "read_set_a28.parquet")
cmp = {"columns_equal": list(rec.columns) == list(mine.columns), "len_equal": len(rec) == len(mine)}
for c in mine.columns:
    cmp[c] = bool(len(rec) == len(mine) and np.array_equal(rec[c].to_numpy(), mine[c].to_numpy()))
cmp["dtypes"] = {c: str(rec[c].dtype) for c in rec.columns}
out["read_set_parquet_equal"] = cmp
out["held_rows"] = int(mine.held.sum())

# V (own eligible: values with >= 30 distinct paintings on kept rows with all three labels known)
kept = mine[~mine.dropped]
kept = kept[(kept["emotion"] >= 0) & (kept["style"] >= 0) & (kept["genre"] >= 0)]
out["v_pool_rows"] = len(kept)
dev = json.loads(DEV.read_text())
V, elig_all = {}, {}
for a in ("emotion", "style", "genre"):
    npa = kept.groupby(a).painting.nunique()
    elig = sorted(int(v) for v, n in npa.items() if v >= 0 and n >= 30)
    devc = {int(r["code"]) for r in dev[a]}
    elig_all[a] = {"eligible": elig, "outside_dev": sorted(set(elig) - devc),
                   "near_30": {int(v): int(n) for v, n in npa.items() if 25 <= n <= 35}}
    V[a] = sorted(set(elig) & devc)
out["V"] = V
out["V_detail"] = elig_all
vrec = json.loads((REC / "value_sets_a28.json").read_text())
names = {"emotion": emo_names, "style": sty_names, "genre": GENRE_NAMES}
out["V_record_equal"] = vrec == {a: [{"code": c, "name": names[a][c]} for c in V[a]] for a in V}

# language flags (own)
words = np.array([len(x.split()) if isinstance(x, str) else 0 for x in text])
eng_med = float(np.median([len(str(x).split()) for x in csv[on & eng].caption]))
out["english_median_words"] = eng_med


def nonlatin_share(s):
    letters = [ch for ch in s if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(not unicodedata.name(ch, "").startswith("LATIN") for ch in letters) / len(letters)


flags_mine, cov_recount = {}, {}
lang = mine.language.to_numpy()
for L in sorted(set(lang)):
    c = cov[L]
    n = c["rows"]
    failed = c["empty_outputs"] + c["outputs_equal_to_source"] + c["hit_max_new_tokens"]
    script = c["code"].split("_")[-1]
    med = float(np.median(words[(lang == L) & ~dropped]))
    r = med / eng_med
    crit = {"proxy": bool(c["proxy"]), "failed_outputs": failed / n > 0.05,
            "source_script": script != "Latn" and c["outputs_still_in_source_script"] / n > 0.05,
            "length": r < 0.5 or r > 2.0}
    flags_mine[L] = {"failed_share": failed / n, "script": script,
                     "script_share": c["outputs_still_in_source_script"] / n, "median": med, "ratio": r,
                     "crit": crit, "flagged": any(crit.values())}
    # recount coverage.json from translations.parquet (all translated rows of the language)
    tl = trans[trans.language == L]
    cov_recount[L] = {"rows": len(tl) == n,
                      "empty": int((tl.translation.str.strip() == "").sum()) == c["empty_outputs"],
                      "equal_src": int((tl.translation == tl.source).sum()) == c["outputs_equal_to_source"],
                      "src_script": int(sum(nonlatin_share(x) > 0.5 for x in tl.translation))
                      == c["outputs_still_in_source_script"],
                      "proxy_codes": (c["code"] in ("zul_Latn", "swh_Latn")) == bool(c["proxy"])}
out["flagged_mine"] = sorted(k for k, v in flags_mine.items() if v["flagged"])
out["flags_detail"] = {k: v for k, v in flags_mine.items() if v["flagged"]}
out["coverage_recount_bad"] = {k: [f for f, ok in v.items() if not ok] for k, v in cov_recount.items()
                               if not all(v.values())}
out["proxy_languages"] = sorted(k for k, v in cov.items() if v["proxy"])
fr = json.loads((REC / "language_flags_a28.json").read_text())
diff = []
for L, v in flags_mine.items():
    g = fr["languages"][L]["criteria"]
    for k, val in (("failed_outputs", v["failed_share"]), ("source_script", v["script_share"]),
                   ("length", v["ratio"])):
        if g[k]["value"] != val:
            diff.append([L, k, "value", g[k]["value"], val])
    for k in v["crit"]:
        if g[k]["fired"] != v["crit"][k]:
            diff.append([L, k, "fired"])
    if fr["languages"][L]["flagged"] != v["flagged"]:
        diff.append([L, "flagged"])
    if fr["languages"][L]["median_output_words_read"] != v["median"]:
        diff.append([L, "median"])
out["flags_record_diff"] = diff
out["flags_record_flagged"] = fr["flagged"]
out["flags_record_eng_median"] = fr["english_median_words"]
out["flags_record_thresholds"] = fr["thresholds"]

# a28_counts.json and a28_style_differs.json
cnt = json.loads((REC / "a28_counts.json").read_text())
mine_cnt = {"paintings": len(read), "held": len(rs & held), "new": len(rs - held), "train_side": len(a28 & train_side),
            "rows": len(mine), "languages": int(mine.language.nunique()), "english_dropped": int((on & eng).sum()),
            "style_differs": len(differs), "dropped": int(dropped.sum()),
            "rows_by_language": {k: int((lang == k).sum()) for k in sorted(set(lang))},
            "dropped_by_language": {k: int(dropped[lang == k].sum()) for k in sorted(set(lang))},
            "held_rows": int(mine.held.sum()), "english_median_words": eng_med}
out["counts_record_diff"] = {k: [cnt.get(k), v] for k, v in mine_cnt.items() if cnt.get(k) != v}
out["counts_record_extra_keys"] = sorted(set(cnt) - set(mine_cnt))
sd = json.loads((REC / "a28_style_differs.json").read_text())
out["style_differs_record_equal"] = sd == differs

(HERE / "fr_b1_counts.json").write_text(json.dumps(out, indent=1, default=str))
print("written")
