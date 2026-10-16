"""Check 3: ArtELingo-28 inventory and between-language emotion disagreement (Jensen-Shannon, log2)."""
import json, collections
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).parent
RAW = Path("/data/SSD2/pre_extract/artelingo_raw_json")
df = pd.read_parquet("/data/SSD2/pre_extract/artelingo28/read_set_a28.parquet")
EMO8 = ["amusement", "anger", "awe", "contentment", "disgust", "excitement", "fear", "sadness"]  # sorted names; 'something else' = -1
res = {"n_rows": int(len(df)), "n_paintings": int(df.painting.nunique()), "n_languages": int(df.language.nunique()),
       "dropped_rows": int(df.dropped.sum()), "held_paintings": int(df.groupby("painting").held.first().sum()),
       "captions_per_language": df.language.value_counts().to_dict(),
       "captions_per_painting_mean": float(len(df) / df.painting.nunique()),
       "emotion_values_per_caption": "own label in column `emotion` (int; -1 = catch-all 'something else', inferred from the sorted-name coding of src/data/artelingo_splits.encode_labels)",
       "emotion_counts": {int(k): int(v) for k, v in df.emotion.value_counts().sort_index().items()},
       "genre_labelled_rows": int((df.genre >= 0).sum()), "style_one_per_painting": bool((df.groupby("painting").style.nunique() == 1).all())}
nl = df.groupby("painting").language.nunique()
res["paintings_with_>=3_languages"] = int((nl >= 3).sum())
res["languages_per_painting_quantiles(min,25,50,75,max)"] = [float(x) for x in nl.quantile([0, .25, .5, .75, 1])]
cpl = df.groupby(["painting", "language"]).size()
res["captions_per_painting_language_mean"] = float(cpl.mean())
res["painting_language_groups_with_>=3_captions"] = int((cpl >= 3).sum())
res["raw_json_note"] = {}
# raw json (English only)
eng = {}
for f in ("artelingo_train", "artelingo_val_test", "artelingo_genre_emotion_eng"):
    d = json.load(open(RAW / f"{f}.json"))
    res["raw_json_note"][f] = {"rows": len(d), "paintings": len({x["painting"] for x in d}), "keys": sorted(d[0].keys()),
                               "has_language_field": "language" in d[0], "has_per_caption_emotion": all(bool(x.get("emotion")) for x in d)}
    for x in d:
        eng.setdefault(x["painting"], set()).add((x["caption"], x["emotion"]))
ps = set(df.painting)
e_ps = [p for p in ps if p in eng]
res["english_in_raw_json"] = {"parquet_paintings_with_english_captions": len(e_ps),
                              "english_captions_for_them": int(sum(len(eng[p]) for p in e_ps))}
res["english_in_raw_json"]["note"] = "raw json files are English only; the parquet has no English rows (27 non-English languages)"
emo_idx = {n: i for i, n in enumerate(EMO8)}

def js_pairs(H):
    """mean pairwise JS (log2) between the rows of H (G x 9 count matrix), or nan if G<2. Also returns pair count."""
    G = len(H)
    if G < 2: return np.nan
    Pm = H / H.sum(1, keepdims=True)
    i, j = np.triu_indices(G, 1)
    M = 0.5 * (Pm[i] + Pm[j])
    def kl(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(a > 0, a * np.log2(a / b), 0).sum(1)
    return float((0.5 * kl(Pm[i], M) + 0.5 * kl(Pm[j], M)).mean())

def analyse(frame, min_group=1, n_perm=20, seed=0, min_groups=2):
    rng = np.random.default_rng(seed)
    real, rand, used = [], [], []
    for p, g in frame.groupby("painting"):
        lab = (g.emotion.values + 1) % 9  # -1 -> 0 ... keep 9 bins
        grp = g.language.values
        langs, inv = np.unique(grp, return_inverse=True)
        sizes = np.bincount(inv)
        keep = sizes >= min_group
        if keep.sum() < min_groups: continue
        def hist(labels, inv_):
            H = np.zeros((len(langs), 9)); np.add.at(H, (inv_, labels), 1); return H[keep]
        real.append(js_pairs(hist(lab, inv)))
        # random groups of the same sizes on the same painting: permute the caption->group assignment
        rr = [js_pairs(hist(lab, inv[rng.permutation(len(inv))])) for _ in range(n_perm)]
        rand.append(float(np.mean(rr))); used.append(p)
    real, rand = np.array(real), np.array(rand)
    diff = real - rand
    return {"n_paintings": int(len(real)), "JS_between_languages": float(real.mean()), "JS_random_same_size_groups": float(rand.mean()),
            "mean_paired_diff": float(diff.mean()), "paired_diff_se": float(diff.std(ddof=1) / np.sqrt(len(diff))),
            "frac_paintings_real_gt_random": float((diff > 0).mean())}

res["js"] = {
    "all_paintings_>=2_languages": analyse(df),
    "paintings_>=3_languages": analyse(df[df.painting.isin(nl[nl >= 3].index)]),
    "groups_with_>=3_captions_only (paintings with >=2 such groups)": analyse(df, min_group=3),
}
# with an English group added for paintings that have English captions in the raw json
rows = []
for p in e_ps:
    for cap, em in eng[p]:
        rows.append((p, "English", emo_idx.get(em, -1)))
dfe = pd.concat([df[["painting", "language", "emotion"]], pd.DataFrame(rows, columns=["painting", "language", "emotion"])])
dfe = dfe[dfe.painting.isin(e_ps)]
res["js"]["with_English_group_added (the paintings that have English captions)"] = analyse(dfe)
res["js"]["same_paintings_without_English (parquet only)"] = analyse(df[df.painting.isin(e_ps)])
res["js_note"] = "JS = Jensen-Shannon divergence, log base 2 (0 to 1), mean over language pairs per painting, then over paintings; random = 20 permutations of the caption-to-language assignment of the same painting (group sizes kept); 9 emotion bins incl. catch-all"
json.dump(res, open(HERE / "results_check3.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ("captions_per_language",)}, indent=1))
