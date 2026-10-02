"""Coverage of ArtGAN WikiArt genre labels on ArtELingo paintings, per split (spec §5.3 rule). Metadata only:
reads painting ids, split membership and the genre CSVs; no model, no evaluation result, no held episode."""
import collections, csv, importlib.util, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
from src.data.artelingo import load_artelingo  # noqa: E402
from src.data.splits import grouped_split, leakage_groups  # noqa: E402
spec = importlib.util.spec_from_file_location("ra", ROOT / "src/test/20261018_affect_factor_learning/run_affect.py")
ra = importlib.util.module_from_spec(spec); spec.loader.exec_module(ra)

GENRE_DIR = Path("/data/SSD/wikiart_genre")
data = load_artelingo(); cache, *_ = ra.grid.load_grid()
split = grouped_split(leakage_groups(data.paintings, data.img_features), seed=42)
assert np.array_equal(np.sort(np.concatenate([cache["scorer_train"], cache["selection"]])), np.sort(split.train))
paint = np.asarray(data.paintings)

ids = {}
for f in ("genre_train.csv", "genre_val.csv"):
    for path, k in csv.reader(open(GENRE_DIR / f)):
        name = Path(path).stem
        if name in ids and ids[name] != int(k):
            ids[name] = -1                                     # conflicting labels: drop
        else:
            ids.setdefault(name, int(k))
print("ArtGAN genre entries", len(ids), "conflicts", sum(v == -1 for v in ids.values()))

# id -> name, recovered from the ArtELingo-28 genre names (genre_class.txt is gone upstream)
g28 = json.load(open("/data/PDD/artelingo/artelingo_genre_emotion_eng.json"))
votes = collections.defaultdict(collections.Counter)
for r in g28:
    k = ids.get(r["painting"])
    if k is not None and k >= 0:
        votes[k][r["genre"]] += 1
names = {k: c.most_common(1)[0][0] for k, c in votes.items()}
purity = {k: round(c.most_common(1)[0][1] / sum(c.values()), 3) for k, c in votes.items()}
print("id -> name (purity over ArtELingo-28 rows):", {k: (names[k], purity[k]) for k in sorted(names)})

out = {"mapping": {int(k): names[k] for k in sorted(names)}, "purity": {int(k): purity[k] for k in purity}}
for part, rows in (("scorer_train", cache["scorer_train"]), ("selection", cache["selection"]),
                   ("val", split.val), ("held", split.held)):
    ps = np.unique(paint[rows])
    lab = [ids.get(p) for p in ps]
    covered = [k for k in lab if k is not None and k >= 0]
    per = collections.Counter(names.get(k, f"id{k}") for k in covered)
    out[part] = {"paintings": int(len(ps)), "covered": len(covered), "share": round(len(covered) / len(ps), 4),
                 "per_genre": dict(per.most_common()), "min_genre": min(per.values()) if per else 0}
    print(f"{part:12s} paintings {len(ps):6d} covered {len(covered):6d} ({100*len(covered)/len(ps):.1f}%)  "
          f"min per genre {out[part]['min_genre']}  {dict(per.most_common())}")
ok = all(out[p]["share"] >= 0.5 and out[p]["min_genre"] >= 30 for p in ("selection", "held"))
out["rule_keep_genre"] = ok
print("spec §5.3 rule (>= 50% of selection and held paintings, >= 30 per genre in each):", "KEEP" if ok else "FAIL")
(Path(__file__).parent / "genre_coverage.json").write_text(json.dumps(out, indent=1))
