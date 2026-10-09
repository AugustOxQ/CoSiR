"""Scoped re-review (r6 fix wave), item 1: rule section 5 item 2's development value sets, re-derived from the rule's
text with my own code, then compared with what the regression path writes to value_sets.json.

Mine: labels coded as the rule's artelingo_aspect_labels (sorted names, catch-all 'something else' excluded, genre from
the WikiArt CSVs), groups and selection read from prepare.npz (the rule's section 5 item 1 asserts the recomputed split
equals them; SHA-256 checked here), pool_sel = selection rows with all three labels known, eligible = at least 30
distinct paintings (groups) among pool rows with that value. No feature is loaded; no held row is indexed.

The code's: run_r6_held.load_rows() (the regression's own setup step) and r6_common.write_value_sets into a temp
folder, exactly as run_regression calls it.

Usage: python rr_value_sets.py <out_dir>   -> <out_dir>/rr_value_sets.json (summary), <out_dir>/value_sets.json
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

MAIN = Path("/project/CoSiR")
sys.path.insert(0, str(MAIN))
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

from src.data.artelingo import ANNOTATIONS_PATH, FEATURE_DIR, join_annotations, join_art_styles  # noqa: E402
from src.data.wikiart_genre import GENRE_NAMES, load_wikiart_genre  # noqa: E402
from src.utils import FeatureManager  # noqa: E402

PREPARE = MAIN / "src/test/20261013_stage_d_selection/cache/prepare.npz"
PREPARE_SHA = "d30f0281bb521d18c7a4d4adca1a7938689e2789db7fd528ce0322b90ac44c8b"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def code(values, exclude=()):
    vals = list(np.asarray(values, dtype=object).tolist())
    bad = {"", None, *exclude}
    names = sorted({v for v in vals if v not in bad})
    look = {n: i for i, n in enumerate(names)}
    return np.asarray([look.get(v, -1) for v in vals], dtype=np.int64), names


def mine():
    assert sha(PREPARE) == PREPARE_SHA
    with np.load(PREPARE, allow_pickle=False) as z:
        groups, selection = z["groups"], z["selection"]
    ids = np.asarray(FeatureManager(storage_dir=FEATURE_DIR).get_all_sample_ids(), dtype=np.int64)
    ann = json.loads(Path(ANNOTATIONS_PATH).read_text())
    assert len(ids) == len(ann) == 308_723 == len(groups)
    emotions, paintings = join_annotations(ids, ann)              # C6: joined by extraction sample_ids
    styles = join_art_styles(ids, ann)
    emo, emo_names = code(emotions, exclude=("something else",))
    sty, sty_names = code(styles)
    gen = load_wikiart_genre(np.asarray(paintings, dtype=object))
    labels = {"emotion": emo, "style": sty, "genre": gen}
    names = {"emotion": emo_names, "style": sty_names, "genre": list(GENRE_NAMES)}
    sel = np.asarray(selection, dtype=np.int64)
    pool = sel[(emo[sel] >= 0) & (sty[sel] >= 0) & (gen[sel] >= 0)]
    out = {}
    for x, lab in labels.items():
        vals = []
        for v in sorted(set(lab[pool].tolist())):
            if v < 0:
                continue
            if len(set(groups[pool[lab[pool] == v]].tolist())) >= 30:
                vals.append(int(v))
        out[x] = [{"code": v, "name": str(names[x][v])} for v in vals]
    return out, {"n_selection": int(len(sel)), "n_pool": int(len(pool))}


def theirs(tmp):
    F = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(F))
    import r6_common as R
    import run_r6_held as RH
    rows = RH.load_rows()
    path = tmp / "value_sets.json"
    R.write_value_sets(path, rows.value_sets, rows.data)            # as run_regression calls it
    return json.loads(path.read_text()), sha(path)


if __name__ == "__main__":
    m, meta = mine()
    t, t_sha = theirs(OUT)
    res = {"mine_counts": {x: len(v) for x, v in m.items()}, "theirs_counts": {x: len(v) for x, v in t.items()},
           "equal": m == t, "theirs_sha256": t_sha, **meta,
           "mine_names": {x: [e["name"] for e in v] for x, v in m.items()}}
    (OUT / "rr_value_sets.json").write_text(json.dumps(res, indent=1))
    print("counts mine", res["mine_counts"], "theirs", res["theirs_counts"], "equal", res["equal"], "pool",
          meta["n_pool"])
