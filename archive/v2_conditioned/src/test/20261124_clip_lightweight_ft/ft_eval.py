"""Task 4: descriptive evaluation of the fine-tuned CLIP variants on the aspect episodes (spec section 5).

  python ft_eval.py --features LP=runs/LP/features.npz LB=... --out results/eval.json
  python ft_eval.py --self-check            # frozen CLIP as a stand-in variant; prints only pass/fail, writes nothing

Each features.npz holds `rows` (int64 feature-row indices of the val and selection rows) and `img`, `txt` (float32,
(len(rows), 512), raw projection outputs). They are placed in feature-row order with NaN outside the selection rows
(val rows are dropped). The cosine of a variant comes from src.eval.aspect_scorers.cosine_scores and per_anchor on the
episodes of seeds 42, 49, 50 and 51. Plain CLIP goes through the same path on the cached frozen features and is
asserted equal to the stored `cosine__*` arrays. AFF, B, B'(A0) and B'(A1) are read from earlier rounds' stored
per-anchor arrays (SOURCES below). Statistics are in percentage points with a painting-cluster bootstrap
(src.eval.aspect_metrics.cluster_bootstrap: 5,000 resamples, seed 42, chunk 250); the pooled scope concatenates seeds
49, 50, 51 in that order with one cluster per painting across seeds.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from src.eval.aspect_episodes import AspectEpisodes, concat_episodes, episodes_sha256  # noqa: E402
from src.eval.aspect_metrics import METRICS, cluster_bootstrap, per_anchor  # noqa: E402
from src.eval.aspect_scorers import EvalInputs, cosine_scores  # noqa: E402

E1 = ROOT / "src/test/20261030_aspect_baselines/results"
R3 = ROOT / "src/test/20261121_round3_affect_gate/results"
R4 = ROOT / "src/test/20261122_round4_aff_vetoes/results"
SEEDS = (42, 49, 50, 51)
POOLED_SEEDS = (49, 50, 51)
PAIRS = (("emotion", "style"), ("emotion", "genre"), ("style", "genre"))
POOLED_ORDER = [f"{a}__{b}" for a, b in PAIRS]
FIELDS = ("anchor", "candidates", "pairs_a_img", "pairs_a_txt", "pairs_b_img", "pairs_b_txt")
REPORT_METRICS = ("r1", "either", "gain", "other")   # either = r1 + other (round 1 common.either)
# scorer name -> (file, key prefix); the metric name follows "__". Seeds 49-51: round 3's go_seed{s}.npz; seed 42: round 4.
SOURCES = {
    "49-51": {"AFF": (R3 / "go_seed{s}.npz", "aff_fused"), "B": (R3 / "go_seed{s}.npz", "B"),
              "Bp0": (R3 / "go_seed{s}.npz", "Bp")},
    "42": {"AFF": (R4 / "seed42_arrays.npz", "aff_fused"), "B": (R4 / "seed42_arrays.npz", "B"),
           "Bp0": (R4 / "seed42_arrays.npz", "Bp0"), "Bp1": (R4 / "seed42_arrays.npz", "Bp1")},
}
COMPARISONS = {"AFF_minus_ft": ("AFF", "ft"), "ft_minus_plain": ("ft", "plain"), "ft_minus_Bp0": ("ft", "Bp0")}


# ----- pure core -------------------------------------------------------------------------------------------------

def place_features(rows, img, txt, n_rows, selection, allowed_extra=None):
    """Selection-row features in feature-row order (NaN elsewhere, val rows dropped). Asserts: rows unique, every row
    in selection (or allowed_extra, the val rows), every selection row present, selection features finite."""
    rows = np.asarray(rows, dtype=np.int64)
    selection = np.asarray(selection, dtype=np.int64)
    if len(np.unique(rows)) != len(rows):
        raise AssertionError("features.npz rows are not unique")
    if not (len(img) == len(txt) == len(rows)):
        raise AssertionError("rows, img and txt differ in length")
    ok = np.isin(rows, selection)
    if allowed_extra is not None:
        ok |= np.isin(rows, np.asarray(allowed_extra, dtype=np.int64))
    if not ok.all():
        raise AssertionError(f"{int((~ok).sum())} rows are neither selection nor val rows")
    if not np.isin(selection, rows).all():
        raise AssertionError("a selection row is missing from features.npz")
    keep = np.isin(rows, selection)
    out = []
    for values in (img, txt):
        values = np.asarray(values, dtype=np.float32)
        full = np.full((n_rows, values.shape[1]), np.nan, dtype=np.float32)
        full[rows[keep]] = values[keep]
        if not np.isfinite(full[selection]).all():
            raise AssertionError("non-finite feature on a selection row")
        mask = np.zeros(n_rows, dtype=bool)
        mask[selection] = True
        if not np.isnan(full[~mask]).all():
            raise AssertionError("selection masking failed")
        out.append(full)
    return out[0], out[1]


def score_features(img, txt, ep) -> dict:
    """Per-anchor metrics of the plain cosine of (img, txt) on episodes ep."""
    return per_anchor(cosine_scores(EvalInputs(img, txt), ep))


def ci(values, clusters) -> dict:
    """Mean, 95% painting-bootstrap interval and cluster count, in percentage points."""
    r = cluster_bootstrap(np.asarray(values, dtype=np.float64), clusters)
    return {"point": 100 * r["point"], "ci95": [100 * c for c in r["ci95"]], "n_clusters": r["n_clusters"]}


def assert_aligned(cl_a, pair_a, cl_b, pair_b, what: str):
    if not (np.array_equal(cl_a, cl_b) and np.array_equal(pair_a, pair_b)):
        raise AssertionError(f"{what}: anchor paintings or pair index differ from the episodes")


def metric_values(per_anchor_dict, metric):
    """Per-anchor values of a reported metric; either = R@1 + other-aspect rate."""
    if metric == "either":
        return (np.asarray(per_anchor_dict["r1"], dtype=np.float64)
                + np.asarray(per_anchor_dict["other"], dtype=np.float64))
    return np.asarray(per_anchor_dict[metric], dtype=np.float64)


def pool(data, seeds, scorer, metric):
    return np.concatenate([metric_values(data[s]["scorers"][scorer], metric) for s in seeds])


def _scope(data, seeds, variants, mask_fn=None):
    """Summary of one scope: scorers present on every seed of the scope, and the comparisons for each variant."""
    cl = np.concatenate([np.asarray(data[s]["cl"]) for s in seeds])
    mask = np.ones(len(cl), bool) if mask_fn is None else np.concatenate([mask_fn(data[s]) for s in seeds])
    names = [n for n in data[seeds[0]]["scorers"] if all(n in data[s]["scorers"] for s in seeds)]
    vals = {(n, m): pool(data, seeds, n, m)[mask] for n in names for m in REPORT_METRICS}
    out = {"n_episodes": int(mask.sum()), "seeds": list(seeds),
           "scorers": {n: {m: ci(vals[n, m], cl[mask]) for m in REPORT_METRICS} for n in names}, "comparisons": {}}
    for v in variants:
        ft = f"ft:{v}"
        pick = lambda tag: ft if tag == "ft" else tag
        out["comparisons"][v] = {
            cname: {m: ci(vals[pick(a), m] - vals[pick(b), m], cl[mask]) for m in REPORT_METRICS}
            for cname, (a, b) in COMPARISONS.items()}
    return out


def build_report(data, variants, pooled_seeds=POOLED_SEEDS) -> dict:
    """data[seed] = {"cl": (E,) anchor painting ids, "pair_index": (E,), "scorers": {name: {metric: (E,)}}} with names
    plain, B, Bp0, AFF, ('Bp1' on seed 42 only) and 'ft:<variant>'."""
    rep = {"seeds": {str(s): _scope(data, (s,), variants) for s in data}}
    rep["pooled"] = _scope(data, tuple(pooled_seeds), variants)
    rep["pairs"] = {str(i): _scope(data, tuple(pooled_seeds), variants, lambda d, i=i: d["pair_index"] == i)
                    for i in np.unique(data[pooled_seeds[0]]["pair_index"])}
    if 42 in data:
        rep["pairs_seed42"] = {str(i): _scope(data, (42,), variants, lambda d, i=i: d["pair_index"] == i)
                               for i in np.unique(data[42]["pair_index"])}
    return rep


def format_table(rep, variants) -> str:
    def f(r):
        return f"{r['point']:+7.2f} [{r['ci95'][0]:+6.2f},{r['ci95'][1]:+6.2f}]"
    lines = []
    scopes = [(f"seed {k}", v) for k, v in rep["seeds"].items()] + [("pooled 49-51", rep["pooled"])] + \
             [(f"pair {POOLED_ORDER[int(k)]} (49-51)", v) for k, v in rep["pairs"].items()] + \
             [(f"pair {POOLED_ORDER[int(k)]} (seed 42)", v) for k, v in rep.get("pairs_seed42", {}).items()]
    for title, sc in scopes:
        lines.append(f"== {title}  (n={sc['n_episodes']}, pp, 95% painting-bootstrap)")
        lines.append(f"{'scorer':<26}" + "".join(f"{m:>25}" for m in REPORT_METRICS))
        for n, r in sc["scorers"].items():
            lines.append(f"{n:<26}" + "".join(f"{f(r[m]):>25}" for m in REPORT_METRICS))
        for v in variants:
            for cname, r in sc["comparisons"][v].items():
                lines.append(f"{v + ': ' + cname:<26}" + "".join(f"{f(r[m]):>25}" for m in REPORT_METRICS))
        lines.append("")
    return "\n".join(lines)


# ----- real-data loading -----------------------------------------------------------------------------------------

def load_context():
    from src.data.artelingo import load_artelingo
    from src.data.artelingo_splits import artelingo_splits
    data = load_artelingo()
    sp = artelingo_splits(data)
    return data, sp


def load_episodes(seed, selection):
    z = np.load(E1 / f"episodes_seed{seed}.npz")
    base = json.loads((E1 / f"baselines_seed{seed}.json").read_text())
    if list(z["pair_order"]) != POOLED_ORDER or base["pair_order"] != POOLED_ORDER:
        raise AssertionError(f"pair order must be {POOLED_ORDER}")
    parts = []
    for a, b in PAIRS:
        ep = AspectEpisodes(a, b, *(z[f"{a}__{b}__{k}"].astype(np.int64) for k in FIELDS))
        if episodes_sha256(ep) != base["episodes_sha256"][f"{a}__{b}"]:
            raise AssertionError(f"episodes_seed{seed} {a}__{b}: SHA-256 differs from baselines_seed{seed}.json")
        if not np.isin(ep.rows(), selection).all():
            raise AssertionError(f"episodes_seed{seed} {a}__{b}: a row outside selection")
        parts.append(ep)
    n_per = len(parts[0].anchor)
    if any(len(p.anchor) != n_per for p in parts) or n_per != base["n_per_pair"]:
        raise AssertionError("episode counts differ between pairs or from the baselines record")
    return concat_episodes(parts), np.repeat(np.arange(len(PAIRS)), n_per)


def load_stored(seed):
    """AFF, B, Bp0 (and Bp1 on seed 42) per-anchor arrays, plus the stored cl and pair_index to align against."""
    src = SOURCES["42"] if seed == 42 else SOURCES["49-51"]
    scorers, cl, pi, stored_cos = {}, None, None, {}
    for name, (path, prefix) in src.items():
        z = np.load(str(path).format(s=seed))
        stored_cos[name] = {m: np.asarray(z[f"cosine__{m}"], dtype=np.float64) for m in METRICS}
        scorers[name] = {m: np.asarray(z[f"{prefix}__{m}"], dtype=np.float64) for m in METRICS}
        if cl is None:
            cl, pi = z["cl"], z["pair_index"]
    return scorers, cl, pi, stored_cos


def assert_stored_cosine(seed, plain, stored_cos):
    """Every stored scorer file's own cosine__* equals the recomputed plain cosine (ties each stored row to the episodes)."""
    for name, cos in stored_cos.items():
        for m in METRICS:
            if not np.array_equal(plain[m], cos[m]):
                raise AssertionError(f"seed {seed}: stored cosine__{m} next to {name} differs from the episodes' cosine")


def assert_plain_matches(seed, plain):
    """Plain CLIP recomputed from the frozen cached features equals the stored cosine__* arrays exactly."""
    z = np.load(E1 / f"per_anchor_seed{seed}.npz")
    for m in METRICS:
        if not np.array_equal(plain[m], z[f"cosine__{m}"]):
            raise AssertionError(f"seed {seed}: plain cosine {m} differs from per_anchor_seed{seed}.npz")


def run(features: dict, out: Path | None, self_check: bool = False, seeds=SEEDS, quiet: bool = False):
    data, sp = load_context()
    n_rows = len(sp.groups)
    sel = np.asarray(sp.selection)
    allowed_val = np.asarray(sp.val)
    frozen = place_features(np.concatenate([sel, allowed_val]),
                            data.img_features[np.concatenate([sel, allowed_val])],
                            data.txt_features[np.concatenate([sel, allowed_val])], n_rows, sel, allowed_val)
    variants_feats = {}
    for name, path in features.items():
        z = np.load(path)
        variants_feats[name] = place_features(z["rows"], z["img"], z["txt"], n_rows, sel, allowed_val)
    if self_check:       # real features.npz path: frozen val+selection features in a shuffled row order
        import tempfile
        both = np.concatenate([sel, allowed_val])
        both = both[np.random.default_rng(0).permutation(len(both))]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "features.npz"
            np.savez(p, rows=both, img=data.img_features[both], txt=data.txt_features[both])
            z = np.load(p)
            variants_feats["frozen_standin"] = place_features(z["rows"], z["img"], z["txt"], n_rows, sel, allowed_val)
        if not all(np.array_equal(a, b, equal_nan=True) for a, b in zip(variants_feats["frozen_standin"], frozen)):
            raise AssertionError("shuffled stand-in placement differs from the ordered placement")
    per_seed, checks = {}, {}
    for s in seeds:
        ep, pair_index = load_episodes(s, sel)
        cl = sp.groups[ep.anchor]
        scorers, cl_s, pi_s, stored_cos = load_stored(s)
        assert_aligned(cl_s, pi_s, cl, pair_index, f"seed {s} stored arrays")
        plain = score_features(*frozen, ep)
        assert_plain_matches(s, plain)
        assert_stored_cosine(s, plain, stored_cos)
        scorers["plain"] = plain
        for name, (fi, ft) in variants_feats.items():
            scorers[f"ft:{name}"] = score_features(fi, ft, ep)
        per_seed[s] = {"cl": cl, "pair_index": pair_index, "scorers": scorers}
        checks[s] = "ok"
        if self_check:
            same = all(np.array_equal(scorers["ft:frozen_standin"][m], plain[m]) for m in METRICS)
            if not same:
                raise AssertionError(f"seed {s}: frozen stand-in differs from plain")
    if self_check:
        print("PASS self-check: seeds", list(seeds), "episodes SHA-256, rows inside selection, stored-array alignment "
              "(cl, pair_index, stored cosine__*), plain == per_anchor cosine__*, shuffled npz placement, stand-in == plain")
        return None
    names = list(variants_feats)
    rep = build_report(per_seed, names)
    rep["alignment_checks"] = {str(s): v for s, v in checks.items()}
    rep["sources"] = {k: {n: [str(p), pre] for n, (p, pre) in v.items()} for k, v in SOURCES.items()}
    rep["sources"]["plain"] = "frozen cached CLIP features, same path; asserted equal to per_anchor_seed{s}.npz cosine__*"
    table = format_table(rep, names)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, indent=1))
        out.with_suffix(".txt").write_text(table)
    if quiet:
        print(f"PASS alignment checks, seeds {list(seeds)}; wrote {out}")
    else:
        print(table)
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", nargs="*", default=[], metavar="VARIANT=features.npz")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "eval.json")
    ap.add_argument("--self-check", action="store_true", help="alignment asserts with frozen CLIP as a stand-in")
    ap.add_argument("--quiet", action="store_true", help="print only the output path and pass/fail lines")
    a = ap.parse_args()
    if any("=" not in kv for kv in a.features):
        ap.error("--features entries must be VARIANT=path/to/features.npz")
    feats = dict(kv.split("=", 1) for kv in a.features)
    if not feats and not a.self_check:
        ap.error("give --features or --self-check")
    run(feats, a.out, self_check=a.self_check, quiet=a.quiet)


if __name__ == "__main__":
    main()
