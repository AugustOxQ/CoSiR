"""POST-HOC, DESCRIPTIVE diagnostic (not pre-registered, outside the E3 decision map): did the trained factor models
fit the pseudo-aspect TRAINING task? Scores pseudo-aspect episodes over scorer-train rows with the same scorers E3 uses
on labelled episodes. Run from the repo root (CPU only):

  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python src/test/20261101_aspect_factor_gonogo/train_fit_diagnostic.py

Fresh episodes are NEW episodes (seed 777 + pair index) over scorer-train rows, built from the E2 partitions; they are
not the training bank's episodes. Bank episodes are the first 2,048 of each block of bank_AIC.npz (seen in training by
A1..A6; H1 and S1 trained on other banks).
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(8)

from src.data.artelingo import load_artelingo  # noqa: E402
from src.data.artelingo_splits import artelingo_splits  # noqa: E402
from src.eval.aspect_episodes import (AspectEpisodes, PaintingValueIndex, concat_episodes,  # noqa: E402
                                      validate_aspect_episodes)
from src.eval.aspect_metrics import METRICS, per_anchor, summarize  # noqa: E402
from src.eval.aspect_scorers import (EvalInputs, agreement_term, cosine_scores, crossfit_lambda,  # noqa: E402
                                     fixed_beta_scores)
from src.eval.aspect_episodes import build_aspect_episodes  # noqa: E402
from src.train.train_factors import encode_rows, load_factor_checkpoint  # noqa: E402

E2 = ROOT / "src/test/20261031_pseudo_partitions/results"
GRID_CK = ROOT / "src/test/20261016_factor_learning_grid/checkpoints"
N_EP, BETA = 2048, 0.3
PAIRS = (("affect", "caption", "image"), ("affect", "image", "caption"), ("caption", "image", "affect"))
PAIR_NAMES = [f"{a}__{b}" for a, b, _ in PAIRS]
FIELDS = ("anchor", "candidates", "pairs_a_img", "pairs_a_txt", "pairs_b_img", "pairs_b_txt")
MODELS = {**{r: HERE / "checkpoints" / f"{r}_seed42.pt" for r in ("A1", "A2", "A3", "A4", "A5", "A6", "H1", "S1")},
          "C0": GRID_CK / "C0_seed42.pt",
          "SE": ROOT / "src/test/20261018_affect_factor_learning/checkpoints/SE_seed42.pt",   # as run_affect.model_path("SE")
          "S": GRID_CK / "S_seed42.pt"}                  # extra reference: the factor-learning grid's style cell, NOT SE


def finite_tree(o, where=""):
    if isinstance(o, dict):
        for k, v in o.items():
            finite_tree(v, f"{where}/{k}")
    elif isinstance(o, float):
        assert np.isfinite(o), f"non-finite at {where}"


def main():
    data = load_artelingo()
    sp = artelingo_splits(data)
    st = np.asarray(sp.scorer_train)
    n = len(st)
    img, txt = data.img_features[st], data.txt_features[st]
    part = np.load(E2 / "partitions.npz")
    labels = {k: part[k] for k in ("affect", "image", "caption")}
    groups = part["local_groups"]
    assert len(groups) == n == len(labels["affect"]), (len(groups), n)
    index = PaintingValueIndex(labels, groups)
    rows = np.arange(n)

    fresh_parts = []
    for i, (a, b, third) in enumerate(PAIRS):
        ep = build_aspect_episodes(labels, groups, rows, a, b, N_EP, 777 + i, third=third, index=index)
        validate_aspect_episodes(ep, labels, groups, index, third=third)
        assert ep.rows().max() < n
        fresh_parts.append(ep)
        print("fresh episodes built and validated:", a, b, "third", third, flush=True)
    z = np.load(E2 / "bank_AIC.npz")
    assert list(z["block_pairs"]) == PAIR_NAMES
    edges = np.concatenate([[0], np.cumsum(z["block_sizes"])])
    bank_parts = []
    for i, (a, b, _) in enumerate(PAIRS):
        sl = slice(int(edges[i]), int(edges[i]) + N_EP)
        bank_parts.append(AspectEpisodes(a, b, *(z[f][sl].astype(np.int64) for f in FIELDS)))
    sets = {"fresh": concat_episodes(fresh_parts), "bank": concat_episodes(bank_parts)}
    pair_index = np.repeat(np.arange(3), N_EP)
    clusters = groups[sets["fresh"].anchor]
    cl = {k: groups[v.anchor] for k, v in sets.items()}
    parity = np.arange(3 * N_EP) % 2

    def summ(pa, k, mask):
        return summarize({m: pa[m][mask] for m in METRICS}, cl[k][mask])

    def report(pa, k):
        out = {"pooled": summ(pa, k, np.ones(3 * N_EP, bool))}
        for i, name in enumerate(PAIR_NAMES):
            out[name] = summ(pa, k, pair_index == i)
        return out

    res = {"note": "POST-HOC, DESCRIPTIVE; not pre-registered; outside the E3 decision map",
           "fresh_episodes": "new episodes (seed 777+pair index) over scorer-train rows, not the training bank's",
           "n_per_pair": N_EP, "beta": BETA, "models": {}, "cosine": {}, "checkpoint_sha256": {}}
    cos = {}
    for k, ep in sets.items():
        cos[k] = cosine_scores(EvalInputs(img, txt), ep)
        res["cosine"][k] = report(per_anchor(cos[k]), k)
    for name, path in MODELS.items():
        res["checkpoint_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        model, _ = load_factor_checkpoint(path, device="cpu")
        ic, tc = encode_rows(model, data.img_features, data.txt_features, rows=st, device="cpu")
        assert np.isfinite(ic).all() and np.isfinite(tc).all() and len(ic) == n
        res["models"][name] = {}
        for k, ep in sets.items():
            inp = EvalInputs(img, txt, ic, tc)
            entry = {}
            for label, uniform in (("agreement", False), ("uniform", True)):
                term = agreement_term(inp, ep, uniform=uniform)
                fused, _ = crossfit_lambda(cos[k], term, parity)
                entry[label] = report(per_anchor(fused), k)
            entry["fixed_beta"] = report(per_anchor(fixed_beta_scores(inp, ep, BETA)), k)
            res["models"][name][k] = entry
        print("scored", name, flush=True)
    assert res["checkpoint_sha256"]["SE"].startswith("93add21b") and res["checkpoint_sha256"]["S"].startswith("33d35943")
    finite_tree(res)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "train_fit_diagnostic.json").write_text(json.dumps(res, indent=1))

    def c(x):
        return f"{x['point']:5.2f}"

    lines = []
    for k in ("fresh", "bank"):
        for scorer in ("agreement", "uniform", "fixed_beta"):
            lines += [f"\n[{k} episodes, pooled over 3 pairs; {scorer}]  R@1 / gain / other / swap  (cosine: "
                      + " / ".join(c(res['cosine'][k]['pooled'][m]) for m in ("r1", "gain", "other", "swap")) + ")"]
            for name in MODELS:
                p = res["models"][name][k][scorer]["pooled"]
                lines.append(f"  {name:3s} " + " / ".join(c(p[m]) for m in ("r1", "gain", "other", "swap")))
    print("\n".join(lines))
    (HERE / "results" / "train_fit_diagnostic.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
