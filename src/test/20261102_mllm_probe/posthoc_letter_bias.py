"""POST-HOC, DESCRIPTIVE (not pre-registered; outside the decision map; written after the final review of 2026-10-03):
letter-position preference of the in-context MLLM probe and the probe's resolution. CPU, stored files only.

For v1 (results/) and v2 (results/v2/), from probe_partial.npz: scores[c, d, i] are the 13 letter scores mapped back to
the episode's candidate columns, and perms[i, c, d, j] is the column shown under letter j. The letter that got the top
score in a ranking is therefore j with perms[i, c, d, j] == argmax(scores[c, d, i]). Because the candidates were
permuted at random before lettering, a letter preference adds noise but cannot favour the target over the other aspect's
candidate in expectation. The probe's resolution is read from probe.json: the half-width of the pooled paired
condition-gain interval (MLLM minus cosine) is the smallest point gain whose interval would have cleared zero; the
80%-power value uses the normal approximation (1.96 + 0.84) * SE with SE = half-width / 1.96.

Run from the repository root:
  /root/miniconda3/envs/CoSiR/bin/python src/test/20261102_mllm_probe/posthoc_letter_bias.py
Writes results/posthoc_letter_bias.json.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import chisquare

HERE = Path(__file__).resolve().parent
LETTERS = "ABCDEFGHIJKLM"
RUNS = {"v1 (pre-fix prompt)": HERE / "results", "v2 (fixed prompt)": HERE / "results" / "v2"}


def letter_histogram(scores: np.ndarray, perms: np.ndarray) -> tuple[np.ndarray, int]:
    n_cond, n_dir, n_ep, k = scores.shape
    counts, ties = np.zeros(k, dtype=np.int64), 0
    for c in range(n_cond):
        for d in range(n_dir):
            s = scores[c, d]
            assert np.isfinite(s).all(), "non-finite letter score"
            top = s.argmax(1)
            ties += int(((s == s.max(1, keepdims=True)).sum(1) > 1).sum())
            p = perms[:, c, d, :]                                   # (n_ep, 13): column shown under letter j
            letter = np.argmax(p == top[:, None], axis=1)
            assert (p[np.arange(n_ep), letter] == top).all()
            counts += np.bincount(letter, minlength=k)
    return counts, ties


def main() -> None:
    out = {"note": "POST-HOC, DESCRIPTIVE; not pre-registered; outside the decision map", "uniform_share": 100 / 13,
           "runs": {}}
    for tag, folder in RUNS.items():
        z = np.load(folder / "probe_partial.npz")
        rec = json.loads((folder / "probe.json").read_text())
        assert int(z["done"]) == z["scores"].shape[2] == 3 * rec["n_per_pair"]
        counts, ties = letter_histogram(z["scores"], z["perms"])
        total = int(counts.sum())
        g = rec["compare_pooled"]["gain"]
        half = 0.5 * (g["ci95"][1] - g["ci95"][0])
        out["runs"][tag] = {
            "rankings": total, "top_score_ties": ties,
            "top_letter_counts": {L: int(c) for L, c in zip(LETTERS, counts)},
            "top_letter_share_pct": {L: 100 * float(c) / total for L, c in zip(LETTERS, counts)},
            "chi2_vs_uniform": {"statistic": float(chisquare(counts).statistic),
                                "p_value": float(chisquare(counts).pvalue), "df": len(LETTERS) - 1},
            "gain_interval_half_width": half, "gain_80pct_power": (1.96 + 0.84) * half / 1.96,
            "compare_pooled_gain": g, "probe_partial_sha256": hashlib.sha256(
                (folder / "probe_partial.npz").read_bytes()).hexdigest()}
        share = out["runs"][tag]["top_letter_share_pct"]
        print(f"{tag}: {total} rankings, top-score ties {ties}; top letter shares (uniform {100 / 13:.1f}%): "
              + ", ".join(f"{L} {share[L]:.1f}" for L in LETTERS))
        print(f"   most chosen {max(share, key=share.get)} ({counts.max()}/{total}), least chosen "
              f"{min(share, key=share.get)} ({counts.min()}/{total}); chi2 {out['runs'][tag]['chi2_vs_uniform']['statistic']:.1f}"
              f" (df 12), p {out['runs'][tag]['chi2_vs_uniform']['p_value']:.2g}; pooled gain interval half-width "
              f"{half:.2f} points, 80%-power gain {out['runs'][tag]['gain_80pct_power']:.2f}")
    (HERE / "results" / "posthoc_letter_bias.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
