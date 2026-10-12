"""R7 §9 Phases K/A/B: independent per-anchor arrays, R3 D3–D11, L11–L14.

Copied R3 re-derivation arithmetic lives in rd7_core/rd7_family. This runner
reads only frozen heads/readers/thresholds, stored probabilities, and the gate
verdict's picks. It never reads/imports a round-7 implementation or its arrays.

Run from RF: python rederive/rd7_arrays.py --phase K --seed 42
Chain smoke reads/writes RESULTS/smoke/chain. Rule smoke reads frozen objects
from RESULTS and per-seed probabilities from RESULTS/smoke/rule, writing there.
Output: rederive/phaseK.npz, phaseA_seed{s}.npz, phaseB_seed{s}.npz (then _fix<n>).
"""
import argparse
import json
import re
import sys
from pathlib import Path

# Support both the executable file and the namespace package used by tests.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import r7_common as R
from rederive import rd7_bundle as BU, rd7_core as K


PARTS = K.A0
VARIANTS = tuple(f"V{k}" for k in range(1, 7))


def check_phase_seed(phase, seed, chain_smoke=False):
    smoke = seed in (9001, 9002, 9003)
    dev, tests = ((9001, (9002, 9003)) if smoke else (42, (55, 56, 57)))
    if smoke and not chain_smoke:
        tests = (9001, 9002, 9003)  # rule smoke reruns stage 3 on frozen real objects
    assert phase in ("K", "A", "B"), "phase must be K, A or B"
    assert (phase in ("K", "A") and seed == dev) or (phase == "B" and seed in tests), "phase seed role"
    assert not chain_smoke or smoke, "phase seed role: chain smoke requires smoke seeds"
    return smoke


def phase_path(phase, seed):
    return Path("rederive") / ("phaseK.npz" if phase == "K" else f"phase{phase}_seed{seed}.npz")


def add_metrics(out, name, metrics):
    assert set(metrics) == set(K.METRICS), f"metric keys: {name}"
    for metric, v in metrics.items():
        assert v.ndim == 1 and v.dtype == np.float64 and np.isfinite(v).all(), f"per-anchor metric: {name}/{metric}"
        assert f"{name}__{metric}" not in out, f"duplicate scorer: {name}"
        out[f"{name}__{metric}"] = v


def add_family(out, name, scored):
    add_metrics(out, name, scored["fused"])
    add_metrics(out, name + "_cf", scored["cf"])
    rec = scored["record"]
    for short, key in (("fused_cells", "fused_cells"), ("cf_cells", "counterpart_cells")):
        out[f"{name}__{short}"] = np.array([rec[key][h]["cell"] for h in "01"], np.int64)
    out[f"{name}__sigma"] = np.array([rec["control"][h]["sigma_star"] for h in "01"], np.float64)
    for c in K.COND:
        out[f"{name}__gate_{c}"] = scored[f"gate_{c}"]


def thresholds(root, dev_seed=42):
    """L12: validate frozen taus against the stored development margins."""
    z = json.loads((Path(root) / "taus.json").read_text())
    assert isinstance(z, dict), "threshold table"
    assert z.get("rule_sha256", R.RULE_SHA256) == R.RULE_SHA256, "threshold rule SHA"
    out = {}
    for name, v in z.items():
        if name in ("rule_sha256", "written"):
            continue
        assert re.fullmatch(r"V[1-6]|Z[1-6]|R0|N", name), f"threshold grouping: {name}"
        tau = np.asarray(v, np.float64)
        assert tau.shape == (4,) and np.isfinite(tau).all() and np.all(np.diff(tau) >= 0), f"four ordered thresholds: {name}"
        assert np.all((tau >= 0) & (tau <= 1)), f"margin thresholds: {name}"
        reader = BU.checked_npz(Path(root) / f"seed{dev_seed}/reader_{name}.npz")
        assert {"m__a", "m__b"} <= set(reader), f"development reader margins: {name}"
        ma, mb = reader["m__a"], reader["m__b"]
        assert ma.ndim == 1 and ma.size > 0 and ma.shape == mb.shape, f"development margin rows: {name}"
        assert all(m.dtype == np.float64 and np.isfinite(m).all() and np.all((m >= 0) & (m <= 1))
                   for m in (ma, mb)), f"development margin values: {name}"
        assert np.array_equal(tau, K.thresholds(ma, mb)), f"frozen development thresholds: {name}"
        out[name] = tau
    return out


def picks(root):
    verdict = json.loads((Path(root) / "gate_verdict.json").read_text())
    assert {"picks", "n"} <= set(verdict), "gate verdict keys"
    assert verdict.get("rule_sha256", R.RULE_SHA256) == R.RULE_SHA256, "gate rule SHA"
    assert set(verdict["picks"]) == {"csd", "generic"}, "gate families"
    values = [v for v in verdict["picks"].values() if v is not None]
    assert len(values) == verdict["n"] and len(set(values)) == len(values), "picked family count"
    for family, name in verdict["picks"].items():
        if name is not None:
            assert re.fullmatch(r"(?:R1|XAFF)_V[1-6]", name), f"pick name: {name}"
            idx = int(name[-1])
            assert (idx <= 4) == (family == "csd"), "pick grouping family"
    return values


def derive(phase, b, root, names=None, reader_root=None, dev_seed=42):
    """The in-memory seam: full cross-fits use this seed's pooled episode parity."""
    reader_root = Path(root if reader_root is None else reader_root)
    out = {"cl": np.asarray(b.cl, np.int64), "pair_index": np.asarray(b.pair_index, np.int64),
           "parity": np.asarray(b.parity, np.int64)}
    assert all(v.shape == (b.ctx.n,) for v in out.values()), "episode ID shapes"
    if phase == "K":
        BU.load_post(b, root, "T")
        add_family(out, "told_T", BU.told(b, "T"))
        return out

    assert phase in ("A", "B"), "array phase"
    BU.load_post(b, root, "R0")
    for name, values in BU.baselines(b, "R0").items():
        add_metrics(out, name, values)
    # AFF keeps its frozen A0 reader and R3 thresholds, rerunning both cross-fits.
    reader = BU.read(b, PARTS, BU.legacy_reader("A0"), reader_root / f"seed{b.ctx.seed}/reader_A0.npz")
    add_family(out, "AFF", BU.score(b, reader, BU.aff_taus(), "AFF"))
    taus = thresholds(root, dev_seed)
    if phase == "A":
        names = [f"{kind}_{x}" for x in VARIANTS if x in taus for kind in ("R1", "XAFF")]
        # Absent thresholds denote a dropped grouping/reader (L7, L10, L11).
        # Refuse an incomplete build that still has a fitted reader.
        for x in VARIANTS:
            assert x in taus or not (Path(root) / f"readers/reader_{x}.pkl").exists(), f"missing thresholds for fitted reader: {x}"
    elif names is None:
        names = picks(root)
    by_group = {}
    for name in names:
        assert re.fullmatch(r"(?:R1|XAFF)_V[1-6]", name), f"candidate name: {name}"
        kind, x = name.split("_")
        assert x in taus, f"pick lacks frozen thresholds: {x}"
        if x not in by_group:
            BU.load_post(b, root, x)
            parts = (*PARTS, x)
            by_group[x] = BU.read(b, parts, BU.fitted_reader(root, x),
                                 reader_root / f"seed{b.ctx.seed}/reader_{x}.npz")
            add_metrics(out, f"BX_{x}", BU.condition_free(b, parts))
        add_family(out, name, BU.score(b, by_group[x], taus[x], kind))
    return out


def write_once(path, arrays):
    """Use the shared writer; a repeat keeps the original and writes _fix<n>."""
    path = Path(path)
    if path.exists():
        path = R.fix_path(path)
    return R.write_npz_once(path, **arrays)


def run(phase, seed, chain_smoke=False):
    smoke = check_phase_seed(phase, seed, chain_smoke)
    assert R.sha_file(R.RULE) == R.RULE_SHA256, "round-7 rule SHA"
    root = R.SMOKE_CHAIN if chain_smoke else R.RESULTS
    output = R.SMOKE_CHAIN if chain_smoke else (R.SMOKE_RULE if smoke else R.RESULTS)
    destination = output / phase_path(phase, seed)
    names = picks(root) if phase == "B" else None
    b = BU.build(seed, root, smoke)
    arrays = derive(phase, b, root, names, reader_root=output, dev_seed=9001 if chain_smoke else 42)
    destination = write_once(destination, arrays)
    print(f"phase {phase}: {b.ctx.n} episodes; {len(arrays)} arrays; {destination}")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--phase", choices=("K", "A", "B"), required=True)
    parser.add_argument("--chain-smoke", action="store_true")
    args = parser.parse_args()
    run(args.phase, args.seed, args.chain_smoke)


if __name__ == "__main__":
    main()
