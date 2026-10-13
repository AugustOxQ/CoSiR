"""Round 9 pins and cells (DECISION_RULE.md of this folder: section 3.1, section 5 item 4, section 6 item 3; ticket 06).

  gate_share(gates, parity, cells)  {gate_share, gate_open, gate_total}: the gates of AFF's chosen tau on the other
                                    parity, conditions a and b, both tune halves; integers, then one float64 division
  switched_off(cells)               true when AFF's cell has no agreement weight (F2.cell_values(c)[3] == 0) in both halves
  clip_cells()                      CLIP's cells record (what a recipe run writes for another backbone): the cells of
                                    r9_score.CELLS (equal to round 3's AFF_CELLS and RC_CELLS), sigma*, switched_off
                                    and the gate share from round 3's seed42_arrays.npz
  write_cells_clip(out)             writes it once (the file exists already: an overwrite raises FileExistsError)
  build_pins(backbone)              the record of pins_<backbone>.json: input SHA-256s, every part's path and SHA-256,
                                    the picks, the lambdas, the cells, the taus, the SHA-256s of the gate references
  write_pins(backbone, out=None)    writes it once; ticket 13 reuses it for the other backbones
  rel_path(path)                    a part's path relative to the checkout that holds it (this folder's files: the
                                    checkout of this folder; every other file: the main checkout)
  read_cells                        r9_picks.read_cells (the schema check), re-exported

    cd <checkout> && CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/CoSiR/bin/python \
        src/test/20261202_r9_scope/r9_pins.py --backbone clip_b32
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r9_common as R  # noqa: E402  (first, before anything that imports src)
import r9_backbone as BB  # noqa: E402
import r9_bundle as B  # noqa: E402
import r9_picks as P  # noqa: E402
import r9_score as S  # noqa: E402
from r9_picks import read_cells  # noqa: E402,F401  (re-exported: the tests call PINS.read_cells)

import numpy as np  # noqa: E402

R3, F2 = R.R3, R.F2
R3_ARRAYS_REL = "20261121_round3_affect_gate/results/seed42_arrays.npz"
TAUS_PER_CELL = F2.CELLS_PER_KAPPA // F2.N_TAU          # cell // 56 is the tau index
HALVES = (0, 1)


def _require(cond, msg, exc=AssertionError):
    if not cond:
        raise exc(msg)


# ---------------------------------------------------------------- the cells record

def gate_share(gates, parity, cells) -> dict:
    """The gates of AFF's chosen tau (cell // 56) of tune half h, on the episodes of parity 1 - h (hazard 7), for
    conditions a and b: {"gate_share": open / total (float64), "gate_open": int, "gate_total": int}."""
    parity = np.asarray(parity)
    _require(len(cells) == 2, f"gate_share: one cell per tune half, got {cells!r}")
    n_open = total = 0
    for h in HALVES:
        tau = int(cells[h]) // TAUS_PER_CELL
        for c in "ab":
            g = np.asarray(gates[c])[tau][parity == 1 - h]
            _require(bool(np.isin(g, (0, 1)).all()), f"gate {c} (tau {tau}) holds a value that is not 0 or 1")
            n_open += int((g == 1).sum())
            total += int(g.size)
    _require(total > 0, "gate_share: no gate counted")
    return {"gate_share": n_open / total, "gate_open": n_open, "gate_total": total}


def switched_off(cells) -> bool:
    """True when AFF has no agreement weight (lambda_a = 0, F2.cell_values(c)[3]) in both halves."""
    return bool(all(F2.cell_values(int(c))[3] == 0 for c in cells))


def clip_cells() -> dict:
    """CLIP's cells record. The cells come from r9_score.CELLS (asserted equal to round 3's AFF_CELLS / RC_CELLS and to
    the cells stored in seed42_arrays.npz); sigma* from round 3 (AFF and R1 agree); the gate share from the stored AFF
    gates."""
    aff, r1 = R3.AFF_CELLS, R3.RC_CELLS
    _require(S.CELLS["aff"]["fused"] == tuple(aff["fused"]) and S.CELLS["aff"]["cf"] == tuple(aff["cf"])
             and S.CELLS["r1"]["fused"] == tuple(r1["fused"]) and S.CELLS["r1"]["cf"] == tuple(r1["cf"]),
             "r9_score.CELLS differ from round 3's AFF_CELLS / RC_CELLS")
    _require(tuple(aff["sigma"]) == tuple(r1["sigma"]), "round 3's AFF and R1 sigma* differ")
    R.assert_input(R3_ARRAYS_REL)
    with np.load(R.INPUT_PATHS[R3_ARRAYS_REL]) as z:
        stored = {c: z[f"aff_gate__{c}"] for c in "ab"}
        parity = z["parity"]
        _require(tuple(int(x) for x in z["aff_fused_cells"]) == S.CELLS["aff"]["fused"]
                 and tuple(int(x) for x in z["aff_cf_cells"]) == S.CELLS["aff"]["cf"],
                 "seed42_arrays.npz holds other AFF cells than r9_score.CELLS")
    gs = gate_share(stored, parity, S.CELLS["aff"]["fused"])
    fused = S.CELLS["aff"]["fused"]
    return {"aff": list(fused), "cf": list(S.CELLS["aff"]["cf"]), "r1_fused": list(S.CELLS["r1"]["fused"]),
            "r1_counterpart": list(S.CELLS["r1"]["cf"]), "sigma_star": [float(x) for x in aff["sigma"]],
            "switched_off": switched_off(fused), **gs}


def _write_new(path, rec):
    path = Path(path)
    with open(path, "x") as f:                                 # exclusive: FileExistsError on an existing file
        f.write(json.dumps(rec, indent=1) + "\n")
    return path


def write_cells_clip(out=None) -> Path:
    out = Path(out or BB.parts("clip_b32")["cells_seed42"])
    rec = clip_cells()
    path = _write_new(out, rec)
    _require(read_cells(path) == rec, "the cells file written differs from the record")
    return path


# ---------------------------------------------------------------- pins_<backbone>.json

def rel_path(path) -> str:
    """The path relative to the checkout that holds it: this folder's files to this folder's checkout, every other file
    to the main checkout (untracked earlier-round files live only there)."""
    path = Path(path)
    base = R.CHECKOUT if path.is_relative_to(R.HERE) else R.MAIN
    return path.relative_to(base).as_posix()


def build_pins(backbone="clip_b32", *, part_paths=None, stage2_results=None, stage0b_path=None, dry_run=False) -> dict:
    """Every part must exist; new backbones also require ticket 13's passed stage
    2, unchanged producer records, passed picks, seed-42 record and sensitivity.
    The extra arguments serve the read-only CLIP replay dry run; default CLIP
    pins retain ticket 06's exact schema for the stage 0a regression.
    """
    import run_r9_stage0a as RS                                # the gate's reference files (imports this module)
    parts = BB.parts(backbone) if part_paths is None else part_paths
    _require(set(parts) == set(BB.CLIP_PARTS), "pins: incomplete part roles")
    missing = sorted(role for role, path in parts.items() if not Path(path).is_file())
    _require(not missing, f"{backbone}: parts missing: {missing}")
    evidence = {}
    if backbone != "clip_b32" or dry_run:
        from run_r9_stage2 import pins_evidence
        evidence = pins_evidence(backbone, parts, stage2_results, stage0b_path, dry_run=dry_run)
    if backbone == "clip_b32" and not dry_run:
        input_sha = R.assert_inputs(backbone)                  # the R6 set, 51 files, each at its pin
    elif dry_run:
        replacements = {BB.CLIP_PARTS[role]: parts[role] for role in BB.RECIPE_ROLES}
        input_sha = {rel: R.sha256_file(replacements.get(R.TEST / rel, R.TEST / rel)) for rel in R.INPUT_SHA256}
    else:
        input_sha = {rel: R._sha_cached(p) for rel, p in R.input_paths(backbone).items()}
    part_rec = {role: {"path": (str(Path(path).resolve()) if dry_run else rel_path(path)),
                       "sha256": R.sha256_file(path)} for role, path in parts.items()}
    picks = json.loads(Path(parts["picks_seed42"]).read_text())
    base = json.loads(Path(parts["baselines_seed42"]).read_text())
    P.parse_lambda_picks(base)                                 # keys, grid
    cells = read_cells(parts["cells_seed42"])
    taus = json.loads(Path(parts["rc_tau"]).read_text())["taus"]
    _require(len(taus) == 4 and all(type(t) is float for t in taus), f"rc_tau.json taus: {taus!r}")
    if backbone == "clip_b32":
        _require(tuple(taus) == tuple(R3.TAUS), "rc_tau.json taus differ from the rule's (R3 D6)")
    return {
        "backbone": backbone,
        "input_sha256": input_sha,
        "parts": part_rec,
        "picks": {**{name: picks[name] for name in P.NESTED}, "mean_r1": picks["mean_r1"],
                  "lambda": {name: base["scorers"][name]["lambda_picks"] for name in P.LAMBDA_SCORERS},
                  "cells": cells, "taus": taus},
        "gate_references": {name: sha for name, (_, sha) in RS.REFERENCES.items()},
        **evidence,
    }


def write_pins(backbone="clip_b32", out=None) -> Path:
    out = Path(out or R.pins_path(backbone))
    _require(not out.exists(), f"{out} exists", FileExistsError)   # before the (long) build
    return _write_new(out, build_pins(backbone))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backbone", default="clip_b32", choices=BB.BACKBONES)
    a = ap.parse_args(argv)
    if a.backbone == "clip_b32":                               # the other backbones' cells files are recipe outputs
        path = write_cells_clip()
        print(f"written {path} (SHA-256 {R.sha256_file(path)})")
    path = write_pins(a.backbone)
    print(f"written {path} (SHA-256 {R.sha256_file(path)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
