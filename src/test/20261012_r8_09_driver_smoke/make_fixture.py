"""Synthetic smoke inputs: R7's R0 labels/posteriors/xheads stand in for grouping V100 and its control ZV100.
Writes only under $R8_RESULTS/smoke/gate (+ smoke dropped example: V67 listed as dropped)."""
import json, shutil, numpy as np
from pathlib import Path
R7 = Path("/project/CoSiR/src/test/20261126_design_l_style_source/results")
root = Path(__import__("os").environ["R8_RESULTS"]) / "smoke/gate"
(root / "groupings").mkdir(parents=True, exist_ok=True); (root / "heads").mkdir(exist_ok=True)
z = np.load(R7 / "groupings/r7_groupings.npz")
lab = {k: z[k] for k in ("scorer_train", "g_st")}
for x in ("V100", "ZV100"):
    lab[x] = z["R0"]
    shutil.copy(R7 / "heads/post_R0.npz", root / f"heads/post_{x}.npz")
    shutil.copy(R7 / "heads/xheads_R0.npz", root / f"heads/xheads_{x}.npz")
np.savez(root / "groupings/r8_groupings.npz", **lab)
(root / "groupings/r8_groupings.json").write_text(json.dumps({"groupings": {"V100": {"K": 17}, "ZV100": {"K": 17}},
    "dropped": {"V67": "fixture drop"}}))
shutil.copy(Path(__import__("os").environ["R8_RESULTS"]) / "smoke/stage0/stage0_regression.json", root / "stage0_regression.json")
