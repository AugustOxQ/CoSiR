"""Final review B1: stage-1 spot-check records against numpy.random.default_rng(0) draws (no re-encoding).
Writes fr_b1_spot.json beside this file."""
import json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
REC = Path("/project/CoSiR/src/test/20261202_r9_scope/results")
STORE = Path("/data/SSD2/pre_extract/artelingo28")
out = {}
for b in ("clip_b32", "qwen3vl_emb_2b", "siglip2_so400m"):
    r = json.loads((REC / f"stage1_checks_{b}.json").read_text())
    idx = json.loads((STORE / b / "index.json").read_text())
    rng = np.random.default_rng(0)
    ir = rng.choice(idx["n_images"], 20, replace=False).tolist()
    tr = rng.choice(idx["n_texts"], 20, replace=False).tolist()
    sc = r["spot_check"]
    out[b] = {"image_rows_eq_rng0": [e["row"] for e in sc["images"]] == ir,
              "text_rows_eq_rng0": [e["row"] for e in sc["texts"]] == tr,
              "image_painting_eq_index": all(e["painting"] == idx["images"][e["row"]]["painting"] for e in sc["images"]),
              "text_row_id_eq_index": all(e["row_id"] == idx["texts"][e["row"]]["row_id"] for e in sc["texts"]),
              "min_image_cos": min(e["cosine"] for e in sc["images"]), "min_text_cos": min(e["cosine"] for e in sc["texts"]),
              "n": [len(sc["images"]), len(sc["texts"])], "device": sc["device"], "seed": sc["seed"],
              "flags": [r.get("row_id_join"), r.get("painting_join_ok"), r["finite_norm_gate"].get("ok")]}
(HERE / "fr_b1_spot.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out))
