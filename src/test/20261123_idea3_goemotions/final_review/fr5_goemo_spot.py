"""Final review of round 5: a third CPU spot check of the GoEmotions file (rule D2, D3 tolerance 1e-4), on a sample
the implementation and the re-derivation did not use (rng 11, 512 selection rows), plus a misalignment control (the
same captions scored against the next row's stored probabilities). Captions joined as rule D2 joins them, from the
file's own sample_ids, which fr5_derive.py checked equal ctx.data.sample_ids[ctx.selection]. Imports only
src.data.affect and src.data.artelingo. CPU only; writes out/fr5_goemo_spot.json.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path("/project/CoSiR")
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

from src.data.affect import goemotions_probabilities, load_goemotions  # noqa: E402
from src.data.artelingo import ANNOTATIONS_PATH, join_captions  # noqa: E402

HERE = ROOT / "src/test/20261123_idea3_goemotions"
OUT = HERE / "final_review/out"
assert os.environ.get("CUDA_VISIBLE_DEVICES", None) == ""
assert "COSIR_ARTELINGO_ANNOTATIONS" not in os.environ
assert str(ANNOTATIONS_PATH) == "/data/PDD/artelingo/artelingo_train.json"
f = HERE / "cache/r5_goemotions_selection.npz"
assert hashlib.sha256(f.read_bytes()).hexdigest() == "f8372a89a808421772e19cce9413dbab9b83dcef5cfd28da8232d77d729e18a8"
z = np.load(f)
probs, sids = z["probs"], z["sample_ids"]
pos = np.sort(np.random.default_rng(11).choice(len(sids), size=512, replace=False))
t0 = time.time()
ann = json.loads(Path(ANNOTATIONS_PATH).read_text())
caps = join_captions(sids[pos], ann)
loaded = load_goemotions(device="cpu")
assert str(next(loaded[1].parameters()).device) == "cpu"
rerun = goemotions_probabilities(list(caps), loaded=loaded, batch_size=256, max_length=64)
diff = np.abs(rerun.astype(np.float64) - probs[pos].astype(np.float64))
pos_shift = np.clip(pos + 1, 0, len(sids) - 1)
shift = np.abs(rerun.astype(np.float64) - probs[pos_shift].astype(np.float64))
rec = {"n": int(len(pos)), "rng": 11, "max_abs": float(diff.max()), "mean_abs": float(diff.mean()),
       "n_above_1e-5": int((diff > 1e-5).sum()), "passed": bool(diff.max() <= 1e-4),
       "shifted_by_one_max_abs": float(shift.max()), "shifted_fails": bool(shift.max() > 1e-4),
       "runtime_s": round(time.time() - t0)}
(OUT / "fr5_goemo_spot.json").write_text(json.dumps(rec, indent=1))
print(rec)
