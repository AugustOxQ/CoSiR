"""E0 check: the new aspect modules reproduce the aspect spike's CLIP and SE numbers on the spike's episodes."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.eval.aspect_episodes import AspectEpisodes  # noqa: E402
from src.eval.aspect_metrics import per_anchor  # noqa: E402
from src.eval.aspect_scorers import EvalInputs, cosine_scores, fixed_beta_scores  # noqa: E402

spec = importlib.util.spec_from_file_location("ra", ROOT / "src/test/20261018_affect_factor_learning/run_affect.py")
ra = importlib.util.module_from_spec(spec); spec.loader.exec_module(ra)
data = ra.load_artelingo(); cache, *_ = ra.grid.load_grid()
prep_record = json.loads((ra.CACHE / "affect_prepare.json").read_text())
z = np.load(ROOT / "src/test/20261023_aspect_episode_spike/results/aspect_episodes.npz")
# spike layout: cands = [p_emo, p_style, negs]; condition emotion = supports P_emo; spike pairs: pe_x image, pe_y caption
ep = AspectEpisodes("emotion", "style", z["anchor"], z["cands"], z["pe_x"], z["pe_y"], z["ps_x"], z["ps_y"])
sl = cache["selection"]
img = ra.sel.masked(data.img_features, sl); txt = ra.sel.masked(data.txt_features, sl)
se_ic, se_tc, _ = ra.model_codes("SE", data, cache, prep_record)
clip = per_anchor(cosine_scores(EvalInputs(img, txt), ep))
se = per_anchor(fixed_beta_scores(EvalInputs(img, txt, se_ic, se_tc), ep, 0.3))
got = {"clip_r1": 100 * clip["r1"].mean(), "se_b03_r1": 100 * se["r1"].mean(), "se_b03_swap": 100 * se["swap"].mean()}
print(json.dumps(got, indent=1))
want = {"clip_r1": 11.13, "se_b03_r1": 10.95, "se_b03_swap": 16.25}
assert all(abs(got[k] - want[k]) < 0.01 for k in want), (got, want)
print("REPRODUCED")
