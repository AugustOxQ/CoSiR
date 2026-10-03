"""fp32 letter scores vs the bf16 full-logit path on real prompts (both directions, 4 episodes)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3])); sys.path.insert(0, str(Path(__file__).parent))
from run_probe import *                                                                         # noqa
from src.eval.aspect_episodes import AspectEpisodes                                              # noqa
from src.eval.mllm_reranker import QwenReranker, build_messages                                  # noqa
z = np.load(Path(__file__).parent / "results/smoke_v2/episodes_seed44.npz")
ep = AspectEpisodes("emotion", "style", **{k.split("__", 2)[2]: z[k] for k in z.files if k.startswith("emotion__style__")})
data = load_artelingo(); path, cap = row_lookups(data, json.load(open(ANNOTATIONS_PATH)), ep.rows())
rr = QwenReranker(MODEL_ID, "cuda", MAX_PIXELS)
for i in range(4):
    for d in DIRECTIONS:
        m = make_prompt(build_messages, ep, i, "a", d, np.arange(13), path, cap)
        a, b = rr.score(m), rr.score_bf16(m)
        ties = len(np.unique(b)) < 13
        print(d, i, "bf16 ties:", ties, "argmax same:", a.argmax() == b.argmax(),
              "full ranking same:", (np.argsort(-a) == np.argsort(-b)).all(), "max abs diff", round(float(np.abs(a - b).max()), 3))
