"""Setup checks on two real smoke prompts (i2t, t2i): letter tokens, token counts, latency, permutation sanity.
Run after run_probe.py --n 4 --out results/smoke."""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).parent))
from run_probe import *                                                                         # noqa
from src.eval.mllm_reranker import LETTERS, QwenReranker, build_messages                        # noqa

out = Path(__file__).parent / "results/smoke"
partial = np.load(out / "probe_partial.npz")
s, p = partial["scores"], partial["perms"]
print("scores", s.shape, "NaN:", bool(np.isnan(s).any()), "perms valid:", bool((np.sort(p, -1) == np.arange(13)).all()),
      "perms differ per (cond,dir):", bool((p[0, 0] != p[0, 1]).any()))
z = np.load(out / "episodes_seed44.npz")
data = load_artelingo(); ann = json.load(open(ANNOTATIONS_PATH))
ep = type("E", (), {})()
a = {k.split("__", 2)[2]: z[k] for k in z.files if k.startswith("emotion__style__")}
from src.eval.aspect_episodes import AspectEpisodes
ep = AspectEpisodes("emotion", "style", **a)
path, cap = row_lookups(data, ann, ep.rows())
rr = QwenReranker(MODEL_ID, "cuda", MAX_PIXELS)
print("letter ids", rr.letter_ids, "single tokens:", all(len(rr.processor.tokenizer.encode(L, add_special_tokens=False)) == 1 for L in LETTERS))
for d in ("i2t", "t2i"):
    msgs = make_prompt(build_messages, ep, 0, "a", d, np.arange(13), path, cap)
    inp = rr.processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    grid = inp["image_grid_thw"]
    n_img_tok = int((grid.prod(1) // 4).sum())
    n_px_max = max(int(h) * 16 * int(w) * 16 for _, h, w in grid.tolist())
    rr.score(msgs); torch = __import__("torch"); torch.cuda.synchronize(); t0 = time.time()
    lg = rr.score(msgs); torch.cuda.synchronize()
    print(d, "images", len(grid), "total tokens", inp["input_ids"].shape[1], "image tokens", n_img_tok,
          "max px per image", n_px_max, "latency s", round(time.time() - t0, 2), "logits", np.round(lg, 1))
