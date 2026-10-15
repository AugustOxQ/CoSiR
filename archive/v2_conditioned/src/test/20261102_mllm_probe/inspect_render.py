"""CPU only (processor, no weights): decode one real rendered prompt per direction around the candidate block and print
the token ids of each candidate label. Reads the episodes of a smoke folder."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).parent))
from run_probe import *                                                                         # noqa
from src.eval.aspect_episodes import AspectEpisodes                                              # noqa
from src.eval.mllm_reranker import LETTERS, build_messages                                       # noqa
from transformers import AutoProcessor

folder = Path(sys.argv[1])
z = np.load(folder / "episodes_seed44.npz")
ep = AspectEpisodes("emotion", "style", **{k.split("__", 2)[2]: z[k] for k in z.files if k.startswith("emotion__style__")})
data = load_artelingo(); ann = json.load(open(ANNOTATIONS_PATH))
path, cap = row_lookups(data, ann, ep.rows())
pr = AutoProcessor.from_pretrained(MODEL_ID, max_pixels=MAX_PIXELS)
tok = pr.tokenizer
for d in DIRECTIONS:
    msgs = make_prompt(build_messages, ep, 0, "a", d, np.arange(13), path, cap)
    ids = pr.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")["input_ids"][0].tolist()
    text = tok.decode(ids)
    i = text.index("Candidates:")
    print(f"===== {d}: tokens {len(ids)}\n--- instruction start:\n{text[text.index('Each'):text.index('Each')+420]}")
    print(f"--- around the candidate block:\n{text[i-40:i+700]}\n...\n{text[-160:]}")
    clean = []
    for k, L in enumerate(LETTERS):
        pos = [j for j in range(len(ids)) if ids[j] == 32 + k and tok.decode(ids[j - 1:j + 1]).endswith(f"\n{L}") ] if False else None
    # label ids: for each letter, the token at the start of "\n{L}. " in the sequence
    found = {}
    for j in range(1, len(ids) - 1):
        for k, L in enumerate(LETTERS):
            if ids[j] == 32 + k and tok.decode([ids[j + 1]]).startswith(".") and "\n" in tok.decode([ids[j - 1]]):
                found.setdefault(L, (tok.decode([ids[j - 1]]), ids[j - 1], ids[j], tok.decode([ids[j + 1]])))
    print("label (prev token, prev id, letter id, next token):", found)
    print("all 13 labels clean:", len(found) == 13 and all(v[2] == 32 + k for k, v in enumerate(found.values())))
