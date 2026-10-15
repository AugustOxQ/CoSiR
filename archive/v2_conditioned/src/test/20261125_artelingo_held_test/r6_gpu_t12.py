"""Round 6 GPU job input loaders of ticket 12 (contracts section 9): rerank_input.npz and ft_rows.npz, read by the
reranker, the FT feature job and the CPU builders alike. They sit beside r6_gpu_common (ticket 10's file, which is not
edited) and use its Manifest, constants and checks. Imports neither src nor r6_common.

Guards carry a `# guard:<name>` marker; the tests delete each on a copy and show that its scenario then passes.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r6_gpu_common as G  # noqa: E402

RERANK_KEYS = ("seed", "episode_index", "query_row", *G.PAIR_FIELDS, "cand_shown")
NUM_CANDIDATES = 13


def load_rerank_input(path, manifest=None) -> dict:
    """rerank_input.npz: seed, episode_index (n,), query_row (n,), pairs_{a,b}_{img,txt} (n, 4), cand_shown
    (n, 2 cond, 2 dir, 13): all int64 row ids; the candidates are already in the order the model sees them. No other
    key is accepted (so no unpermuted candidate array, anchor name, target or label can ride along)."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        G._require(sorted(z.files) == sorted(RERANK_KEYS),
                   f"{path}: keys {sorted(z.files)} differ from {sorted(RERANK_KEYS)}")  # guard:rerank_keys
        d = {k: z[k] for k in RERANK_KEYS}
    n = len(d["episode_index"])
    G._require(all(d[k].dtype == np.int64 for k in RERANK_KEYS), f"{path}: every array must be int64")
    G._require(d["seed"].shape == () and d["episode_index"].shape == (n,) and n > 0, f"{path}: seed or index shape")
    G._require(d["query_row"].shape == (n,) and all(d[k].shape == (n, G.NUM_PAIRS) for k in G.PAIR_FIELDS),
               f"{path}: query_row must be (n,) and the pair arrays (n, 4)")
    G._require(d["cand_shown"].shape == (n, len(G.CONDITIONS), 2, NUM_CANDIDATES),
               f"{path}: cand_shown must be (n, 2, 2, 13)")
    G._require(bool((np.diff(d["episode_index"]) > 0).all()) and int(d["episode_index"][0]) >= 0,
               f"{path}: episode_index must be increasing and non-negative")
    if manifest is not None:
        for k in ("query_row", *G.PAIR_FIELDS, "cand_shown"):
            manifest.index(d[k].ravel())
    d["seed"] = int(d["seed"])
    return d


def load_ft_rows(path, manifest=None) -> np.ndarray:
    """ft_rows.npz: `rows` only (int64, sorted unique); every row must be in the manifest."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        G._require(z.files == ["rows"], f"{path}: keys {z.files} differ from ['rows']")  # guard:ft_rows_keys
        rows = z["rows"]
    G._require(rows.dtype == np.int64 and rows.ndim == 1 and rows.size > 0 and bool((np.diff(rows) > 0).all()),
               f"{path}: rows must be a non-empty sorted unique int64 vector")
    if manifest is not None:
        manifest.index(rows)
    return rows
