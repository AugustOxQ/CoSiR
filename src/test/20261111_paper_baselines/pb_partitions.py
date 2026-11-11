"""Lightweight r25 cluster counts and filenames, shared by both environments and the independent check."""
from pathlib import Path

import pb_paths as P

CLASS_COUNTS = {'emotion': 9, 'style': 27, 'genre': 10}
TARGET_KS = (P.K, *sorted(set(CLASS_COUNTS.values())))


def aspect_ks(aspect):
    return CLASS_COUNTS[aspect], P.K


def topic_path(folder, seed, k=P.K):
    # Preserve the already produced K=20 artifacts, including their sidecars.
    name = f'topics_seed{seed}.npz' if k == P.K else f'topics_k{k}_seed{seed}.npz'
    return Path(folder) / name
