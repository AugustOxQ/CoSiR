"""Aspect episodes (CVPR plan spec §5.1): the condition shows an aspect through value-disjoint cross-item example
pairs, contrasted with pairs that share another aspect. Generic over datasets: labels are per-row int codes, -1 =
unlabelled. Rows, groups and labels index the same global row space."""

import hashlib
from dataclasses import dataclass, fields

import numpy as np

from src.data.sampling import draw_distinct

NUM_PAIRS = 4
NUM_NEGATIVES = 11


@dataclass(frozen=True)
class AspectEpisodes:
    aspect_a: str
    aspect_b: str
    anchor: np.ndarray
    candidates: np.ndarray        # column 0 = p_a, column 1 = p_b, then 11 negatives
    pairs_a_img: np.ndarray       # example pair i of aspect a: the image of row pairs_a_img[:, i] ...
    pairs_a_txt: np.ndarray       # ... with the caption of row pairs_a_txt[:, i]
    pairs_b_img: np.ndarray
    pairs_b_txt: np.ndarray

    def condition(self, which: str):
        """(support_img, support_txt, contrast_img, contrast_txt, target_column) for condition 'a' or 'b'."""
        if which == "a":
            return self.pairs_a_img, self.pairs_a_txt, self.pairs_b_img, self.pairs_b_txt, 0
        if which == "b":
            return self.pairs_b_img, self.pairs_b_txt, self.pairs_a_img, self.pairs_a_txt, 1
        raise ValueError(f"condition must be 'a' or 'b', got {which!r}")

    def rows(self) -> np.ndarray:
        return np.concatenate([self.anchor.ravel(), self.candidates.ravel(), self.pairs_a_img.ravel(),
                               self.pairs_a_txt.ravel(), self.pairs_b_img.ravel(), self.pairs_b_txt.ravel()])


class PaintingValueIndex:
    """``lacks(aspect, value)``: True for rows whose painting (group) has no row labelled ``value`` on ``aspect``."""

    def __init__(self, labels: dict, groups: np.ndarray) -> None:
        self.labels = {k: np.asarray(v, dtype=np.int64) for k, v in labels.items()}
        self.groups = np.asarray(groups)
        self._cache: dict = {}

    def lacks(self, aspect: str, value: int) -> np.ndarray:
        key = (aspect, int(value))
        if key not in self._cache:
            having = np.unique(self.groups[self.labels[aspect] == value])
            mask = ~np.isin(self.groups, having)
            mask.flags.writeable = False
            self._cache[key] = mask
        return self._cache[key]


def eligible_values(labels_a: np.ndarray, groups: np.ndarray, rows: np.ndarray, min_paintings: int) -> list:
    rows = np.asarray(rows, dtype=np.int64)
    out = []
    for value in np.unique(labels_a[rows]):
        if value < 0:
            continue
        if len(np.unique(groups[rows[labels_a[rows] == value]])) >= min_paintings:
            out.append(int(value))
    return out


def _pairs(rng, base, share, differ, values, groups, used):
    """One cross-item pair per value: both rows share ``value`` on ``share`` and differ on ``differ``."""
    img, txt = [], []
    for value in values:
        pool = base[share[base] == value]
        x = draw_distinct(rng, pool, groups, used, 1)[0]
        y = draw_distinct(rng, pool[differ[pool] != differ[x]], groups, used, 1)[0]
        img.append(x)
        txt.append(y)
    return img, txt


def build_aspect_episodes(labels: dict, groups: np.ndarray, rows: np.ndarray, aspect_a: str, aspect_b: str,
                          n_episodes: int, seed: int, third: str | None = None, min_paintings: int = 30,
                          index: PaintingValueIndex | None = None) -> AspectEpisodes:
    groups = np.asarray(groups)
    index = index or PaintingValueIndex(labels, groups)
    la, lb = index.labels[aspect_a], index.labels[aspect_b]
    lt = index.labels[third] if third else None
    rows = np.asarray(rows, dtype=np.int64)
    known = (la[rows] >= 0) & (lb[rows] >= 0)
    if third:
        known &= lt[rows] >= 0
    pool = rows[known]
    ok_a = eligible_values(la, groups, pool, min_paintings)
    ok_b = eligible_values(lb, groups, pool, min_paintings)
    if len(ok_a) <= NUM_PAIRS or len(ok_b) <= NUM_PAIRS:
        raise RuntimeError(f"{aspect_a} x {aspect_b}: need more than {NUM_PAIRS} eligible values per aspect, "
                           f"got {len(ok_a)} and {len(ok_b)}")
    anchors = pool[np.isin(la[pool], ok_a) & np.isin(lb[pool], ok_b)]
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ("anchor", "candidates", "pairs_a_img", "pairs_a_txt", "pairs_b_img", "pairs_b_txt")}
    failures = 0
    while len(out["anchor"]) < n_episodes:
        anchor = int(anchors[rng.integers(len(anchors))])
        a, b = int(la[anchor]), int(lb[anchor])
        lack_a, lack_b = index.lacks(aspect_a, a)[pool], index.lacks(aspect_b, b)[pool]
        lack_t = index.lacks(third, int(lt[anchor]))[pool] if third else np.ones(len(pool), dtype=bool)
        used = {groups[anchor]}
        try:
            p_a = draw_distinct(rng, pool[(la[pool] == a) & lack_b & lack_t], groups, used, 1)
            p_b = draw_distinct(rng, pool[(lb[pool] == b) & lack_a & lack_t], groups, used, 1)
            negatives = draw_distinct(rng, pool[lack_a & lack_b & lack_t], groups, used, NUM_NEGATIVES)
            base = pool[lack_a & lack_b]
            va = rng.choice([v for v in ok_a if v != a], NUM_PAIRS, replace=False)
            vb = rng.choice([v for v in ok_b if v != b], NUM_PAIRS, replace=False)
            pa_img, pa_txt = _pairs(rng, base, la, lb, va, groups, used)
            pb_img, pb_txt = _pairs(rng, base, lb, la, vb, groups, used)
        except ValueError:
            failures += 1
            if failures > 10 * n_episodes:
                raise RuntimeError(f"{aspect_a} x {aspect_b}: could not fill {n_episodes} episodes "
                                   f"({failures} failed draws); the pools are too small for these constraints")
            continue
        for key, value in zip(out, (anchor, p_a + p_b + negatives, pa_img, pa_txt, pb_img, pb_txt)):
            out[key].append(value)
    return AspectEpisodes(aspect_a, aspect_b, *(np.asarray(out[k], dtype=np.int64) for k in out))


def validate_aspect_episodes(ep: AspectEpisodes, labels: dict, groups: np.ndarray, index: PaintingValueIndex,
                             third: str | None = None) -> None:
    """Assert every rule of spec §5.1 on the final arrays."""
    la, lb = index.labels[ep.aspect_a], index.labels[ep.aspect_b]
    lt = index.labels[third] if third else None
    for i in range(len(ep.anchor)):
        r = ep.anchor[i]
        a, b = la[r], lb[r]
        every = np.concatenate([[r], ep.candidates[i], ep.pairs_a_img[i], ep.pairs_a_txt[i], ep.pairs_b_img[i],
                                ep.pairs_b_txt[i]])
        assert len(np.unique(groups[every])) == len(every), f"episode {i}: a painting appears twice"
        assert (la[every] >= 0).all() and (lb[every] >= 0).all(), f"episode {i}: unlabelled row"
        p_a, p_b, neg = ep.candidates[i, 0], ep.candidates[i, 1], ep.candidates[i, 2:]
        assert la[p_a] == a and index.lacks(ep.aspect_b, b)[p_a], f"episode {i}: p_a"
        assert lb[p_b] == b and index.lacks(ep.aspect_a, a)[p_b], f"episode {i}: p_b"
        assert index.lacks(ep.aspect_a, a)[neg].all() and index.lacks(ep.aspect_b, b)[neg].all(), f"episode {i}: neg"
        if third:
            t = index.labels[third][r]
            assert t >= 0 and index.lacks(third, t)[ep.candidates[i]].all(), f"episode {i}: third aspect"
            assert (lt[every] >= 0).all(), f"episode {i}: unlabelled third aspect"
        for share, differ, xs, ys, own in ((la, lb, ep.pairs_a_img[i], ep.pairs_a_txt[i], a),
                                           (lb, la, ep.pairs_b_img[i], ep.pairs_b_txt[i], b)):
            assert (share[xs] == share[ys]).all() and (differ[xs] != differ[ys]).all(), f"episode {i}: pair"
            assert len(np.unique(share[xs])) == NUM_PAIRS and (share[xs] != own).all(), f"episode {i}: pair values"
        examples = np.concatenate([ep.pairs_a_img[i], ep.pairs_a_txt[i], ep.pairs_b_img[i], ep.pairs_b_txt[i]])
        assert index.lacks(ep.aspect_a, a)[examples].all() and index.lacks(ep.aspect_b, b)[examples].all(), \
            f"episode {i}: an example shows the anchor's value"


def episodes_sha256(ep: AspectEpisodes) -> str:
    digest = hashlib.sha256(f"{ep.aspect_a}|{ep.aspect_b}".encode())
    for f in fields(ep)[2:]:
        digest.update(np.ascontiguousarray(getattr(ep, f.name), dtype=np.int64).tobytes())
    return digest.hexdigest()


def concat_episodes(parts: list) -> AspectEpisodes:
    names = {(p.aspect_a, p.aspect_b) for p in parts}
    a, b = next(iter(names)) if len(names) == 1 else ("mixed", "mixed")
    return AspectEpisodes(a, b, *(np.concatenate([getattr(p, f.name) for p in parts]) for f in fields(parts[0])[2:]))
