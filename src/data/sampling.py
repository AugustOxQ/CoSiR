"""Sampling helpers shared by label episodes and condition episodes."""

import numpy as np


def draw_distinct(rng, pool, keys, used, count):
    """Draw `count` rows from `pool` whose keys (leakage groups / paintings) are not yet in `used`.

    Updates `used`. Rejection sampling first, then an exact fallback over one row per unused key.
    Raises ValueError when `pool` cannot supply `count` distinct keys.
    """
    if count > 0 and len(pool) == 0:
        raise ValueError("Not enough distinct paintings to fill an episode")
    picked = []
    for _ in range(50 * count):
        if len(picked) == count:
            break
        row = int(pool[rng.integers(len(pool))])
        if keys[row] not in used:
            used.add(keys[row])
            picked.append(row)
    if len(picked) < count:
        eligible = pool[~np.isin(keys[pool], list(used))]
        _, first = np.unique(keys[eligible], return_index=True)
        eligible = eligible[np.sort(first)]
        if len(eligible) < count - len(picked):
            raise ValueError("Not enough distinct paintings to fill an episode")
        for row in rng.choice(eligible, count - len(picked), replace=False):
            used.add(keys[row])
            picked.append(int(row))
    return picked
