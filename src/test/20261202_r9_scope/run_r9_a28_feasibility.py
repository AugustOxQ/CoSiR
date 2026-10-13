"""Feasibility of the ArtELingo-28 per-language episodes (ticket 17; rule section 5.1 items 7 and 8). Labels only.

Builds smoke seeds 9001 to 9003 (builder seeds 1000 * s + l) on the real read set, read rows before the empty-output
drops (item 7's setting), 4,096 episodes per pair, and prints: the set-aside languages, the shares (minimum and
maximum), the number of languages, the anchor paintings per pair, the worst failed-draw ratios (failed draws per
episode, counted by wrapping the builder's two drawing calls) and, for seed 9001, item 8's option-B row (share of
each emotion among the anchors' emotions, the third aspect included) and the hash guard. Reads
results/read_set_a28.parquet and results/value_sets_a28.json from the main checkout; writes nothing.

    PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= \
    /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/run_r9_a28_feasibility.py
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import r9_a28 as K  # noqa: E402
import r9_a28_episodes as A  # noqa: E402
import r9_common as R  # noqa: E402
import r9_episodes as E  # noqa: E402

RESULTS = R.MAIN / "src/test/20261202_r9_scope/results"
EMOTIONS = ("amusement", "anger", "awe", "contentment", "disgust", "excitement", "fear", "sadness")
RULE_B_ROW = {"emotion__style": [.115, .022, .112, .319, .047, .124, .101, .160],
              "emotion__genre": [.109, .022, .129, .269, .067, .138, .100, .165],
              "style__genre": [.100, .018, .116, .340, .053, .129, .092, .152]}
RULE_ANCHORS = (498, 711, 489)


class CountingBuilder:
    """Wraps E.build_aspect_episodes_r6, counting the ValueErrors of draw_distinct and _pairs per call (each failed
    draw raises exactly one)."""

    def __init__(self):
        self.records = []
        self.count = 0
        self.real = E.build_aspect_episodes_r6
        self.draw, self.pairs = E.draw_distinct, E._pairs

    def _wrap(self, fn):
        def inner(*a, **k):
            try:
                return fn(*a, **k)
            except ValueError:
                self.count += 1
                raise
        return inner

    def __call__(self, **kw):
        self.count = 0
        ep = self.real(**kw)
        self.records.append((kw["seed"] % 1000, f"{kw['aspect_a']}__{kw['aspect_b']}", kw["seed"] // 1000,
                             kw["n_episodes"], self.count))
        return ep

    def install(self):
        E.draw_distinct, E._pairs = self._wrap(self.draw), self._wrap(self.pairs)
        E.build_aspect_episodes_r6 = self


def main():
    table = pd.read_parquet(RESULTS / "read_set_a28.parquet")
    assert list(table.columns) == K.READ_COLUMNS and len(table) == K.EXPECTED["rows"], (list(table.columns), len(table))
    vrec = json.loads((RESULTS / "value_sets_a28.json").read_text())
    V = {a: [int(r["code"]) for r in vrec[a]] for a in R.ASPECTS}
    assert V == K.EXPECTED_V, V
    ctx = A.prepare(table, dropped=np.zeros(len(table), dtype=bool))          # read rows before the drops
    langs = ctx.languages
    assert len(langs) == 27, len(langs)
    counter = CountingBuilder()
    counter.install()
    results = {}
    print(f"languages {len(langs)}; pool sizes min {min(ctx.pool_size)} ({langs[int(np.argmin(ctx.pool_size))]}), "
          f"max {max(ctx.pool_size)} ({langs[int(np.argmax(ctx.pool_size))]}); rows {len(table)}")
    for s in R.SMOKE_SEEDS:
        t0 = time.time()
        res = A.build_seed_a28(ctx, V, s)
        results[s] = res
        print(f"\nseed {s}: built in {time.time() - t0:.1f} s")
        for p in R.PAIR_NAMES:
            sh = res.shares[p]
            lo, hi = min(sh, key=sh.get), max(sh, key=sh.get)
            anchors = len(np.unique(ctx.cl[res.per_pair[R.PAIR_NAMES.index(p)].anchor]))
            print(f"  {p}: languages {len(sh)} of {len(langs)}, set aside {res.set_aside[p]}, shares {sh[lo]} ({lo}) "
                  f"to {sh[hi]} ({hi}), sum {sum(sh.values())}, anchor paintings {anchors}")
    anchors9001 = tuple(len(np.unique(ctx.cl[ep.anchor])) for ep in results[9001].per_pair)
    print(f"\nanchor paintings on seed 9001: {anchors9001} (rule: {RULE_ANCHORS})")
    worst = sorted(((c / n, l, p, s, n, c) for l, p, s, n, c in counter.records), reverse=True)[:5]
    print("worst failed-draw ratios (failures per episode, language, pair, seed, share, failures; limit 10):")
    for ratio, l, p, s, n, c in worst:
        print(f"  {ratio:.2f}  {langs[l - 1]}  {p}  {s}  n={n}  failures={c}")
    print("\nitem 8, option B, seed 9001 (anchor emotion shares):")
    print("  " + " ".join(f"{e[:6]:>7}" for e in EMOTIONS))
    emo = ctx_labels(table, "emotion")
    for p, ep in zip(R.PAIR_NAMES, results[9001].per_pair):
        got = np.bincount(emo[ep.anchor], minlength=8) / len(ep.anchor)
        assert (emo[ep.anchor] >= 0).all()
        print("  " + " ".join(f"{x:7.3f}" for x in got) + f"   {p} (ours)")
        print("  " + " ".join(f"{x:7.3f}" for x in RULE_B_ROW[p]) + f"   {p} (rule)")
        print(f"  max abs difference {np.abs(got - np.array(RULE_B_ROW[p])).max():.4f}")
    new = {s: results[s].sha for s in results}
    seeds = [x for s in results for x in A.builder_seeds([s], len(langs))]
    print("\nper-pair hashes:", {s: {p: h[:12] for p, h in hs.items()} for s, hs in new.items()})
    # the smoke builder seeds are in the ledger's round 9 smoke row; the hashes must be new
    print("hash guard:", A.hash_guard(new, seeds))


def ctx_labels(table, a):
    return table[a].to_numpy(dtype=np.int64)


if __name__ == "__main__":
    main()
