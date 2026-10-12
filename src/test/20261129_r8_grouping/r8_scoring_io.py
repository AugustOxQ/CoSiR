"""Path/rule/ID adapter for the otherwise unchanged R7 bundle and reader copies.

Earlier-round modules are read-only and loaded through r8_common's verified loader.
This adapter never calls an earlier round's writer or recomputes its shared joins.
"""
from pathlib import Path

import numpy as np
import r8_common as R

MAIN, TEST, RESULTS, SMOKE_CHAIN = R.MAIN, R.MAIN / "src/test", R.RESULTS, R.SMOKE
R7_RESULTS = TEST / "20261126_design_l_style_source/results"
RULE_SHA256 = R.RULE_SHA256
assert_rule, assert_threads = R.assert_rule, R.assert_threads
sha_file, write_json_once = R.sha_file, R.write_json_once
write_npz_once, write_pickle_once = R.write_npz_once, R.write_pickle_once

CANDIDATES = ("V100", "V67", "V33", "NB100", "NB67", "NB33") + tuple(
    f"FC{k}" for k in range(1, 7)) + ("NBC", "AL", "DI", "SG")
REFERENCES = ("R0own", "NRK", "TP")
# rule gap: §2 does not list legacy R0, but §4 requires it through these same copies.
# Retain only R0 from the legacy whitelist, specifically for SC1.
GROUPINGS = CANDIDATES + REFERENCES + tuple(f"Z{x}" for x in CANDIDATES) + tuple(
    f"H{x}" for x in CANDIDATES if x not in ("AL", "DI", "SG")) + ("R0",)


def load(name):
    return R.load_r7("r7_common").load(name)


def rule_sha():
    return R.assert_rule()


def check_seed(seed, smoke):
    # This ticket exposes only the development and wiring-smoke path (rule §9).
    allowed = R.SMOKE_SEEDS if smoke else (R.DEV_SEED,)
    assert seed in allowed, f"seed {seed} (smoke {smoke}) is not in {allowed}"
    return seed


def reader_root(root, smoke):
    root = Path(root) if root is not None else (SMOKE_CHAIN if smoke else RESULTS)
    assert not smoke or R.in_smoke(root), "smoke reader outputs must be under results/smoke"
    return root


def load_groupings(names, root=None):
    """Aligned reader labels: R7's frozen A0 sources plus the r8 grouping file."""
    context = R.ctx()
    names = tuple(names)
    base = tuple(n for n in names if n in ("affect", "image", "caption"))
    heads = R.load_r7("r7_heads")
    out = heads.load_groupings(base, context.scorer_train, context.g_st)
    root = Path(root) if root is not None else RESULTS
    extras = tuple(n for n in names if n not in base)
    if extras:
        assert set(extras) <= set(GROUPINGS), "unknown grouping labels"
        path = root / "groupings/r8_groupings.npz"
        with np.load(path, allow_pickle=False) as z:
            assert {"scorer_train", "g_st", *extras} <= set(z.files), f"{path}: missing keys"
            assert np.array_equal(z["scorer_train"], context.scorer_train), f"{path}: scorer-train differs"
            assert np.array_equal(z["g_st"], context.g_st), f"{path}: painting codes differ"
            for name in extras:
                local = z[name]
                assert local.dtype == np.int64 and local.shape == context.scorer_train.shape, f"{name}: labels"
                assert np.array_equal(np.unique(local), np.arange(len(np.unique(local)))), f"{name}: classes"
                assert len(np.unique(local)) >= 2, f"{name}: fewer than 2 groups"
                assert np.array_equal(local[context.first_rows][context.g_st], local), f"{name}: painting alignment"
                out[name] = local
    return out
