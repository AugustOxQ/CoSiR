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
        # H<X> (rule §6.6, the CLIP-head arm) groups by X's labels; only its heads differ (ticket 08 --h-arm).
        source = {n: n[1:] if n.startswith("H") and n[1:] in CANDIDATES else n for n in extras}
        # NBC and ZNBC are collected after the trigger into their own file (ticket 07 `collect --name r8_groupings_nbc`).
        paths = [root / "groupings/r8_groupings.npz", root / "groupings/r8_groupings_nbc.npz"]
        for path in paths[:1] + [p for p in paths[1:] if p.is_file()]:
            with np.load(path, allow_pickle=False) as z:
                assert {"scorer_train", "g_st"} <= set(z.files), f"{path}: missing keys"
                assert np.array_equal(z["scorer_train"], context.scorer_train), f"{path}: scorer-train differs"
                assert np.array_equal(z["g_st"], context.g_st), f"{path}: painting codes differ"
                for name in extras:
                    if source[name] not in z.files:
                        continue
                    assert name not in out, f"{name}: labels in more than one grouping file"
                    local = z[source[name]]
                    assert local.dtype == np.int64 and local.shape == context.scorer_train.shape, f"{name}: labels"
                    assert np.array_equal(np.unique(local), np.arange(len(np.unique(local)))), f"{name}: classes"
                    assert len(np.unique(local)) >= 2, f"{name}: fewer than 2 groups"
                    assert np.array_equal(local[context.first_rows][context.g_st], local), f"{name}: painting alignment"
                    out[name] = local
        missing = [n for n in extras if n not in out]
        assert not missing, f"{root / 'groupings'}: no labels for {missing}"
    return out


def reader_inputs(name, root=None):
    """Shared SC1/ticket-09 reader boundary: return (labels, frozen halves, post).

    A0 labels use R7's frozen sources and A0 xheads always come from R7 results
    (rule §10, all readers). X labels/xheads come from root, default r8 RESULTS.
    SC1 passes name='R0', root=R7_RESULTS, using its r7_groupings.npz instead.
    No head JSON is needed. Pass all three returned objects to build_reader with
    post=post; the old post=None convenience path is not the r8 assembly path.
    """
    import r8_reader as Q

    assert name in GROUPINGS, f"unknown reader grouping {name!r}"
    root = Path(root) if root is not None else RESULTS
    context = R.ctx()
    labels = load_groupings(Q.BASE)
    if name == "R0":
        labels.update(R.load_r7("r7_heads").load_groupings(
            (name,), context.scorer_train, context.g_st, root=root))
        assert np.array_equal(np.unique(labels[name]), np.arange(17)), "R0: frozen 17 classes"
    else:
        labels.update(load_groupings((name,), root=root))
    keys = tuple(f"20261126_design_l_style_source/results/heads/xheads_{h}.npz" for h in Q.BASE)
    R.assert_inputs(keys=keys, r7=False)
    halves = context.halves_dict
    post = {h: Q.load_crossfit(h, labels[h], halves, R7_RESULTS) for h in Q.BASE}
    post[name] = Q.load_crossfit(name, labels[name], halves, root)
    return labels, halves, post
