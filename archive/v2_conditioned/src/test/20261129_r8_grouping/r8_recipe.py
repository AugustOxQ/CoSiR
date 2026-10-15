"""Shared M3/M14 helpers used by SC1 and tickets 07/08. CPU; no writes or joins.

fit_clip_heads returns (global standard posteriors, local cross-fitted posteriors,
provenance) in R7's format. Ticket 08 takes txt from these and replaces img by M11/12.
recipe_graph builds k=20 once; recipe_at_resolution uses it at each M3 resolution.
Callers get painting order, g_st and halves from r8_common.
"""
import numpy as np
import r8_common as R


def fit_clip_heads(data, local, scorer_train, selection, halves, *, smoke=False,
                   head_rows=None, check_rows=None):
    """Refit L10's full image/text heads and both L11 cross-fits with R7's exact recipe.

    Real sizes are fixed at 60,000 draw / 10,000 check rows; smoke defaults 5,000/1,000.
    Neither stored heads nor a stored model is used, including for the SC1d R0 refit.
    """
    R.assert_rule()
    R.assert_threads()
    H = R.load_r7("r7_heads")
    head_rows = (H.SMOKE_HEAD_ROWS if smoke else H.HEAD_ROWS) if head_rows is None else head_rows
    check_rows = (H.SMOKE_CHECK_ROWS if smoke else H.CHECK_ROWS) if check_rows is None else check_rows
    assert smoke or (head_rows, check_rows) == (H.HEAD_ROWS, H.CHECK_ROWS), "real head sizes are fixed"
    assert head_rows > 0 and check_rows > 0, "positive head sizes required"
    st, selection = np.asarray(scorer_train), np.asarray(selection)
    n = len(data.img_features)
    H._rows(st, n, "scorer_train")
    H._rows(selection, n, "selection")
    assert not np.intersect1d(st, selection).size, "selection overlaps scorer-train"
    local, n_cls = H._labels(local, st)
    H._cross_rows(halves, st, n)
    pools = {"standard": st, **{f"half{h}": halves[f"global_rows_half{h}"] for h in (0, 1)}}
    draws = {}
    for key, pool in pools.items():
        assert len(pool) >= head_rows + check_rows, f"{key}: insufficient draw/check rows"
        draws[key] = H._draw_record(local, st, pool, head_rows)
    assert all(v["all_groups_in_draw"] for v in draws.values()), "missing groups in standard/cross-fitted draw (L10/L11)"
    _, post, standard = H.fit_standard(data, local, st, selection, head_rows, check_rows)
    cross, half_records = H.fit_crossfit(data, local, st, halves, head_rows, check_rows)
    assert post["txt"].shape == (n, n_cls) and cross["txt"].shape == (len(st), n_cls), "head output shapes"
    return post, cross, {"n_groups": n_cls, "draws": draws, "standard": standard,
                         "halves": half_records, "head_rows": head_rows, "check_rows": check_rows}


def _unit_phi(phi):
    phi = np.asarray(phi)
    assert phi.dtype == np.float32 and phi.ndim == 2 and len(phi) > 20, "phi: float32 painting matrix, >20 paintings"
    assert np.isfinite(phi).all() and np.allclose(np.linalg.norm(phi, axis=1), 1, rtol=0, atol=1e-6), "phi: unit/finite"
    return phi


def recipe_graph(phi):
    """M3 k=20 union graph, built once per source from unit float32 painting vectors."""
    phi = _unit_phi(phi)
    rs = R.load_r7("r7_common").load("run_step1")
    return rs.rsw.knn_union_graph(phi, 20)


def recipe_at_resolution(phi, g_st, graph, *, resolution=1.0, seed=42, min_rows=200):
    """M3 one resolution: (row labels, {r, K, seed, merge_moves}).

    A SystemExit from merge_small is deliberately propagated: ticket 07 records
    undefined K and skips that resolution. The graph is reused without mutation.
    """
    import leidenalg

    phi = _unit_phi(phi)
    g_st = np.asarray(g_st)
    assert g_st.dtype == np.int64 and g_st.ndim == 1, "g_st: local int64 painting codes"
    assert np.array_equal(np.unique(g_st), np.arange(len(phi))), "phi/g_st: painting alignment"
    assert graph.vcount() == len(phi), "graph: painting count"
    assert np.isfinite(resolution) and resolution > 0 and min_rows > 0, "invalid resolution/merge size"
    partition = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition,
                                         resolution_parameter=float(resolution), seed=int(seed))
    raw = np.unique(np.asarray(partition.membership, dtype=np.int64), return_inverse=True)[1].astype(np.int64)
    rs = R.load_r7("r7_common").load("run_step1")
    local, moves = rs.rto.merge_small(raw[g_st], phi[g_st], min_rows)
    assert local.dtype == np.int64 and local.shape == g_st.shape, "merged row labels"
    k = len(np.unique(local))
    assert np.array_equal(np.unique(local), np.arange(k)), "merged labels: compact classes"
    return local, {"r": float(resolution), "K": k, "seed": int(seed), "merge_moves": moves}
