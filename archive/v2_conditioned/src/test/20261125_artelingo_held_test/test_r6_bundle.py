"""Real-data tests of r6_context and r6_bundle on seed 42, selection rows (ticket 05; rule section 5 item 5, section 6
items 1 and 4, section 11). The synthetic held-mode tests and the guard mutations are in test_r6_context.py.

The bundle comes from the round-6 path: the refit heads (r6_heads), the selection-mode RowContext with the episodes
built in process (r6_episodes.build_seed), fit_pm, build_bundle_r6. The reference is round 3's own computation on seed
42 through round 3's modules, with the stored posteriors where they exist (allowed on selection rows):
run_gonogo.EvalContext(42) on the stored episodes, run_checks.model_inputs, run_n6.load_posteriors + n6_terms,
run_told_oracle.fit_one_head for the D1 affect heads (R3 rule D2), common.grouping_stack, rb_eval.seed42_features; and
round 4's A1 term with step1_heads_style.npz's stored csd posteriors (run_step1.full_post's pattern). PM: the frozen
seed-42 lambda picks of baselines_seed42.json applied to the bundle's terms reproduce per_anchor_seed42.npz exactly.
Selection rows only; nothing of a held row is read. Prints no metric (assertion messages on failure aside).

About 8 to 10 minutes on CPU (head refits about 2 minutes, PM terms about 3, round 3's affect fit under 1):

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider \
        src/test/20261125_artelingo_held_test/test_r6_bundle.py
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r6_common as R  # noqa: E402  (first, before anything that imports src)
import r6_bundle as B  # noqa: E402
import r6_context as X  # noqa: E402
import r6_heads as H  # noqa: E402

import numpy as np  # noqa: E402

from src.data.artelingo import load_artelingo  # noqa: E402
from src.data.artelingo_splits import artelingo_aspect_labels, artelingo_splits  # noqa: E402
from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, METRICS, per_anchor  # noqa: E402
from src.eval.aspect_quick_checks import centered_term, crossfit_condition_free, uniform_probe_scores  # noqa: E402
from src.eval.aspect_scorers import fused_scores  # noqa: E402

C, rg, rc, n6, rto, rbe = R.C, R.rg, R.run_checks, R.run_n6, R.rto, R.rbe
PER_ANCHOR42_REL = "20261030_aspect_baselines/results/per_anchor_seed42.npz"
# rule section 6 item 3: the mean R@1 (x100) of the cross-fitted B, B'(A0), B'(A1) on seed 42
MEAN_R1 = {"t6u_B": 18.341064453125, "t6u_B0": 18.436686197916664, "t6u_B1": 18.804931640625}


def bits(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def scores_bits(s, t) -> bool:
    return all(bits(s[c][d], t[c][d]) for c in CONDITIONS for d in DIRECTIONS)


@pytest.fixture(scope="module")
def real():
    data = load_artelingo()
    split = R.load_split(data)
    labels = artelingo_aspect_labels(data)
    vs = R.development_value_sets(labels, split.groups, split.selection)
    heads = H.fit_heads_r6(data, split.groups, split.scorer_train)
    ctx = X.RowContext("selection", R.DEV_SEED, data, split, labels, heads, vs, R.N_PER_PAIR)
    pm = B.fit_pm(data, split.scorer_train)
    bundle = B.build_bundle_r6(ctx, B.load_readers(), pm)
    return SimpleNamespace(data=data, split=split, ctx=ctx, pm=pm, bundle=bundle)


@pytest.fixture(scope="module")
def ref(real):
    """Round 3's computation on seed 42 (r3_bundle.build_bundle's call sequence, rule section 4 item 1 of the R3 rule,
    without its seed guard and its cross-fits), on the same loaded data, through round 3's modules."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(rg, "load_artelingo", lambda: real.data)
        rctx = rg.EvalContext(R.DEV_SEED, False)
    ep = rctx.pooled
    scorer_train = np.asarray(artelingo_splits(rctx.data).scorer_train)
    inp = rc.model_inputs(rctx, "A3", scorer_train, False)[0]
    t_n1u = centered_term(inp, ep, uniform=True)
    del inp
    post_e2 = n6.load_posteriors(C.INPUT_FILES["n6_posteriors"], rctx)                       # the stored arrays
    t6u_B = n6.n6_terms(post_e2, ep)[2]
    with np.load(C.INPUT_FILES["affect_L_partition(per_anchor_told_oracle)"]) as z:
        partition_L = np.asarray(z["partition_L"], dtype=np.int64)
    post_aff, _ = rto.fit_one_head(rctx, rto.global_labels(partition_L, scorer_train, len(rctx.groups)),
                                   scorer_train, n6.HEAD_ROWS)                               # R3 rule D2
    post = {"affect": post_aff, "image": post_e2["image"], "caption": post_e2["caption"]}
    t6u_B0 = uniform_probe_scores(post, ep, B.A0)
    stack = C.grouping_stack(post, ep, B.A0)
    F, _ = rbe.seed42_features(SimpleNamespace(ctx=rctx, post=post), B.A0)
    # round 4's D2 and D4: csd from step1_heads_style.npz (run_step1.full_post), A1 order
    R.assert_input(H.CSD_HEADS_REL)
    csd = {}
    with np.load(R.INPUT_PATHS[H.CSD_HEADS_REL]) as z:
        assert np.array_equal(z["selection"], rctx.selection)
        for m in ("img", "txt"):
            a = np.asarray(z[f"style_csd__{m}"], dtype=np.float32)
            full = np.full((len(rctx.groups), a.shape[1]), np.nan, dtype=np.float32)
            full[rctx.selection] = a
            csd[m] = full
    t6u_B1 = uniform_probe_scores({**post, "csd": csd}, ep, B.A1)
    return SimpleNamespace(ctx=rctx, t_n1u=t_n1u, post_e2=post_e2, post=post, t6u_B=t6u_B, t6u_B0=t6u_B0,
                           t6u_B1=t6u_B1, stack=stack, F=F)


# ---------------------------------------------------------------- the context against EvalContext

def test_selection_context_equals_eval_context(real, ref):
    ctx, rctx = real.ctx, ref.ctx
    assert np.array_equal(ctx.rows, rctx.selection) and ctx.selection is ctx.rows
    assert np.array_equal(ctx.in_rows, rctx.in_sel)
    assert bits(ctx.img, rctx.img) and bits(ctx.txt, rctx.txt)
    for f in ("anchor", "candidates", "pairs_a_img", "pairs_a_txt", "pairs_b_img", "pairs_b_txt"):
        assert np.array_equal(getattr(ctx.pooled, f), getattr(rctx.pooled, f)), f
    assert ctx.episode_sha == rctx.shas == rctx.baselines["episodes_sha256"]
    assert ctx.n == rctx.n == 12_288 and ctx.n_per_pair == rctx.n_per_pair
    assert np.array_equal(ctx.pair_index, rctx.pair_index) and np.array_equal(ctx.parity, rctx.parity)
    assert np.array_equal(ctx.anchor_group, rctx.anchor_group)
    assert scores_bits(ctx.cos, rctx.cos)


def test_selection_context_codes_equal_eval_context(real, ref):
    a3 = R.TEST / B.A3_REL
    for mine, theirs in zip(real.ctx.encode(a3), ref.ctx.encode(a3)):
        assert bits(mine, theirs)


def test_stored_posteriors_load_on_the_selection_context_and_equal_the_refit(real, ref):
    """The tripwire admits the selection context (its rows are the stored selection); the refit E2 posteriors
    equal the stored ones bit for bit, NaN pattern included."""
    stored = n6.load_posteriors(C.INPUT_FILES["n6_posteriors"], real.ctx)
    for part, head in B.E2_FROM.items():
        for m in ("img", "txt"):
            assert bits(stored[part][m], real.ctx.post[head][m]), (part, m)


# ---------------------------------------------------------------- the bundle against round 3 and round 4

def test_bundle_fields_bit_identical_to_round3(real, ref):
    b, rctx = real.bundle, ref.ctx
    assert b.n == rctx.n and np.array_equal(b.cl, rctx.anchor_group)
    assert np.array_equal(b.parity, rctx.parity) and np.array_equal(b.pair_index, rctx.pair_index)
    assert np.array_equal(b.anchor, rctx.pooled.anchor)
    assert b.episodes_sha256 == rctx.shas
    assert scores_bits(b.cos, rctx.cos)
    assert scores_bits(b.t_n1u, ref.t_n1u)
    assert scores_bits(b.t6u_B, ref.t6u_B)                  # B's T_6u (affect-km, image, caption)
    assert scores_bits(b.t6u_B0, ref.t6u_B0)                # B0's T_6u (A0)
    assert set(b.post) == set(ref.post)
    for h in B.A0:
        for m in ("img", "txt"):
            assert bits(b.post[h][m], ref.post[h][m]), (h, m)
    for d in DIRECTIONS:
        assert bits(b.stack[d], ref.stack[d])
    for c in CONDITIONS:
        assert bits(b.F[c], ref.F[c])


def test_b1_term_equals_round4(real, ref):
    """B'(A1)'s T_6u from the refit csd head equals round 4's A1 term from step1_heads_style.npz."""
    assert scores_bits(real.bundle.t6u_B1, ref.t6u_B1)


def test_cross_fitted_means_hit_the_rule_targets(real):
    """Beyond the ticket (ticket 06 owns the picks): the three T_6u terms, cross-fitted with T_N1u as round 3 and
    round 4 did, give rule section 6 item 3's mean R@1 exactly."""
    b = real.bundle
    for name, want in MEAN_R1.items():
        s = crossfit_condition_free(b.cos, b.t_n1u, getattr(b, name), b.parity)[0]
        assert 100 * float(np.mean(per_anchor(s)["r1"])) == want, name


# ---------------------------------------------------------------- PM and RCA (rule section 11)

def test_fit_pm_sha_equals_baselines(real):
    R.assert_input(B.BASELINES42_REL)
    rec = json.loads(Path(R.INPUT_PATHS[B.BASELINES42_REL]).read_text())
    assert real.pm.fit_rows_sha256 == rec["fit_rows_sha256"] == real.bundle.fit_rows_sha256
    rows = B.fit_rows(real.split.scorer_train)
    assert np.isin(rows, real.split.scorer_train).all() and not np.isin(rows, real.split.selection).any()


def test_pm_terms_reproduce_per_anchor_seed42(real):
    """Each of RCA and the nine PM terms, fused with cosine at the stored seed-42 lambda of the half the parity calls
    for ("0" = tuned on half 0, applied to parity 1), reproduces run_baselines' per_anchor_seed42.npz exactly."""
    b = real.bundle
    R.assert_input(B.BASELINES42_REL)
    R.assert_input(PER_ANCHOR42_REL)
    rec = json.loads(Path(R.INPUT_PATHS[B.BASELINES42_REL]).read_text())
    assert tuple(b.pm_terms) == ("rca",) + B.PM_NAMES
    bad = []
    with np.load(R.INPUT_PATHS[PER_ANCHOR42_REL]) as z:
        assert np.array_equal(z["anchor_group"], b.cl) and np.array_equal(z["pair_index"], b.pair_index)
        cos_pa = per_anchor(b.cos)
        assert all(np.array_equal(cos_pa[m], z[f"cosine__{m}"]) for m in METRICS)
        for name, term in b.pm_terms.items():
            picks = rec["scorers"][name]["lambda_picks"]
            lam = {h: (np.inf if picks[str(h)] == "inf" else float(picks[str(h)])) for h in (0, 1)}
            out = {c: {d: np.empty_like(b.cos[c][d]) for d in DIRECTIONS} for c in CONDITIONS}
            for h in (0, 1):
                fused = fused_scores(b.cos, term, lam[h])
                apply = b.parity == 1 - h
                for c in CONDITIONS:
                    for d in DIRECTIONS:
                        out[c][d][apply] = fused[c][d][apply]
            pa = per_anchor(out)
            bad += [f"{name}__{m}" for m in METRICS if not np.array_equal(pa[m], z[f"{name}__{m}"])]
    assert not bad, bad


# ---------------------------------------------------------------- records

def test_bundle_records(real):
    b = real.bundle
    assert b.module_sha256 == R.r6_module_shas()
    assert b.input_sha256 == R.assert_inputs()
    assert b.coef_sha256 == H.coef_sha256(real.ctx.heads)
    assert b.mode == "selection" and b.seed == R.DEV_SEED and not b.smoke
    assert b.checks and all(v is True for v in b.checks.values())
