import numpy as np

from src.eval.factor_gates import (
    FactorGateThresholds, community_spanning, dead_and_private, evaluate_factor_gates,
    factor_correlation_summary, linear_readout_rel_l2, paired_retrieval_recall,
    participation_ratio, relative_l2,
)


def _collapsed_codes(n=4000, factors=32, seed=0):
    """32 near-copies / anti-copies of one latent axis: the observed failure mode."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 1))
    signs = np.where(np.arange(factors) % 3 == 0, -1.0, 1.0)
    return np.maximum(0.0, 1.0 + z * signs + 0.05 * rng.normal(size=(n, factors)))


def _independent_codes(n=4000, factors=32, seed=0):
    rng = np.random.default_rng(seed)
    return np.maximum(0.0, rng.normal(size=(n, factors)) - 0.5)


def test_participation_ratio_separates_collapsed_from_independent():
    assert participation_ratio(_collapsed_codes()) < 2.0
    assert participation_ratio(_independent_codes()) > 20.0


def test_redundancy_summary_flags_copies_and_reports_constant_columns():
    codes = _collapsed_codes()
    codes[:, 5] = 0.7
    summary = factor_correlation_summary(codes)
    assert np.isfinite(summary["max_abs"]) and summary["max_abs"] > 0.9
    assert summary["constant_factors"] == [5]
    independent = factor_correlation_summary(_independent_codes())
    assert independent["max_abs"] < 0.2 and independent["pairs_at_or_above"] == 0


def test_all_zero_codes_are_degenerate_not_errors():
    zeros = np.zeros((2000, 32))
    assert participation_ratio(zeros) == 0.0
    assert paired_retrieval_recall(zeros, zeros, pool=500, k=10) == 0.0


def test_ties_count_against_the_positive_in_retrieval():
    tied = np.ones((2000, 32))
    assert paired_retrieval_recall(tied, tied, pool=500, k=10) == 0.0
    codes = _independent_codes(n=2000) + 1e-3
    assert paired_retrieval_recall(codes, codes, pool=500, k=1) == 1.0


def test_linear_readout_recovers_linear_features_and_not_noise():
    rng = np.random.default_rng(1)
    codes = _independent_codes(n=6000)
    features = codes @ rng.normal(size=(32, 64)) + 3.0
    assert linear_readout_rel_l2(codes[:4000], features[:4000], codes[4000:], features[4000:]) < 0.05
    noise = rng.normal(size=codes.shape)
    mean_only = relative_l2(features[4000:], np.broadcast_to(features[:4000].mean(0), features[4000:].shape))
    assert linear_readout_rel_l2(noise[:4000], features[:4000], noise[4000:], features[4000:]) > 0.95 * mean_only


def test_dead_private_spanning_match_task3_definitions():
    img, txt = np.zeros((6, 4)), np.zeros((6, 4))
    img[:, 0] = txt[:, 0] = 1.0          # shared, spread over 3 communities -> spanning
    img[:3, 1] = txt[:3, 1] = 1.0        # shared, 2/3 of mass in community 0 -> topic-like
    img[:, 2] = 1.0                      # image-only -> private (still spanning)
    labels = np.array([0, 0, 1, 1, 2, 2])  # factor 3 all zero -> dead
    dead, private = dead_and_private(img, txt, 1e-4, 0.05)
    assert dead.tolist() == [3] and private.tolist() == [2]
    span = community_spanning(img, txt, labels, 1e-4, 0.50)
    assert span["spanning_indices"] == [0, 2] and span["topic_like_indices"] == [1]
    assert span["spanning_fraction"] == 0.5


def test_evaluate_factor_gates_fails_collapsed_and_passes_independent():
    rng = np.random.default_rng(2)
    latent_img = _independent_codes(n=6000, seed=3)
    # same support as the image codes, so noise cannot switch on extra factors (sparsity gate)
    latent_txt = np.where(latent_img > 0,
                          np.maximum(0.0, latent_img + 0.05 * rng.normal(size=latent_img.shape)), 0.0)
    mixing = rng.normal(size=(32, 64))
    img_feat = latent_img @ mixing + 0.1 * rng.normal(size=(6000, 64)) + 2.0
    txt_feat = latent_txt @ mixing + 0.1 * rng.normal(size=(6000, 64)) + 2.0
    labels = rng.integers(0, 8, size=4000)
    thresholds = FactorGateThresholds(retrieval_pool=500)

    def gates(img_codes, txt_codes):
        return evaluate_factor_gates(
            fit_img_codes=img_codes[:4000], fit_txt_codes=txt_codes[:4000],
            fit_img_features=img_feat[:4000], fit_txt_features=txt_feat[:4000],
            eval_img_codes=img_codes[4000:], eval_txt_codes=txt_codes[4000:],
            eval_img_features=img_feat[4000:], eval_txt_features=txt_feat[4000:],
            community_img_codes=img_codes[:4000], community_txt_codes=txt_codes[:4000],
            community_labels=labels, thresholds=thresholds,
        )

    good = gates(latent_img, latent_txt)
    assert good.all_passed, good.passed
    collapsed = _collapsed_codes(n=6000, seed=4)
    bad = gates(collapsed, collapsed.copy())
    assert not bad.all_passed
    for name in ("participation_ratio", "redundancy", "readout", "sparsity"):
        assert bad.passed[name] is False, name
