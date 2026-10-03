import torch

from src.model.aspect_rule import agreement_weights, zfuse, zscore_rows


def test_agreement_weights_pick_coactive_factors_and_normalize():
    sup_i = torch.zeros(1, 4, 3); sup_t = torch.zeros(1, 4, 3)
    sup_i[..., 0] = 1.0; sup_t[..., 0] = 2.0                  # factor 0 co-active in supports
    con_i = torch.zeros(1, 4, 3); con_t = torch.zeros(1, 4, 3)
    con_i[..., 1] = 1.0; con_t[..., 1] = 1.0                  # factor 1 co-active in contrasts
    w = agreement_weights(sup_i, sup_t, con_i, con_t)
    assert torch.allclose(w, torch.tensor([[1.0, 0.0, 0.0]]))


def test_agreement_weights_all_zero_when_no_positive_gap():
    z = torch.zeros(2, 4, 3)
    assert torch.equal(agreement_weights(z, z, z + 1, z + 1), torch.zeros(2, 3))


def test_zfuse_constant_term_falls_back_to_cosine():                  # Review Focus 3
    cos = torch.tensor([[0.3, 0.1, 0.2]])
    term = torch.tensor([[5.0, 5.0, 5.0]])
    out = zfuse(cos, term, 4.0)
    assert torch.isfinite(out).all() and torch.equal(out.argsort(), zscore_rows(cos).argsort())


def test_zfuse_inf_is_term_only_and_zero_is_cos_only():
    cos = torch.tensor([[0.3, 0.1, 0.2]]); term = torch.tensor([[0.0, 1.0, 2.0]])
    assert torch.equal(zfuse(cos, term, float("inf")), zscore_rows(term))
    assert torch.equal(zfuse(cos, term, 0.0), zscore_rows(cos))


def test_nonfinite_rows_stay_nonfinite_through_zfuse():                 # Review Focus 4
    cos = torch.tensor([[0.3, 0.1, 0.2], [0.3, float("nan"), 0.2]])
    term = torch.tensor([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
    assert torch.isnan(zscore_rows(cos)[1]).all() and torch.isfinite(zscore_rows(cos)[0]).all()
    assert torch.isnan(zfuse(cos, term, 4.0)[1]).all()               # NaN in cos, finite term
    assert torch.isfinite(zfuse(cos, term, float("inf"))).all()      # term-only ignores cos
    assert torch.isnan(zfuse(cos, term, 0.0)[1]).all()
    cos2 = torch.tensor([[0.3, 0.1, 0.2]] * 2)
    term2 = torch.tensor([[0.0, 1.0, 2.0], [0.0, float("inf"), 2.0]])
    assert torch.isnan(zfuse(cos2, term2, 4.0)[1]).all() and torch.isfinite(zfuse(cos2, term2, 4.0)[0]).all()
    assert torch.isnan(zfuse(cos2, term2, float("inf"))[1]).all()
    assert torch.isfinite(zfuse(cos2, term2, 0.0)).all()


def test_agreement_weights_nan_codes_give_nan_row():
    sup_i = torch.ones(2, 4, 3); sup_t = torch.ones(2, 4, 3)
    con_i = torch.zeros(2, 4, 3); con_t = torch.zeros(2, 4, 3)
    sup_i[1, 0, 0] = float("nan")
    w = agreement_weights(sup_i, sup_t, con_i, con_t)
    assert torch.isfinite(w[0]).all() and torch.isnan(w[1]).all()
