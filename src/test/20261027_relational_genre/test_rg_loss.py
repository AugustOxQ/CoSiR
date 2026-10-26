"""Tiny CPU checks of the public genre-loss interfaces (ticket 01)."""
import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import ab_train
import rb_rows
import rg_variants as V
from ab_paths import ms_train


def rows(n=6, width=5, seed=7):
    generator = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(n, width, generator=generator), dim=1)


@pytest.fixture
def batch():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        models = nn.ModuleDict({s: nn.Sequential(nn.Linear(5, 5), nn.Tanh(), nn.Dropout(0.1))
                                for s in ('img', 'cap')})
    return dict(models=models, img=rows(8), cap=rows(8, seed=9), teacher=rows(8, seed=11),
                pairs=torch.tensor([[0, 1], [2, 3], [1, 4], [5, 6]]),
                weights=torch.tensor([0.2, 1.0, 0.7, 0.5]), row=rb_rows.recipe('R6-5')['base'],
                aspect='genre', switches=rb_rows.recipe('R6-5')['switches'])


def assert_same_computation(batch, actual_fn, expected_fn):
    """Compare loss, parameter gradients and dropout's final RNG state."""
    rng = torch.get_rng_state()
    actual = actual_fn()
    after = torch.get_rng_state()
    torch.set_rng_state(rng)
    expected = expected_fn()
    assert torch.equal(actual, expected)
    assert torch.equal(after, torch.get_rng_state())
    parameters = tuple(batch['models'].parameters())
    for a, e in zip(torch.autograd.grad(actual, parameters), torch.autograd.grad(expected, parameters)):
        assert torch.equal(a, e)


def test_relational_default_matches_frozen_loss_and_gradient_bit_for_bit():
    from rg_loss import relational_loss

    z = rows().requires_grad_()
    teacher = rows(seed=11)
    ids = torch.tensor([4, 1, 4, 2, 1, 3])
    expected = ms_train.distill_loss(z, teacher, ids, 'relational')
    actual = relational_loss(z, teacher, ids, tau_t=0.05)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.autograd.grad(actual, z)[0], torch.autograd.grad(expected, z)[0])


def test_anchors_equal_in_batch_with_repeated_paintings_and_live_student_gradients():
    from rg_loss import anchors_loss

    z = rows().requires_grad_()
    teacher = rows(seed=11)
    ids = torch.tensor([4, 1, 4, 2, 1, 3])
    expected = ms_train.distill_loss(z, teacher, ids, 'relational')
    actual = anchors_loss(z, teacher, ids, z, teacher, ids)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)
    torch.testing.assert_close(torch.autograd.grad(actual, z)[0],
                               torch.autograd.grad(expected, z)[0], atol=1e-6, rtol=0)


def test_draw_anchors_is_seeded_distinct_fit_local_and_preserves_global_rng():
    from rg_loss import draw_anchors

    before = torch.get_rng_state().clone()
    # CUDA is disabled for this ticket: sampling must not even initialise it.
    cuda_initialised = torch.cuda.is_initialized()
    actual = draw_anchors(5000, 13)
    assert torch.equal(before, torch.get_rng_state())
    assert torch.cuda.is_initialized() == cuda_initialised
    expected = torch.randperm(5000, generator=torch.Generator().manual_seed(13))[:4096]
    assert torch.equal(actual, expected)
    assert len(actual.unique()) == 4096 and actual.min() >= 0 and actual.max() < 5000
    assert torch.equal(actual, draw_anchors(5000, 13))
    assert not torch.equal(actual, draw_anchors(5000, 14))
    with pytest.raises(AssertionError):
        draw_anchors(4095, 13)


@pytest.mark.parametrize('arm,aspect', [('R6-5', 'genre'), (None, 'genre')] +
                         [(arm, aspect) for arm in V.ARMS for aspect in ('emotion', 'style')])
def test_base_and_other_aspects_preserve_original_computation(batch, arm, aspect):
    from rg_loss import GenreLoss, switched_head_loss

    batch['aspect'] = aspect
    genre = None if arm is None else GenreLoss(V.switches(arm), seed=3)
    assert_same_computation(batch, lambda: switched_head_loss(**batch, genre=genre),
                            lambda: ab_train.switched_head_loss(**batch))


@pytest.mark.parametrize('sw', [sw for arm, sw in V.ARMS.items() if arm.startswith('GR-')] +
                         [V.LOSS_SCALE_SWITCHES], ids=lambda sw: f"lambda-{sw['lam']}")
def test_gr_matches_frozen_genre_loss_bit_for_bit_including_dropout(batch, sw):
    import eg_train3
    from rg_loss import GenreLoss, switched_head_loss

    genre = GenreLoss(sw, seed=3)
    frozen = eg_train3.GenreLoss(dict(m=1, lam=sw['lam'], w=0), seed=3)
    assert_same_computation(batch, lambda: switched_head_loss(**batch, genre=genre),
                            lambda: eg_train3.switched_head_loss(**batch, genre=frozen))


def test_gn_has_buddy_and_relational_terms_without_a_direct_term(batch):
    from rg_loss import GenreLoss, switched_head_loss

    batch['models'].eval()
    ids = torch.cat(tuple(batch['pairs'].unbind(1)))
    z = batch['models']['img'](batch['img'][ids])
    buddy_args = dict(batch, row=dict(batch['row'], distill='none'))
    expected = (ab_train.switched_head_loss(**buddy_args) +
                8 * ms_train.distill_loss(z, batch['teacher'][ids], ids, 'relational'))
    actual = switched_head_loss(**batch, genre=GenreLoss(V.switches('GN-8')))
    assert torch.equal(actual, expected)
    parameters = tuple(batch['models'].parameters())
    for a, e in zip(torch.autograd.grad(actual, parameters), torch.autograd.grad(expected, parameters)):
        # The reference uses separate forwards, so parameter-gradient accumulation can round differently.
        torch.testing.assert_close(a, e)


@pytest.mark.parametrize('dev', [False, True], ids=['fit-local-mask', 'dev-no-mask'])
def test_gq_uses_fixed_teacher_anchors_and_masks_only_fit_paintings(batch, dev):
    from rg_loss import GenreLoss, anchors_loss, draw_anchors, switched_head_loss

    batch['models'].eval()
    n_fit = 4100
    fit_teacher = rows(n_fit, seed=13)
    anchor_ids = draw_anchors(n_fit, 3)
    anchors = fit_teacher[anchor_ids]
    genre = GenreLoss(V.switches('GQ-8'), seed=3, anchors=anchors,
                      anchor_ids=anchor_ids, n_fit=n_fit, n_dev=8)
    if not dev:
        # Local positions, unrelated to extraction IDs or the full sample's order.
        batch.update(img=rows(n_fit), cap=rows(n_fit, seed=9), teacher=fit_teacher)
    ids = torch.cat(tuple(batch['pairs'].unbind(1)))
    z = batch['models']['img'](batch['img'][ids])
    expected = (ab_train.switched_head_loss(**batch, include_mask_penalty=not dev) +
                8 * anchors_loss(z, batch['teacher'][ids], ids, anchors, anchors,
                                 None if dev else anchor_ids))
    actual = switched_head_loss(**batch, genre=genre, include_mask_penalty=not dev)
    assert torch.equal(actual, expected)
    assert torch.isfinite(actual)


@pytest.mark.parametrize('arm', ['R6-5', V.LOSS_SCALE, 'GN-8', 'GS-0.02', 'GS-0.01', 'GQ-8'])
def test_per_call_log_is_raw_detached_and_does_not_change_the_computation(batch, arm):
    from rg_loss import GenreLoss, draw_anchors, switched_head_loss

    sw = V.LOSS_SCALE_SWITCHES if arm == V.LOSS_SCALE else V.switches(arm)
    extra = {}
    if arm == 'GQ-8':
        anchor_ids = draw_anchors(4096, 0)
        extra = dict(anchors=rows(4096)[anchor_ids], anchor_ids=anchor_ids, n_fit=4096, n_dev=8)
        # Use the small disjoint dev tensor for this logging test.
    batch['include_mask_penalty'] = arm != 'GQ-8'
    log = {'stale': torch.tensor(-1)}
    logged = GenreLoss(sw, term_log=log, **extra)
    plain = GenreLoss(sw, **extra)
    assert_same_computation(batch, lambda: switched_head_loss(**batch, genre=logged),
                            lambda: switched_head_loss(**batch, genre=plain))
    keys = ({'direct'} if sw['m'] else set())
    if sw['lam']:
        keys.add('anchors' if sw['anchors'] else 'relational')
    assert log.keys() == keys
    assert all(value.ndim == 0 and not value.requires_grad and value.grad_fn is None for value in log.values())
    # A subsequent emotion call clears the log, rather than retaining genre terms.
    switched_head_loss(**dict(batch, aspect='emotion'), genre=logged)
    assert log == {}


def scalar_relational_kl(z, teacher, ids, teacher_tau, student_tau=0.05):
    """Independent scalar arithmetic oracle, without torch matrix/softmax operations."""
    z, teacher, ids = z.tolist(), teacher.tolist(), ids.tolist()
    divergences = []
    for i in range(len(ids)):
        others = [j for j in range(len(ids)) if ids[j] != ids[i]]
        teacher_logits = [sum(x * y for x, y in zip(teacher[i], teacher[j])) / teacher_tau for j in others]
        student_logits = [sum(x * y for x, y in zip(z[i], z[j])) / student_tau for j in others]
        log_probabilities = []
        for logits in (teacher_logits, student_logits):
            peak = max(logits)
            normalizer = math.log(sum(math.exp(x - peak) for x in logits))
            log_probabilities.append([x - peak - normalizer for x in logits])
        logp, logq = log_probabilities
        divergences.append(sum(math.exp(p) * (p - q) for p, q in zip(logp, logq)))
    return sum(divergences) / len(divergences)


@pytest.mark.parametrize('arm', ['GS-0.02', 'GS-0.01'])
def test_gs_sharpens_only_teacher_rows_and_keeps_direct_at_005(batch, arm):
    from rg_loss import GenreLoss, relational_loss, switched_head_loss

    batch['models'].eval()
    ids = torch.cat(tuple(batch['pairs'].unbind(1)))
    z = batch['models']['img'](batch['img'][ids])
    t = batch['teacher'][ids]
    sw = V.switches(arm)
    expected = scalar_relational_kl(z, t, ids, sw['tau_t'])
    assert relational_loss(z, t, ids, sw['tau_t']).item() == pytest.approx(expected, abs=2e-6)
    # Sharpening the student too would produce a different loss.
    assert abs(expected - scalar_relational_kl(z, t, ids, sw['tau_t'], sw['tau_t'])) > 0.01
    log = {}
    actual = switched_head_loss(**batch, genre=GenreLoss(sw, term_log=log))
    assert torch.equal(log['direct'], ms_train.distill_loss(z, t, ids, 'direct'))
    assert log['relational'].item() == pytest.approx(expected, abs=2e-6)
    assert actual.item() == pytest.approx(ab_train.switched_head_loss(**batch).item() + 8 * expected,
                                          abs=2e-5)


def test_anchor_kl_masks_all_matching_columns_and_zeros_excluded_logs():
    from rg_loss import anchors_loss

    z = torch.zeros(2, 1, requires_grad=True)
    t = torch.ones(2, 1)
    a_s = torch.tensor([[1.0], [2.0], [3.0]])
    a_t = torch.tensor([[0.0], [0.05 * math.log(3)], [0.05 * math.log(2)]])
    ids, a_ids = torch.tensor([7, 5]), torch.tensor([7, 7, 9])
    # First row has one surviving anchor: KL=0. Second has teacher
    # probabilities (1/6, 1/2, 1/3), versus a uniform student row.
    full_kl = math.log(0.5) / 6 + math.log(1.5) / 2
    actual = anchors_loss(z, t, ids, a_s, a_t, a_ids)
    assert actual.item() == pytest.approx(full_kl / 2, abs=1e-7)
    assert torch.isfinite(torch.autograd.grad(actual, z)[0]).all()
    # None declares disjoint painting sets: both rows retain all anchors.
    assert anchors_loss(z, t, ids, a_s, a_t, None).item() == pytest.approx(full_kl, abs=1e-7)


def test_rows_without_another_painting_are_rejected():
    from rg_loss import anchors_loss, relational_loss

    z, t, ids = rows(3), rows(3, seed=1), torch.ones(3, dtype=torch.long)
    with pytest.raises(AssertionError, match='other paintings'):
        relational_loss(z, t, ids)
    with pytest.raises(AssertionError, match='another painting'):
        anchors_loss(z, t, ids, z, t, ids)


def test_relational_rejects_ids_that_do_not_name_every_row():
    from rg_loss import relational_loss

    with pytest.raises(AssertionError):
        relational_loss(rows(), rows(seed=1), torch.tensor([1, 2]))


@pytest.mark.parametrize('problem', ['missing-count', 'missing-ids', 'global-ids', 'duplicate-ids',
                                   'trainable-teacher', 'wrong-count', 'missing-dev-count', 'dev-count-is-fit-count'])
def test_gq_rejects_invalid_anchor_context(problem):
    from rg_loss import GenreLoss

    args = dict(anchors=rows(4096), anchor_ids=torch.arange(4096), n_fit=4096, n_dev=8)
    if problem == 'missing-count':
        args['n_fit'] = None
    elif problem == 'missing-ids':
        args['anchor_ids'] = None
    elif problem == 'global-ids':
        args['anchor_ids'] += 5000
    elif problem == 'duplicate-ids':
        args['anchor_ids'].zero_()
    elif problem == 'missing-dev-count':
        args['n_dev'] = None
    elif problem == 'dev-count-is-fit-count':
        args['n_dev'] = 4096
    elif problem == 'trainable-teacher':
        args['anchors'].requires_grad_()
    else:
        args['anchors'] = args['anchors'][:2]
    with pytest.raises(AssertionError):
        GenreLoss(V.switches('GQ-8'), **args)


def gq_fit_context(n_fit=4096, n_dev=8):
    """A GQ genre loss whose anchors are the rows of ``fit_teacher`` at ``anchor_ids``."""
    from rg_loss import GenreLoss, draw_anchors

    fit_teacher = rows(n_fit, seed=13)
    anchor_ids = draw_anchors(n_fit, 3)
    genre = GenreLoss(V.switches('GQ-8'), anchors=fit_teacher[anchor_ids], anchor_ids=anchor_ids,
                      n_fit=n_fit, n_dev=n_dev)
    return genre, fit_teacher


def test_gq_checks_fit_tensor_count_and_dev_teacher_count(batch):
    from rg_loss import switched_head_loss

    genre, _ = gq_fit_context()
    with pytest.raises(AssertionError, match='fit-local'):
        switched_head_loss(**batch, genre=genre)
    with pytest.raises(AssertionError):
        switched_head_loss(**dict(batch, teacher=rows(4096)), genre=genre, include_mask_penalty=False)


def test_gq_dev_call_refuses_fit_sized_tensors(batch):
    from rg_loss import switched_head_loss

    genre, fit_teacher = gq_fit_context()
    fit = dict(batch, img=rows(4096), cap=rows(4096, seed=9), teacher=fit_teacher)
    # Fit tensors on a dev call would silently skip the anchor masking.
    with pytest.raises(AssertionError, match='dev'):
        switched_head_loss(**fit, genre=genre, include_mask_penalty=False)
    # The dev-sized tensors the genre was built for still pass.
    assert torch.isfinite(switched_head_loss(**batch, genre=genre, include_mask_penalty=False))


def test_gq_student_rows_stay_at_005_so_a_sharpened_teacher_is_refused(batch):
    from rg_loss import switched_head_loss

    genre, _ = gq_fit_context()
    genre.tau_t = 0.02
    with pytest.raises(AssertionError, match='tau_t'):
        switched_head_loss(**batch, genre=genre, include_mask_penalty=False)


def test_gq_fit_call_refuses_anchor_rows_that_are_not_the_fit_teacher_rows(batch):
    from rg_loss import GenreLoss, switched_head_loss

    genre, fit_teacher = gq_fit_context()
    fit = dict(batch, img=rows(4096), cap=rows(4096, seed=9), teacher=fit_teacher)
    assert torch.isfinite(switched_head_loss(**fit, genre=genre))
    stale = GenreLoss(V.switches('GQ-8'), anchors=rows(4096, seed=99)[genre.anchor_ids],
                      anchor_ids=genre.anchor_ids, n_fit=4096, n_dev=8)
    with pytest.raises(AssertionError, match='anchor rows'):
        switched_head_loss(**fit, genre=stale)
    # The same tensors are a legitimate dev call: no anchor-row check there.
    assert torch.isfinite(switched_head_loss(**batch, genre=stale, include_mask_penalty=False))


def test_gq_moves_anchors_to_the_teacher_device_once_without_changing_values(batch, monkeypatch):
    from rg_loss import switched_head_loss

    genre, _ = gq_fit_context()
    values = genre.anchors.clone(), genre.anchor_ids.clone()
    seen = []
    to = genre.to
    monkeypatch.setattr(genre, 'to', lambda device: seen.append(device) or to(device))
    switched_head_loss(**batch, genre=genre, include_mask_penalty=False)
    assert seen == [batch['teacher'].device]
    assert torch.equal(genre.anchors, values[0]) and torch.equal(genre.anchor_ids, values[1])
    moved = genre.to(torch.device('meta'))
    assert moved is genre and genre.anchors.device.type == genre.anchor_ids.device.type == 'meta'


@pytest.mark.parametrize('logged', [False, True])
def test_gn_computes_no_direct_term_at_all(batch, monkeypatch, logged):
    from rg_loss import GenreLoss, switched_head_loss

    real = ms_train.distill_loss

    def no_direct(z, t, ids, mode, *args, **kwargs):
        assert mode != 'direct', 'GN must not compute a direct term, not even to multiply it by 0'
        return real(z, t, ids, mode, *args, **kwargs)

    monkeypatch.setattr(ms_train, 'distill_loss', no_direct)
    log = {} if logged else None
    switched_head_loss(**batch, genre=GenreLoss(V.switches('GN-8'), term_log=log))
    if logged:
        assert 'direct' not in log
    # Sanity: the guard does fire for an arm that has the direct term.
    with pytest.raises(AssertionError, match='direct'):
        switched_head_loss(**batch, genre=GenreLoss(V.switches('GR-8')))
