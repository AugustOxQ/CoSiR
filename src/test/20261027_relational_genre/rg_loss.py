"""Genre loss for the relational round (rule §2.1); older round code stays unchanged."""
import torch
from torch.nn import functional as F

import rg_paths  # noqa: F401 (older rounds on sys.path)
import rg_variants as V
import ab_train
import rb_rows
from ab_paths import ms_train


class GenreLoss:
    """One run's genre switches, kept separate from the unchanged recipe switches.

    GQ callers pass the fixed teacher ``anchors``, their ``anchor_ids`` in the
    fit tensor's local order, ``n_fit`` (the fit tensor's row count) and ``n_dev``
    (the dev tensor's row count, different from ``n_fit``); never full-sample or
    extraction IDs. Build/draw these after the model is built. Dev tensors are
    disjoint from fit; the trainer's ``include_mask_penalty=False`` identifies
    those calls, which must carry exactly ``n_dev`` rows. On fit calls the anchor
    rows must equal the fit teacher's rows at ``anchor_ids``. The anchors move to
    the teacher's device on the first GQ call (``to``).

    Optional ``term_log`` is a caller-owned dict, cleared on each call and filled
    with detached, unweighted scalar tensors for the terms present (``direct``,
    ``relational`` or ``anchors``). Copy it after each call to retain a history.
    """

    def __init__(self, switches, seed=0, *, anchors=None, anchor_ids=None, n_fit=None, n_dev=None,
                 term_log=None):
        assert switches in list(V.ARMS.values()) + [V.LOSS_SCALE_SWITCHES], switches
        self.switches = dict(switches)
        self.seed = seed
        self.m, self.lam = switches['m'], switches['lam']
        self.tau_t, self.n_anchors = switches['tau_t'], switches['anchors']
        self.anchors, self.anchor_ids, self.n_fit, self.n_dev = anchors, anchor_ids, n_fit, n_dev
        self.term_log = term_log
        if anchors is not None:
            assert self.n_anchors == V.N_ANCHORS and n_fit is not None and n_fit >= V.N_ANCHORS
            assert anchors.ndim == 2 and len(anchors) == V.N_ANCHORS and not anchors.requires_grad
            assert n_dev is not None and n_dev > 0 and n_dev != n_fit, 'dev and fit row counts must differ'
            assert anchor_ids is not None and anchor_ids.shape == (V.N_ANCHORS,)
            assert anchor_ids.dtype == torch.long
            assert len(anchor_ids.unique()) == V.N_ANCHORS
            assert bool(((anchor_ids >= 0) & (anchor_ids < n_fit)).all()), 'anchors must use fit-local IDs'

    def to(self, device):
        """Move the anchors and their positions to ``device`` (values unchanged); returns self."""
        if self.anchors is not None and (self.anchors.device != device or self.anchor_ids.device != device):
            self.anchors, self.anchor_ids = self.anchors.to(device), self.anchor_ids.to(device)
        return self

    @property
    def off(self):
        return V.is_off(self.switches)


def switched_head_loss(models, img, cap, pairs, weights, row, aspect, teacher=None, extras=None,
                       include_mask_penalty=True, *, switches=None, genre=None):
    """Original base/non-genre losses, or the genre switches on a single output z."""
    log = None if genre is None else genre.term_log
    if log is not None:
        log.clear()
    if genre is None or genre.off or aspect != 'genre':
        if log is not None and aspect == 'genre':
            # Keep the baseline call itself untouched. Observe its existing z,
            # then evaluate the requested diagnostic without a second forward,
            # autograd graph, or random draw. No hook exists unless logging is on.
            outputs = []
            handle = models['img'].register_forward_hook(lambda _m, _a, z: outputs.append(z.detach()))
            try:
                loss = ab_train.switched_head_loss(models, img, cap, pairs, weights, row, aspect, teacher,
                                                   extras, include_mask_penalty, switches=switches)
            finally:
                handle.remove()
            assert len(outputs) == 1
            ids = torch.cat(tuple(torch.as_tensor(pairs, device=img.device).unbind(1)))
            with torch.no_grad():
                log['direct'] = ms_train.distill_loss(outputs[0], teacher[ids], ids, 'direct').detach()
            return loss
        return ab_train.switched_head_loss(models, img, cap, pairs, weights, row, aspect, teacher, extras,
                                           include_mask_penalty, switches=switches)
    # Copied from eg_train3.switched_head_loss's R6-5 body: one z, the same buddy
    # operation order, and a left-to-right sum. Do not call a second image forward.
    assert switches == rb_rows.RECIPES[V.BASE]['switches'], switches
    assert row['form'] == 'a' and row['graph_terms'] and extras is None and row['distill'] == 'direct'
    assert row['distill_weight'] == V.TEACHER_WEIGHT
    assert teacher is not None and teacher.shape == img.shape and not teacher.requires_grad
    a, b = torch.as_tensor(pairs, device=img.device).unbind(1)
    ids = torch.cat((a, b))
    z = models['img'](img[ids])
    ia, ib = z.chunk(2)
    w = torch.as_tensor(weights, device=img.device)
    ca, cb = models['cap'](cap[ids]).chunk(2)
    sym = ms_train.symmetric
    terms = [(sym(ia, ib, w) + sym(ia, cb, w) + sym(ca, ib, w) + sym(ca, cb, w)) / 4]
    if genre.m:
        direct = ms_train.distill_loss(z, teacher[ids], ids, 'direct')
        terms.append(row['distill_weight'] * direct)
        if log is not None:
            log['direct'] = direct.detach()
    if genre.lam:
        if genre.n_anchors:
            assert genre.anchors is not None and genre.anchor_ids is not None
            assert genre.tau_t == V.TAU, 'GQ anchors need tau_t equal to the student temperature (no arm sharpens it)'
            genre.to(teacher.device)
            assert teacher.shape[0] == len(img), 'teacher must have this fit/dev tensor row count'
            if include_mask_penalty:
                assert len(img) == genre.n_fit, 'fit tensors must use fit-local IDs'
                assert torch.equal(teacher[genre.anchor_ids], genre.anchors), \
                    'anchor rows must equal the fit teacher rows at the anchor positions'
            else:
                assert len(img) == teacher.shape[0] == genre.n_dev, 'dev call must carry the dev tensors'
            # Dev painting sets are disjoint from fit. Their local indices can
            # coincide numerically, so comparing them to fit-local IDs is wrong.
            a_ids = genre.anchor_ids if include_mask_penalty else None
            relational = anchors_loss(z, teacher[ids], ids, genre.anchors, genre.anchors, a_ids, V.TAU)
            term_name = 'anchors'
        else:
            relational = relational_loss(z, teacher[ids], ids, genre.tau_t)
            term_name = 'relational'
        terms.append(genre.lam * relational)
        if log is not None:
            log[term_name] = relational.detach()
    loss = terms[0]
    for term in terms[1:]:
        loss = loss + term
    return loss


def draw_anchors(n_fit, seed):
    """Draw 4,096 fit-local CPU positions without touching global CPU or CUDA RNGs."""
    assert n_fit >= V.N_ANCHORS, 'GQ needs at least 4,096 fit paintings'
    generator = torch.Generator(device='cpu').manual_seed(seed)
    return torch.randperm(n_fit, generator=generator, device='cpu')[:V.N_ANCHORS]


def relational_loss(z, t, ids, tau_t=V.TAU):
    """In-batch teacher-to-student KL; only the teacher temperature varies."""
    # Copied from ms_train.distill_loss's relational body, with a teacher temperature.
    assert z.ndim == 2 and t.shape == z.shape and not t.requires_grad
    assert ids.shape == (len(z),) and len(z) > 0
    assert tau_t > 0
    same = ids[:, None] == ids[None, :]
    assert (~same).any(dim=1).all(), 'relational rows need other paintings'
    student = (z @ z.T / 0.05).masked_fill(same, -torch.inf)
    target = (t @ t.T / tau_t).masked_fill(same, -torch.inf)
    logp, logq = F.log_softmax(target, dim=1), F.log_softmax(student, dim=1)
    return (logp.exp() * (logp.masked_fill(same, 0) - logq.masked_fill(same, 0))).sum(1).mean()


def anchors_loss(z, t, ids, a_s, a_t, a_ids, tau=V.TAU):
    """KL over two anchor sets, excluding every anchor of the row's painting.

    ``a_ids=None`` means the caller has established disjoint painting sets (dev).
    Student anchors stay attached: EQ needs their column-side gradients too.
    """
    assert z.ndim == t.ndim == a_s.ndim == a_t.ndim == 2
    assert z.shape == t.shape and a_s.shape == a_t.shape and z.shape[1] == a_s.shape[1]
    assert not t.requires_grad and not a_t.requires_grad
    assert ids.shape == (len(z),) and len(z) > 0 and len(a_s) > 0 and tau > 0
    if a_ids is None:
        same = torch.zeros((len(z), len(a_s)), dtype=torch.bool, device=z.device)
    else:
        assert a_ids.shape == (len(a_s),)
        same = ids[:, None] == a_ids[None, :]
    assert (~same).any(dim=1).all(), 'anchor rows need another painting'
    student = (z @ a_s.T / tau).masked_fill(same, -torch.inf)
    target = (t @ a_t.T / tau).masked_fill(same, -torch.inf)
    logp, logq = F.log_softmax(target, dim=1), F.log_softmax(student, dim=1)
    return (logp.exp() * (logp.masked_fill(same, 0) - logq.masked_fill(same, 0))).sum(1).mean()
