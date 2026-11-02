"""Non-Qwen c* genre loss: the frozen body with only its teacher shape assertion corrected.

The teacher matches the indexed image tensor's rows and the student's output width.
All computation and dependencies remain the frozen loss's; Qwen never uses this copy.
"""
import torch

import rb_student
import rg_variants as V
from rg_loss import ab_train, anchors_loss, ms_train, rb_rows, relational_loss


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
    assert teacher is not None and teacher.shape == (len(img), rb_student.OUT) and not teacher.requires_grad
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
