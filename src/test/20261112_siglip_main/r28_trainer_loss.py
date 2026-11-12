"""r10's buddy-removal genre body, changing only r14's teacher-shape assertion.

Other heads use r10's wrappers and AB loss; other rows use r14's loss unchanged.
"""
import torch

import rb_student
import rg_loss
import rg_variants as V
from ab_paths import ms_train


def genre_loss(models, img, cap, pairs, weights, row, teacher, switches, genre):
    """The genre head's loss with `ab_train`'s buddy switches (copied from `rg_loss.switched_head_loss`'s genre body, which
    asserts R6-5's switches): the operations and their order are the frozen body's, so at R6-5's switches the loss is
    the frozen loss bit for bit (a test checks it, in training mode). `buddy` 0 drops the four buddy terms (the caption
    side is still run, as in `ab_train`); `endpoints` 'image' keeps the image-image term alone, not averaged (D2's R7).
    The direct term is `genre.m`'s, the relational term `genre.lam`'s at `genre.tau_t`."""
    log = genre.term_log
    if log is not None:
        log.clear()
    assert genre.n_anchors == 0 and not switches['own'] and switches['teacher'] == 'instructed'
    assert row['form'] == 'a' and row['graph_terms'] and row['distill'] == 'direct'
    assert row['distill_weight'] == V.TEACHER_WEIGHT
    assert teacher is not None and teacher.shape == (len(img), rb_student.OUT) and not teacher.requires_grad
    a, b = torch.as_tensor(pairs, device=img.device).unbind(1)
    ids = torch.cat((a, b))
    z = models['img'](img[ids])
    ia, ib = z.chunk(2)
    w = torch.as_tensor(weights, device=img.device)
    sym = ms_train.symmetric
    terms = []
    if switches['endpoints'] == 'image':
        if switches['buddy']:
            terms.append(sym(ia, ib, w))
    else:
        ca, cb = models['cap'](cap[ids]).chunk(2)
        if switches['buddy']:
            terms.append((sym(ia, ib, w) + sym(ia, cb, w) + sym(ca, ib, w) + sym(ca, cb, w)) / 4)
    if genre.m:
        direct = ms_train.distill_loss(z, teacher[ids], ids, 'direct')
        terms.append(row['distill_weight'] * direct)
        if log is not None:
            log['direct'] = direct.detach()
    if genre.lam:
        relational = rg_loss.relational_loss(z, teacher[ids], ids, genre.tau_t)
        terms.append(genre.lam * relational)
        if log is not None:
            log['relational'] = relational.detach()
    assert terms, 'the removal left no loss term'
    loss = terms[0]
    for term in terms[1:]:
        loss = loss + term
    return loss
