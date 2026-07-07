import torch
import torch.nn.functional as F

from src.metrics.regularizer import buddy_contrastive_loss, build_neighbor_csr


def _toy_csr(edges, n):
    if len(edges) == 0:
        ei = torch.empty(2, 0, dtype=torch.long)
    else:
        ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return build_neighbor_csr(ei, n)


def test_isolated_anchor_zero():
    # node 0—1 are buddies; node 2 is isolated. Batch = [2] only -> exactly 0, no grad.
    indptr, indices = _toy_csr([[0, 1]], 3)
    N, D = 3, 5
    table = torch.randn(N, D)
    comb = torch.randn(1, D, requires_grad=True)
    neg = torch.randn(1, D)
    loss, align = buddy_contrastive_loss(
        comb, torch.tensor([2]), table, lambda x: x, neg, indptr, indices, num_pos=2
    )
    assert loss.item() == 0.0
    assert align.item() == 0.0
    loss.backward()  # must not raise; grad-safe zero
    print("test_isolated_anchor_zero PASS")


def test_positive_gathered_from_right_row():
    # anchor at z-pos 0, single buddy z-pos 3. alignment must equal cos(comb, table[3]).
    indptr, indices = _toy_csr([[0, 3]], 4)
    N, D = 4, 6
    table = F.normalize(torch.randn(N, D), dim=-1)
    comb = F.normalize(torch.randn(1, D), dim=-1)
    neg = F.normalize(torch.randn(1, D), dim=-1)
    _, align = buddy_contrastive_loss(
        comb, torch.tensor([0]), table, lambda x: x, neg, indptr, indices, num_pos=1
    )
    expected = F.cosine_similarity(comb, table[3:4], dim=-1).item()
    assert abs(align.item() - expected) < 1e-5
    print("test_positive_gathered_from_right_row PASS")


def test_self_masking_excludes_own_row():
    # B=1: the only candidate negative is the anchor's own row. Masking it leaves
    # only positives -> loss is exactly 0. If self were NOT masked, a real negative
    # would remain and the loss would be > 0. (Scaling the negative would NOT test
    # this, since the loss L2-normalizes negatives and so ignores their magnitude.)
    indptr, indices = _toy_csr([[0, 1]], 2)
    N, D = 2, 4
    table = F.normalize(torch.randn(N, D), dim=-1)
    comb = F.normalize(torch.randn(1, D), dim=-1)
    neg = F.normalize(torch.randn(1, D), dim=-1)
    loss, _ = buddy_contrastive_loss(comb, torch.tensor([0]), table, lambda x: x, neg, indptr, indices, num_pos=1)
    assert abs(loss.item()) < 1e-6, loss.item()
    print("test_self_masking_excludes_own_row PASS")


def test_gradient_raises_alignment():
    # Optimising comb should pull anchors toward their (frozen) buddy targets.
    torch.manual_seed(0)
    indptr, indices = _toy_csr([[0, 1], [2, 3]], 4)
    N, D = 4, 8
    table = F.normalize(torch.randn(N, D), dim=-1)
    comb = torch.randn(2, D, requires_grad=True)  # anchors at z-pos 0 and 2
    neg = F.normalize(torch.randn(2, D), dim=-1)
    anchor_pos = torch.tensor([0, 2])

    def alignment(c):
        with torch.no_grad():
            _, a = buddy_contrastive_loss(c, anchor_pos, table, lambda x: x, neg, indptr, indices, num_pos=1)
        return a.item()

    before = alignment(comb)
    opt = torch.optim.SGD([comb], lr=1.0)
    for _ in range(100):
        opt.zero_grad()
        loss, _ = buddy_contrastive_loss(comb, anchor_pos, table, lambda x: x, neg, indptr, indices, num_pos=1)
        loss.backward()
        opt.step()
    after = alignment(comb)
    assert after > before + 0.1, (before, after)
    print("test_gradient_raises_alignment PASS")


def test_scalar_and_finite():
    indptr, indices = _toy_csr([[0, 1], [1, 2]], 3)
    N, D = 3, 4
    table = F.normalize(torch.randn(N, D), dim=-1)
    comb = F.normalize(torch.randn(3, D), dim=-1)
    neg = F.normalize(torch.randn(3, D), dim=-1)
    pos = torch.tensor([0, 1, 2])
    for temp in (0.07, 1.0):
        loss, align = buddy_contrastive_loss(comb, pos, table, lambda x: x, neg, indptr, indices, num_pos=2, temperature=temp)
        assert loss.dim() == 0 and torch.isfinite(loss)
        assert torch.isfinite(align)
    print("test_scalar_and_finite PASS")


if __name__ == "__main__":
    test_isolated_anchor_zero()
    test_positive_gathered_from_right_row()
    test_self_masking_excludes_own_row()
    test_gradient_raises_alignment()
    test_scalar_and_finite()
    print("ALL PASS")
