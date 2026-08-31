import torch

from src.metrics.regularizer import reorder_features_to_z


def test_reorder_aligns_to_z_order():
    feat = torch.tensor([[10.0], [20.0], [30.0], [40.0]])  # in feature-store order
    feat_ids = [103, 101, 104, 102]                        # store order of sample ids
    z_ids = [101, 102, 103, 104]                           # embedding-manager order
    out = reorder_features_to_z(feat, feat_ids, z_ids)
    assert out[0].item() == 20.0  # z-pos 0 = sample 101 -> store idx 1
    assert out[1].item() == 40.0  # 102 -> store idx 3
    assert out[2].item() == 10.0  # 103 -> store idx 0
    assert out[3].item() == 30.0  # 104 -> store idx 2
    print("test_reorder_aligns_to_z_order PASS")


def test_reorder_identity_when_orders_match():
    feat = torch.randn(5, 3)
    ids = [7, 8, 9, 10, 11]
    out = reorder_features_to_z(feat, ids, ids)
    assert torch.equal(out, feat)
    print("test_reorder_identity_when_orders_match PASS")


if __name__ == "__main__":
    test_reorder_aligns_to_z_order()
    test_reorder_identity_when_orders_match()
    print("ALL PASS")
