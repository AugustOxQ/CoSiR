import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
import discover_checkpoints as dc


def _write_fake_checkpoint(root, combine_side, conditioning_mode, init_strategy, with_embeddings=True):
    exp_dir = tempfile.mkdtemp(dir=root)
    cfg = {
        "model": {"combine_side": combine_side, "conditioning_mode": conditioning_mode},
        "train": {"initialization_strategy": init_strategy},
    }
    with open(os.path.join(exp_dir, "experiment_metadata.json"), "w") as f:
        json.dump({"config": repr(cfg)}, f)
    if with_embeddings:
        emb_dir = os.path.join(exp_dir, "final_embeddings")
        os.makedirs(emb_dir)
        open(os.path.join(emb_dir, "embeddings.npy"), "wb").close()
    return exp_dir


def test_finds_matching_asymmetric_buddy_checkpoint():
    with tempfile.TemporaryDirectory() as root:
        good = _write_fake_checkpoint(root, "img", "asymmetric", "buddies")
        found = dc.find_candidate_checkpoints([os.path.join(root, "*")])
        assert found == [good]


def test_rejects_symmetric_and_wrong_init():
    with tempfile.TemporaryDirectory() as root:
        _write_fake_checkpoint(root, "img", "symmetric_shared", "buddies")
        _write_fake_checkpoint(root, "img", "asymmetric", "imgtxt")
        _write_fake_checkpoint(root, "txt", "asymmetric", "buddies")
        found = dc.find_candidate_checkpoints([os.path.join(root, "*")])
        assert found == []


def test_skips_dir_missing_embeddings():
    with tempfile.TemporaryDirectory() as root:
        _write_fake_checkpoint(root, "img", "asymmetric", "buddies", with_embeddings=False)
        found = dc.find_candidate_checkpoints([os.path.join(root, "*")])
        assert found == []
