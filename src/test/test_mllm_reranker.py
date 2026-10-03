import numpy as np
import pytest

from src.eval.mllm_reranker import LETTERS, build_messages, unpermute


def _render(msgs):
    """Join the parts the way the Qwen3-VL chat template does: no separator; an image is a placeholder token."""
    return "".join(p["text"] if p["type"] == "text" else "<img>" for m in msgs for p in m["content"])


def _cases():
    sup = [("img_s%d.jpg" % i, "support caption %d" % i) for i in range(4)]
    con = [("img_c%d.jpg" % i, "contrast caption %d" % i) for i in range(4)]
    yield "i2t", build_messages(("query.jpg", None), ["cand %d" % i for i in range(13)], sup, con, "i2t"), 1 + 8
    yield "t2i", build_messages((None, "query caption"), ["cimg%d.jpg" % i for i in range(13)], sup, con, "t2i"), 8 + 13


@pytest.mark.parametrize("direction", ["i2t", "t2i"])
def test_messages_hide_the_aspect_and_label_13_candidates(direction):
    msgs, n_img = {d: (m, n) for d, m, n in _cases()}[direction]
    text = _render(msgs)
    for word in ("emotion", "style", "genre", "colour", "color"):
        assert word not in text.lower()
    assert all(f"\n{L}. " in text for L in LETTERS) and text.endswith("\nAnswer:")
    assert "Candidates:\n" in text and "one letter.\n\n" in text
    images = [p for m in msgs for p in m["content"] if p["type"] == "image"]
    assert len(images) == n_img      # i2t: query + 8 example images; t2i: 8 example images + 13 candidates, no query


def test_unpermute_returns_column_order():
    rng = np.random.default_rng(0)
    for _ in range(5):
        perm = rng.permutation(13)
        shown = perm.astype(float)                  # stub logit of letter j = the column it displays
        assert (unpermute(shown, perm) == np.arange(13)).all()
