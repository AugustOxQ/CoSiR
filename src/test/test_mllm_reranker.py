from src.eval.mllm_reranker import LETTERS, build_messages


def test_messages_hide_the_aspect_and_label_13_candidates():
    sup = [("img_s%d.jpg" % i, "support caption %d" % i) for i in range(4)]
    con = [("img_c%d.jpg" % i, "contrast caption %d" % i) for i in range(4)]
    msgs = build_messages(("query.jpg", None), ["cand %d" % i for i in range(13)], sup, con, "i2t")
    text = " ".join(part.get("text", "") for m in msgs for part in m["content"] if part["type"] == "text")
    for word in ("emotion", "style", "genre", "colour", "color"):
        assert word not in text.lower()
    assert all(f"{L}." in text for L in LETTERS) and text.rstrip().endswith("Answer:")
    images = [p for m in msgs for p in m["content"] if p["type"] == "image"]
    assert len(images) == 1 + 4 + 4                                  # query + example images
