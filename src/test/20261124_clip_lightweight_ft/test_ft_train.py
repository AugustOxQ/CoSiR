"""Unit tests for ft_train: tiny CPU inputs, real CLIP ViT-B/32 weights from the local HF cache (offline)."""
import copy
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ft_data  # noqa: E402
import ft_train  # noqa: E402

TOWERS = ("vision_model", "text_model")


@pytest.fixture(scope="module")
def clip():
    return ft_train.load_clip()


@pytest.fixture(scope="module")
def tokenizer():
    return ft_train.load_tokenizer()


# --------------------------------------------------------------------------------------------- sampler
def _positions(n_paint, max_caps, seed):
    """Painting position of each toy train row: many captions per painting, rows in shuffled order."""
    rng = np.random.default_rng(seed)
    pos = np.repeat(np.arange(n_paint), rng.integers(1, max_caps + 1, size=n_paint))
    return pos[rng.permutation(len(pos))]


@pytest.mark.parametrize("n_paint,max_caps,bs", [(40, 15, 8), (600, 20, 256)])
def test_sampler_one_caption_per_painting_per_epoch(n_paint, max_caps, bs):
    pos = _positions(n_paint, max_caps, seed=n_paint)
    drawn, orders = set(), []
    for epoch in range(1, 201):
        order = ft_train.epoch_order(pos, epoch)
        assert order.dtype == np.int64 and len(order) == n_paint
        assert sorted(pos[order].tolist()) == list(range(n_paint))  # each painting exactly once per epoch
        chunks = ft_train.batches(order, bs)
        assert np.array_equal(np.concatenate(chunks), order)
        assert [len(c) for c in chunks[:-1]] == [bs] * (len(chunks) - 1)
        for b in chunks:
            assert len(np.unique(pos[b])) == len(b)  # no painting twice in a batch
        assert np.array_equal(order, ft_train.epoch_order(pos, epoch))  # deterministic for (seed 0, epoch)
        orders.append(order)
        drawn.update(order.tolist())
    assert not np.array_equal(orders[0], orders[1])  # reshuffled every epoch
    assert not np.array_equal(ft_train.epoch_order(pos, 1, seed=1), orders[0])  # the seed enters
    assert drawn == set(range(len(pos)))  # over many epochs every caption is drawn


# --------------------------------------------------------------------------------------------- variants
def _lb_expected():
    names = {"logit_scale", "vision_model.post_layernorm.weight", "vision_model.post_layernorm.bias",
             "text_model.final_layer_norm.weight", "text_model.final_layer_norm.bias",
             "visual_projection.weight", "text_projection.weight"}
    for t in TOWERS:
        for sub in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj",
                    "layer_norm1", "mlp.fc1", "mlp.fc2", "layer_norm2"):
            for w in ("weight", "bias"):
                names.add(f"{t}.encoder.layers.11.{sub}.{w}")
    return names


def _lora_expected():
    return {"logit_scale"} | {f"{t}.encoder.layers.{i}.self_attn.{p}_proj.lora_{ab}.default.weight"
                              for t in TOWERS for i in range(12) for p in ("q", "k", "v", "out") for ab in "AB"}


def _groups(model, variant):
    return {g["weight_decay"]: set(g["names"]) for g in ft_train.make_optimizer(model, variant, 1e-4).param_groups}


def test_lp_identity_at_step_0(clip):
    lp = ft_train.build_model("LP", clip)
    x = torch.randn(32, 512)
    assert torch.equal(lp.encode_image(x), x) and torch.equal(lp.encode_text(x), x)
    assert lp.logit_scale.item() == clip.logit_scale.item()
    assert ft_train.trainable_names(lp) == {"img_map.weight", "txt_map.weight", "logit_scale"}
    assert _groups(lp, "LP") == {0.0: {"img_map.weight", "txt_map.weight", "logit_scale"}}
    # through the row path: frozen cached features come back bit for bit
    data = SimpleNamespace(paintings=np.array(["b", "a", "b", "c"]), img_features=np.random.randn(4, 512).astype(np.float32),
                           txt_features=np.random.randn(4, 512).astype(np.float32), sample_ids=np.array([3, 1, 0, 2]))
    rs = ft_train.make_rowset("LP", data, np.array([3, 0, 2]))
    img, txt = ft_train.encode_rowset(lp, "LP", rs, torch.device("cpu"))
    assert rs.rows.tolist() == [0, 2, 3]
    assert np.array_equal(img.numpy(), data.img_features[[0, 2, 3]])
    assert np.array_equal(txt.numpy(), data.txt_features[[0, 2, 3]])


def test_lb_trainable_set(clip):
    m = ft_train.build_model("LB", copy.deepcopy(clip))
    assert ft_train.trainable_names(m) == _lb_expected()  # 39 tensors
    decay = {n for n in _lb_expected() if n.endswith(".weight") and "norm" not in n}
    assert len(decay) == 14
    assert _groups(m, "LB") == {0.1: decay, 0.0: _lb_expected() - decay}
    assert ft_train.n_trainable(m) == 10_898_177


def test_lora_trainable_set(clip):
    m = ft_train.build_model("LoRA", copy.deepcopy(clip))
    assert ft_train.trainable_names(m) == _lora_expected()  # 192 adapters + logit_scale
    assert _groups(m, "LoRA") == {0.0: _lora_expected()}
    params = dict(m.named_parameters())
    for t, width in (("vision_model", 768), ("text_model", 512)):
        q = m.get_submodule(f"{t}.encoder.layers.5.self_attn.q_proj")
        assert q.r["default"] == 16 and q.scaling["default"] == 2.0 and q.lora_dropout["default"].p == 0.05
        assert params[f"{t}.encoder.layers.5.self_attn.q_proj.lora_A.default.weight"].shape == (16, width)
        assert not params[f"{t}.encoder.layers.5.self_attn.q_proj.lora_B.default.weight"].any()  # B = 0: plain CLIP at step 0
    assert ft_train.n_trainable(m) == 1_966_081


def _toy_batch(variant, tokenizer, n=8):
    g = torch.Generator().manual_seed(0)
    if variant == "LP":
        return torch.randn(n, 512, generator=g), torch.randn(n, 512, generator=g)
    colours = torch.randint(0, 256, (n, 1, 1, 3), generator=g, dtype=torch.uint8)
    noise = torch.randint(0, 40, (n, 224, 224, 3), generator=g, dtype=torch.uint8)
    u8 = (colours.int() + noise.int()).clamp(0, 255).to(torch.uint8)
    caps = ["a red sunset over the sea", "a portrait of an old man", "a bowl of fruit on a table", "a stormy night",
            "a quiet village in winter", "an abstract composition of lines", "a bright field of flowers", "a dark forest"]
    enc = tokenizer(caps[:n], return_tensors="pt", **ft_train.TOKENS)
    return u8, enc["input_ids"], enc["attention_mask"]


@pytest.mark.parametrize("variant,lr", [("LP", 1e-3), ("LB", 1e-5), ("LoRA", 1e-4)])
def test_one_step_lowers_loss(clip, tokenizer, variant, lr):
    torch.manual_seed(0)
    m = ft_train.build_model(variant, copy.deepcopy(clip))
    batch = _toy_batch(variant, tokenizer)
    dev = torch.device("cpu")
    frozen = {"LP": [],
              "LB": ["vision_model.encoder.layers.10.mlp.fc1.weight", "text_model.embeddings.token_embedding.weight",
                     "text_model.encoder.layers.10.self_attn.q_proj.weight"],
              "LoRA": ["vision_model.encoder.layers.11.mlp.fc1.weight", "text_model.embeddings.token_embedding.weight",
                       "vision_model.encoder.layers.11.self_attn.q_proj.base_layer.weight",
                       "visual_projection.weight"]}[variant]
    params = dict(m.named_parameters())
    before_frozen = {n: params[n].detach().clone() for n in frozen}
    before_train = {n: p.detach().clone() for n, p in params.items() if p.requires_grad}
    opt = ft_train.make_optimizer(m, variant, lr)
    loss0 = ft_train.batch_loss(m, variant, batch, dev)
    step_loss = ft_train.train_step(m, variant, batch, opt, None, dev)
    loss1 = ft_train.batch_loss(m, variant, batch, dev)
    assert math.isfinite(step_loss) and loss1 < loss0
    for n in frozen:
        assert torch.equal(params[n], before_frozen[n]), n
    moved = {n for n in before_train if not torch.equal(params[n], before_train[n])}
    if variant == "LoRA":  # B starts at 0, so A's first gradient is exactly 0: only B moves on step 1
        assert moved | {"logit_scale"} == {n for n in before_train if "lora_B" in n} | {"logit_scale"}
    else:
        assert moved | {"logit_scale"} == set(before_train)
    assert m.logit_scale.item() <= math.log(100)


def test_logit_scale_clamped_at_log_100(clip):
    m = ft_train.build_model("LP", clip)
    with torch.no_grad():
        m.logit_scale.fill_(6.0)
    ft_train.train_step(m, "LP", _toy_batch("LP", None), ft_train.make_optimizer(m, "LP", 1e-3), None,
                        torch.device("cpu"))
    assert m.logit_scale.item() == pytest.approx(math.log(100), abs=1e-6)


def test_schedule_warmup_then_cosine():
    warmup, f = ft_train.schedule(1430)
    assert warmup == 72
    assert f(0) == pytest.approx(1 / 72) and f(71) == pytest.approx(1.0) and f(72) == pytest.approx(1.0)
    vals = [f(s) for s in range(72, 1430)]
    assert all(a >= b for a, b in zip(vals, vals[1:])) and 0 < vals[-1] < 1e-4
    assert ft_train.schedule(2)[0] == 1


# --------------------------------------------------------------------------------------------- retrieval and selection
def test_retrieval_on_hand_made_similarity():
    cap_img = np.array([0, 0, 1, 1, 2, 2])  # caption j belongs to image cap_img[j]
    # Every tie resolves correctly under "lowest index wins" and wrongly under "highest index wins".
    sim = torch.tensor([[0.9, 0.1, 0.9, 0.0, 0.3, 0.0],   # i0: tie c0 (own) / c2 -> c0: correct
                        [0.0, 0.1, 0.7, 0.2, 0.7, 0.1],   # i1: tie c2 (own) / c4 -> c2: correct
                        [0.0, 0.0, 0.0, 0.0, 0.3, 0.8]])  # i2: c5 (own): correct
    # captions -> images: c0 i0 ok; c1 tie i0 (own) / i1 -> i0 ok; c2 i0 wrong; c3 i1 ok; c4 i1 wrong; c5 i2 ok
    for chunk in (1, 2, 4, 100):
        r = ft_train.r1_from_sim(sim, cap_img, chunk=chunk)
        assert (r["i2t_correct"], r["t2i_correct"], r["n_images"], r["n_captions"]) == (3, 4, 3, 6)
        assert r["i2t_r1"] == 1.0 and r["t2i_r1"] == 4 / 6 and r["selection"] == (1.0 + 4 / 6) / 2


def test_retrieval_from_features_matches_similarity():
    g = torch.Generator().manual_seed(1)
    img, txt = torch.randn(7, 16, generator=g), torch.randn(30, 16, generator=g)
    cap_img = np.random.default_rng(0).integers(0, 7, 30)
    sim = torch.nn.functional.normalize(img, dim=-1) @ torch.nn.functional.normalize(txt, dim=-1).T
    ref = ft_train.r1_from_sim(sim, cap_img, chunk=1000)
    for chunk in (3, 30):
        assert ft_train.r1_from_features(img, txt, cap_img, chunk=chunk) == ref


def test_best_epoch_and_cross_run_selection():
    ep = lambda e, s: {"epoch": e, "selection": s}  # noqa: E731
    assert ft_train.best_epoch([ep(0, 0.9), ep(1, 0.5), ep(2, 0.6), ep(3, 0.6)]) == 2  # epoch 0 is reference only
    runs = [{"lr": 3e-4, "epochs": [ep(0, 0.9), ep(1, 0.5), ep(2, 0.6)]},
            {"lr": 1e-4, "epochs": [ep(0, 0.9), ep(1, 0.6), ep(2, 0.6)]},
            {"lr": 1e-3, "epochs": [ep(0, 0.9), ep(1, 0.55), ep(2, 0.6)]}]
    assert ft_train.select_runs(runs) == {"lr": 1e-4, "epoch": 1, "selection": 0.6}
    runs[0]["epochs"][1]["selection"] = 0.61
    assert ft_train.select_runs(runs) == {"lr": 3e-4, "epoch": 1, "selection": 0.61}


# --------------------------------------------------------------------------------------------- rows
def test_check_rows_rejects_held():
    idx = {"scorer_train": np.array([0, 1]), "val": np.array([2]), "selection": np.array([3])}
    ft_train.check_rows(np.array([3, 0, 2]), idx)
    with pytest.raises(ValueError, match="held"):
        ft_train.check_rows(np.array([0, 4]), idx)


def test_smoke_subset():
    paintings = np.array([f"p{i % 50:02d}" for i in range(400)])
    idx = {"scorer_train": np.arange(0, 300), "val": np.arange(300, 350), "selection": np.arange(350, 400)}
    s = ft_train.smoke_subset(idx, paintings, n_paint=10, n_val=5, n_sel=4)
    assert np.unique(paintings[s["scorer_train"]]).tolist() == [f"p{i:02d}" for i in range(10)]
    assert set(s["scorer_train"].tolist()) == {r for r in range(300) if r % 50 < 10}  # all rows of the 10 paintings
    assert s["val"].tolist() == [300, 301, 302, 303, 304] and s["selection"].tolist() == [350, 351, 352, 353]
    avail = {f"p{i:02d}" for i in range(20, 50)}
    s = ft_train.smoke_subset(idx, paintings, available=avail, n_paint=3, n_val=2, n_sel=2)
    assert set(paintings[np.concatenate(list(s.values()))]) <= avail
    assert np.unique(paintings[s["scorer_train"]]).tolist() == ["p20", "p21", "p22"]


# --------------------------------------------------------------------------------------------- end to end
class _Held(dict):
    """A held row's annotation: any access fails the test."""

    def __getitem__(self, key):
        raise AssertionError("a held annotation was read")


def _toy_run_data(tmp_path):
    """32 feature rows over 10 paintings with shuffled sample_ids; p09 is held (no image file, poisoned
    annotations, NaN features)."""
    caps_per = [4, 3, 3, 4, 3, 3, 3, 3, 3, 3]
    split_of = ["scorer_train"] * 5 + ["val"] * 2 + ["selection"] * 2 + ["held"]
    wiki = tmp_path / "wikiart"
    (wiki / "S").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for k in range(9):
        arr = (rng.integers(0, 256, (1, 1, 3)) + rng.integers(0, 60, (48, 64, 3))).clip(0, 255).astype(np.uint8)
        Image.fromarray(arr).save(wiki / "S" / f"p{k:02d}.jpg", quality=95)
    ann_paint = [k for k, c in enumerate(caps_per) for _ in range(c)]  # annotation order
    words = ["red", "blue", "green", "gold", "grey", "pink", "black", "white"]
    ann = []
    for a, k in enumerate(ann_paint):
        rec = {"image": f"S/p{k:02d}.jpg", "caption": f"caption {a}: a {words[a % 8]} painting number {k}",
               "painting": f"p{k:02d}", "emotion": "awe", "art_style": "S"}
        ann.append(_Held(rec) if split_of[k] == "held" else rec)
    sample_ids = rng.permutation(len(ann))
    paintings = np.array([f"p{ann_paint[s]:02d}" for s in sample_ids])
    split_row = np.array([split_of[ann_paint[s]] for s in sample_ids])
    img = rng.standard_normal((len(ann), 512)).astype(np.float32)
    txt = rng.standard_normal((len(ann), 512)).astype(np.float32)
    img[split_row == "held"] = np.nan
    txt[split_row == "held"] = np.nan
    data = SimpleNamespace(sample_ids=sample_ids, paintings=paintings, img_features=img, txt_features=txt)
    splits = SimpleNamespace(**{k: np.flatnonzero(split_row == k) for k in ("scorer_train", "val", "selection", "held")})
    cache = tmp_path / "cache"
    ft_data.build_image_cache(cache, data, ann, wiki, splits=splits, workers=1, verbose=False)
    return data, ann, splits, cache


@pytest.mark.parametrize("variant,lr", [("LP", 1e-3), ("LB", 1e-5), ("LoRA", 1e-4)])
def test_run_outputs_and_row_order(tmp_path, clip, tokenizer, variant, lr):
    data, ann, splits, cache = _toy_run_data(tmp_path)
    out = tmp_path / "out"
    args = ft_train.parse_args(["--variant", variant, "--lr", str(lr), "--epochs", "2", "--out", str(out),
                                "--device", "cpu", "--workers", "0"])
    ft_train.run(args, data=data, annotations=ann, splits=splits, cache_dir=cache, clip=copy.deepcopy(clip),
                 verbose=False)
    assert sorted(p.name for p in out.iterdir()) == sorted(ft_train.OUT_FILES)

    met = json.loads((out / "metrics.json").read_text())
    assert [e["epoch"] for e in met["epochs"]] == [0, 1, 2]
    for e in met["epochs"]:
        assert {"i2t_r1", "t2i_r1", "selection", "i2t_correct", "t2i_correct", "train_loss", "logit_scale"} <= set(e)
        assert e["n_images"] == 2 and e["n_captions"] == 6  # val: p05, p06 with 3 captions each
    assert met["epochs"][0]["train_loss"] is None
    assert met["best_epoch"] == ft_train.best_epoch(met["epochs"]) and met["best_epoch"] >= 1
    assert met["lr"] == lr and met["variant"] == variant

    best = torch.load(out / "best_params.pt")
    model = ft_train.build_model(variant, copy.deepcopy(clip))
    assert set(best["params"]) == ft_train.trainable_names(model) and best["epoch"] == met["best_epoch"]
    ft_train.load_trainable(model, best["params"])
    model.eval()

    z = np.load(out / "features.npz")
    assert set(z.files) == {"rows", "img", "txt"}
    want_rows = np.sort(np.concatenate([splits.val, splits.selection]))
    assert z["rows"].dtype == np.int64 and np.array_equal(z["rows"], want_rows)
    assert z["img"].dtype == z["txt"].dtype == np.float32 and z["img"].shape == z["txt"].shape == (12, 512)
    assert np.isfinite(z["img"]).all() and np.isfinite(z["txt"]).all()

    # Row alignment, computed independently: row r = the caption of annotations[sample_ids[r]] + its painting's image
    rows = z["rows"]
    with torch.no_grad():
        if variant == "LP":
            exp_img = torch.from_numpy(data.img_features[rows]) @ model.img_map.weight.T
            exp_txt = torch.from_numpy(data.txt_features[rows]) @ model.txt_map.weight.T
        else:
            images, index, _ = ft_data.load_image_cache(cache)
            u8 = torch.from_numpy(np.stack([images[index[data.paintings[r]]] for r in rows]))
            mean = torch.tensor(ft_data.CLIP_MEAN).view(1, 3, 1, 1)
            std = torch.tensor(ft_data.CLIP_STD).view(1, 3, 1, 1)
            px = (u8.permute(0, 3, 1, 2).float() / 255.0 - mean) / std
            enc = tokenizer([ann[int(data.sample_ids[r])]["caption"] for r in rows], return_tensors="pt",
                            max_length=77, padding="max_length", truncation=True)
            exp_img = model.visual_projection(model.vision_model(pixel_values=px).pooler_output)
            exp_txt = model.text_projection(model.text_model(input_ids=enc["input_ids"],
                                                             attention_mask=enc["attention_mask"]).pooler_output)
    np.testing.assert_allclose(z["img"], exp_img.numpy(), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(z["txt"], exp_txt.numpy(), atol=1e-4, rtol=1e-4)
    assert np.abs(z["txt"] - exp_txt.numpy()[::-1]).max() > 1e-2  # a misaligned order would fail

    rec = json.loads((out / "run_record.json").read_text())
    for key in ("args", "git_commit", "hostname", "gpu_name", "versions", "start_time", "end_time", "best_epoch",
                "best_selection", "n_trainable_params", "data", "training", "checks"):
        assert key in rec, key
    assert set(rec["versions"]) >= {"torch", "transformers", "peft"}
    assert rec["best_epoch"] == met["best_epoch"] and rec["n_trainable_params"] == ft_train.n_trainable(model)
    assert rec["checks"]["reload"]["ok"] is True
    assert rec["data"]["n_train_paintings"] == 5 and rec["training"]["steps_per_epoch"] == 1
