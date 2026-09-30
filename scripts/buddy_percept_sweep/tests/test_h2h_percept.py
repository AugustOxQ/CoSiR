import importlib.util
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.buddy_percept_sweep import h2h_percept
from scripts.buddy_percept_sweep.h2h_percept import (
    PerceptStage1Config, fit_percept_stage1, load_percept_modules, n_initial_for, percept_constants,
    relabel_native,
)
from scripts.buddy_percept_sweep.h2h_store import H2HStore

_ROOT = Path(__file__).resolve().parents[3]
_STAGE2_FIXED = _ROOT / "src/test/20260922_percept_topic_pipeline/run_percept_stage2_fixed_pilot.py"
_SNAPSHOT_PILOT = _ROOT / "src/test/20260927_deep_stage_analysis/run_percept_fixed_snapshot_pilot.py"
_SYMSWEEP_PILOT = _ROOT / "src/test/20260927_deep_stage_analysis/run_percept_mapper_symmetric_sweep_pilot.py"


def test_n_initial_never_below_k():
    assert n_initial_for(PerceptStage1Config(n_initial_factor=1.0), 40) == 40
    assert n_initial_for(PerceptStage1Config(n_initial_factor=1.5), 40) == 60
    assert n_initial_for(PerceptStage1Config(n_initial_factor=2.5), 16) == 40


def test_constants_target_the_modules_that_read_them():
    c = percept_constants(PerceptStage1Config(), 40)
    # defaults reproduce the §6g fixed pilot: N 60 -> 40, lambda_balance 1000, lambda_recon 1
    assert c["s2"]["N_INITIAL_CLUSTERS"] == 60 and c["s2"]["N_SURVIVING_CLUSTERS"] == 40
    assert c["s2"]["LAMBDA_BALANCE"] == 1000.0 and c["s2"]["LAMBDA_RECONSTRUCTION"] == 1.0
    assert c["base"]["PRETRAIN_EPOCHS"] == 100 and c["base"]["PRETRAIN_LEARNING_RATE"] == 1e-3
    assert c["sweep"]["DEC_LEARNING_RATE"] == 1e-4 and c["sweep"]["STABILITY_THRESHOLD"] == 1e-3
    assert c["sweep"]["MAX_DEC_EPOCHS"] == 500


def test_defaults_equal_the_fixed_pilot_module_constants():
    """Every constant percept_constants sets, at the default config and K=40,
    equals the value the unmodified fixed pilot modules carry."""
    mods = load_percept_modules()
    targets = {"s2": mods.s2, "base": mods.s2.base, "sweep": mods.s2.sweep}
    for key, values in percept_constants(PerceptStage1Config(), 40).items():
        for name, value in values.items():
            assert getattr(targets[key], name) == value, (key, name)


def test_load_percept_modules_is_cached_and_consistent():
    mods = load_percept_modules()
    assert load_percept_modules() is mods
    assert mods.base is mods.s2.base and mods.sweep is mods.s2.sweep
    assert mods.s2.sweep.prune_centers is mods.s2.prune_centers_fixed   # the pilot's own fix is active


def test_relabel_native_maps_sorted_ids_and_unmapped_heldout_to_nearest_populated():
    train_topic = np.array([3, 1, 3, 0, 1])
    heldout_q = np.array([
        [0.10, 0.20, 0.05, 0.60, 0.05],   # argmax 3 -> label 2
        [0.30, 0.10, 0.40, 0.15, 0.05],   # argmax 2 (no train member) -> best populated 0 -> label 0
        [0.05, 0.25, 0.10, 0.20, 0.40],   # argmax 4 (no train member) -> best populated 1 -> label 1
        [0.70, 0.10, 0.10, 0.05, 0.05],   # argmax 0 -> label 0
    ])
    train_labels, heldout_native, label_ids, n_unmapped = relabel_native(train_topic, heldout_q.argmax(axis=1),
                                                                         heldout_q)
    assert label_ids.tolist() == [0, 1, 3]
    assert train_labels.tolist() == [2, 1, 2, 0, 1]
    assert heldout_native.tolist() == [2, 0, 1, 0]
    assert n_unmapped == 2


def test_k_target_below_one_raises():
    with pytest.raises(ValueError):
        fit_percept_stage1(PerceptStage1Config(), None, 0, 0, None, "cpu")


# ---------------------------------------------------------------- stubbed call-order / restore test

@pytest.fixture
def restore_torch_determinism():
    state = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
             torch.are_deterministic_algorithms_enabled(),
             torch.is_deterministic_algorithms_warn_only_enabled())
    yield state
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = state[0], state[1]
    torch.use_deterministic_algorithms(state[2], warn_only=state[3])


def _kernel_flags():
    return (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled())


def _stub_mods(calls, seen):
    """Module-shaped stubs: every constant starts at a sentinel; the fake
    affect-encoder loaders record the torch seed; build_autoencoder records
    what it sees and then fails."""
    class Tok:
        @staticmethod
        def from_pretrained(name):
            calls.append(("tokenizer", name))

    class Model:
        @staticmethod
        def from_pretrained(name):
            calls.append(("model", name, torch.initial_seed()))

    def build_autoencoder(input_dim):
        calls.append(("build", input_dim))
        seen["constants"] = {key: {name: getattr(mods[key], name) for name in values}
                             for key, values in percept_constants(PerceptStage1Config(), 40).items()}
        seen["seeds"] = {key: mods[key].SEED for key in mods}
        seen["kernels"] = _kernel_flags()
        raise RuntimeError("boom")

    sentinel = dict(N_INITIAL_CLUSTERS=-1, N_SURVIVING_CLUSTERS=-1, LAMBDA_BALANCE=-1.0,
                    LAMBDA_RECONSTRUCTION=-1.0, PRETRAIN_EPOCHS=-1, PRETRAIN_LEARNING_RATE=-1.0,
                    DEC_LEARNING_RATE=-1.0, STABILITY_THRESHOLD=-1.0, MAX_DEC_EPOCHS=-1, SEED=-1)
    base = SimpleNamespace(**sentinel, MODEL_NAME="stub-goemotions", AutoTokenizer=Tok, AutoModel=Model,
                           build_autoencoder=build_autoencoder)
    sweep = SimpleNamespace(**sentinel)
    s2 = SimpleNamespace(**sentinel, base=base, sweep=sweep)
    mods = {"s2": s2, "base": base, "sweep": sweep}
    return SimpleNamespace(s2=s2, base=base, sweep=sweep), sentinel


def _tiny_store(n_train=12, n_heldout=6, dim=5) -> H2HStore:
    rng = np.random.default_rng(0)
    obj = lambda values: np.array(list(values), dtype=object)
    return H2HStore(
        train_paintings=obj(f"t{i}" for i in range(n_train)), heldout_paintings=obj(f"h{i}" for i in range(n_heldout)),
        train_img=np.zeros((n_train, 2), np.float32), train_txt=np.zeros((n_train, 2), np.float32),
        heldout_img=np.zeros((n_heldout, 2), np.float32), heldout_txt=np.zeros((n_heldout, 2), np.float32),
        train_content_raw=np.zeros((n_train, 4)), heldout_content_raw=np.zeros((n_heldout, 4)),
        train_affect28=np.zeros((n_train, 28)), heldout_affect28=np.zeros((n_heldout, 28)),
        train_percept_h=rng.normal(size=(n_train, dim)).astype(np.float32),
        heldout_percept_h=rng.normal(size=(n_heldout, dim)).astype(np.float32),
        train_emotion=obj(["awe"] * n_train), heldout_emotion=obj(["awe"] * n_heldout),
        train_genre=obj([""] * n_train), heldout_genre=obj([""] * n_heldout),
        train_patches=torch.zeros(n_train, 1), heldout_patches=torch.zeros(n_heldout, 1),
    )


def test_port_seeds_replays_encoder_loads_pins_kernels_and_restores_on_failure(restore_torch_determinism):
    calls, seen = [], {}
    mods, sentinel = _stub_mods(calls, seen)
    torch.backends.cudnn.deterministic = True           # caller state the port must not inherit
    torch.use_deterministic_algorithms(True, warn_only=True)
    caller_flags = _kernel_flags()
    cfg = PerceptStage1Config()
    with pytest.raises(RuntimeError, match="boom"):
        fit_percept_stage1(cfg, _tiny_store(), 7, 40, mods, "cpu")

    # Knobs were on the modules that read them, SEED on every module, during the fit.
    assert seen["constants"] == percept_constants(cfg, 40)
    assert seen["seeds"] == {"s2": 7, "base": 7, "sweep": 7}
    # The fixed pilot sets no determinism flags: PyTorch defaults during the fit.
    assert seen["kernels"] == (False, False, False, False)
    # Seed first, then the pilot's two affect-encoder loads (train, held-out), then the autoencoder.
    assert calls == [("tokenizer", "stub-goemotions"), ("model", "stub-goemotions", 7),
                     ("tokenizer", "stub-goemotions"), ("model", "stub-goemotions", 7), ("build", 5)]
    # Everything restored after the failure.
    for module in (mods.s2, mods.base, mods.sweep):
        for name, value in sentinel.items():
            assert getattr(module, name) == value, name
    assert _kernel_flags() == caller_flags


# ---------------------------------------------------------------- pilot equivalence (synthetic, CPU)

_VOCAB = ("calm", "storm", "bright", "dark", "joy", "fear", "soft", "sharp", "old", "new", "sea", "sky")
_WORD_ID = {word: index + 1 for index, word in enumerate(_VOCAB)}   # 0 = padding / unknown
_EMOTIONS = ("awe", "fear", "contentment", "sadness")


class _Batch(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    """Stands in for AutoTokenizer (an external pretrained resource)."""

    @classmethod
    def from_pretrained(cls, name):
        return cls()

    def __call__(self, texts, padding, truncation, max_length, return_tensors):
        ids = [[_WORD_ID.get(word, 0) for word in text.split()][:max_length] for text in texts]
        width = max(len(row) for row in ids)
        input_ids = torch.zeros(len(ids), width, dtype=torch.long)
        attention_mask = torch.zeros(len(ids), width, dtype=torch.long)
        for row, token_ids in enumerate(ids):
            input_ids[row, :len(token_ids)] = torch.tensor(token_ids)
            attention_mask[row, :len(token_ids)] = 1
        return _Batch(input_ids=input_ids, attention_mask=attention_mask)


class _FakeAffectEncoder:
    """Stands in for the GoEmotions RobertaModel: fixed token embeddings (the
    pretrained weights, independent of any RNG), while loading draws from the
    global torch RNG like RobertaModel's newly initialized pooler does."""
    TABLE = torch.from_numpy(np.random.default_rng(5).normal(size=(len(_VOCAB) + 1, 16)).astype(np.float32))
    loads = 0

    @classmethod
    def from_pretrained(cls, name):
        _FakeAffectEncoder.loads += 1
        model = cls()
        model.pooler = torch.nn.Linear(16, 16)   # consumes the global CPU generator
        return model

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.TABLE[input_ids])


def _synthetic_split(rng, n, prefix):
    labels = rng.integers(0, 4, size=n)
    img = (rng.normal(size=(4, 24))[labels] + 0.6 * rng.normal(size=(n, 24))).astype(np.float32)
    txt = (rng.normal(size=(4, 24))[labels] + 0.6 * rng.normal(size=(n, 24))).astype(np.float32)
    paintings = [f"{prefix}{i:04d}" for i in range(n)]
    records, counts = [], []
    for painting, label in zip(paintings, labels):
        words = _VOCAB[3 * int(label):3 * int(label) + 3] + _VOCAB[:2]
        for c in range(1 + int(rng.integers(0, 3))):
            caption = " ".join(rng.choice(words, size=5))
            records.append({"painting": painting, "language": "english",
                            "caption": caption if c % 2 == 0 else [caption]})
        records.append({"painting": painting, "language": "arabic", "caption": "ignored"})
        counts.append(Counter({_EMOTIONS[int(label)]: 2, _EMOTIONS[(int(label) + 1) % 4]: 1}))
    return dict(paintings=paintings, img=img, txt=txt, records=records, counts=counts)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TEST_K = 4
# Case A: every value differs from the pilot modules' own constants, so a knob
# the port forgot to set would change the result; DEC stops on the stability
# criterion (epoch 13). Case B: DEC stops at the epoch ceiling and one
# surviving center has no train member, so held-out nodes take the fallback.
_CFG_A = PerceptStage1Config(pretrain_epochs=30, pretrain_lr=2e-3, dec_lr=3e-4, lambda_balance=50.0,
                             lambda_reconstruction=0.5, n_initial_factor=1.5, stability_threshold=5e-3,
                             max_dec_epochs=30)
_CFG_B = PerceptStage1Config(pretrain_epochs=40, pretrain_lr=1e-3, dec_lr=1e-4, lambda_balance=50.0,
                             lambda_reconstruction=0.5, n_initial_factor=1.5, stability_threshold=5e-3,
                             max_dec_epochs=5)


@pytest.fixture
def percept_on_synthetic_data(monkeypatch, tmp_path, request):
    """Fresh instances of the unmodified fixed pilot modules with only their
    data loading / IO replaced: synthetic 240-train / 120-held-out paintings,
    caption JSON files, fake pretrained affect encoder, synthetic patch cache.
    Runs on CPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    s2 = _load(f"s2_for_h2h_percept_test_{request.node.name}", _STAGE2_FIXED)
    base = s2.base
    pipeline = base.load_sibling_module("pipeline_h2h_percept_test_train", base.PIPELINE_PATH)
    heldout_pipeline = base.load_sibling_module("pipeline_h2h_percept_test_heldout", base.PIPELINE_PATH)
    affect_pilot = base.load_sibling_module("affect_h2h_percept_test", base.AFFECT_PILOT_PATH)
    cca_audit = base.load_sibling_module("cca_h2h_percept_test", base.CCA_AUDIT_PATH)

    rng = np.random.default_rng(0)
    data = {"train": _synthetic_split(rng, 240, "t"), "heldout": _synthetic_split(rng, 120, "h")}
    json_paths = {}
    for key, split in data.items():
        json_paths[key] = tmp_path / f"{key}.json"
        json_paths[key].write_text(json.dumps(split["records"]), encoding="utf-8")
    genres = ("portrait", "landscape", "abstract")
    genre_map = {p: genres[i % 3] for split in data.values() for i, p in enumerate(split["paintings"])}
    messages = []
    for pipe, key in ((pipeline, "train"), (heldout_pipeline, "heldout")):
        monkeypatch.setattr(pipe, "assert_extraction_complete", lambda: None)
        monkeypatch.setattr(pipe, "load_dedup_features", lambda key=key: (
            data[key]["paintings"], data[key]["img"], data[key]["txt"], data[key]["counts"]))
        monkeypatch.setattr(pipe, "load_genre_map", lambda: dict(genre_map))
        monkeypatch.setattr(pipe, "log", messages.append)
    monkeypatch.setattr(pipeline, "TRAIN_JSON", str(json_paths["train"]))
    monkeypatch.setattr(base, "HELDOUT_JSON", str(json_paths["heldout"]))
    monkeypatch.setattr(base, "AutoTokenizer", _FakeTokenizer)
    monkeypatch.setattr(base, "AutoModel", _FakeAffectEncoder)
    by_path = {base.AFFECT_PILOT_PATH: affect_pilot, base.CCA_AUDIT_PATH: cca_audit}

    def load_sibling_module(name, path):
        if path == base.PIPELINE_PATH:
            return heldout_pipeline if name.endswith("_heldout") else pipeline
        return by_path[path]

    monkeypatch.setattr(base, "load_sibling_module", load_sibling_module)
    patches = {"train": torch.from_numpy(rng.normal(size=(240, 50, 512)).astype(np.float32)),
               "held-out": torch.from_numpy(rng.normal(size=(120, 50, 512)).astype(np.float32))}
    monkeypatch.setattr(s2, "load_patch_features", lambda path, n, split_name: patches[split_name])

    # The store's PercepT inputs come from the pilot's own input functions,
    # exactly as h2h_store.build_arrays computes them.
    fused = {}
    for key, json_key in (("train", "train"), ("heldout", "heldout")):
        split = data[key]
        affect768 = base.extract_affect_embedding_nodes(str(json_paths[json_key]), split["paintings"], "cpu",
                                                        messages.append)
        fused[key] = base.fused_embeddings(split["img"], split["txt"], affect768, cca_audit, affect_pilot)
    obj = lambda values: np.array(list(values), dtype=object)
    tr, ho = data["train"], data["heldout"]
    store = H2HStore(
        train_paintings=obj(tr["paintings"]), heldout_paintings=obj(ho["paintings"]),
        train_img=tr["img"], train_txt=tr["txt"], heldout_img=ho["img"], heldout_txt=ho["txt"],
        train_content_raw=cca_audit.content_features(tr["img"], tr["txt"], affect_pilot),
        heldout_content_raw=cca_audit.content_features(ho["img"], ho["txt"], affect_pilot),
        train_affect28=np.zeros((240, 28)), heldout_affect28=np.zeros((120, 28)),
        train_percept_h=fused["train"], heldout_percept_h=fused["heldout"],
        train_emotion=obj(c.most_common(1)[0][0] for c in tr["counts"]),
        heldout_emotion=obj(c.most_common(1)[0][0] for c in ho["counts"]),
        train_genre=obj(genre_map[p] for p in tr["paintings"]),
        heldout_genre=obj(genre_map[p] for p in ho["paintings"]),
        train_patches=patches["train"], heldout_patches=patches["held-out"],
    )
    mods = SimpleNamespace(s2=s2, base=base, sweep=s2.sweep)
    return SimpleNamespace(mods=mods, store=store, messages=messages)


def _module_constants(mods):
    return {key: {name: getattr(module, name) for name in names}
            for key, module, names in (
                ("s2", mods.s2, ("N_INITIAL_CLUSTERS", "N_SURVIVING_CLUSTERS", "LAMBDA_BALANCE",
                                 "LAMBDA_RECONSTRUCTION", "SEED")),
                ("base", mods.base, ("PRETRAIN_EPOCHS", "PRETRAIN_LEARNING_RATE", "SEED")),
                ("sweep", mods.sweep, ("DEC_LEARNING_RATE", "STABILITY_THRESHOLD", "MAX_DEC_EPOCHS", "SEED")))}


def _run_port(env, seed, cfg=_CFG_A):
    """The port, starting from the pilot modules' own (non-test) constants."""
    before = _module_constants(env.mods)
    out = fit_percept_stage1(cfg, env.store, seed, _TEST_K, env.mods, "cpu")
    assert _module_constants(env.mods) == before
    return out


def _configure_pilot(env, seed, monkeypatch, cfg=_CFG_A):
    """Set the same knobs on the pilot modules, the way the pilot scripts read them."""
    mods = env.mods
    for key, values in percept_constants(cfg, _TEST_K).items():
        for name, value in values.items():
            monkeypatch.setattr({"s2": mods.s2, "base": mods.base, "sweep": mods.sweep}[key], name, value)
    monkeypatch.setattr(mods.s2, "SEED", seed)
    # Pilot's own definition MULTI_LABEL_THRESHOLD = 2 / N_SURVIVING_CLUSTERS at the tiny K.
    monkeypatch.setattr(mods.s2, "MULTI_LABEL_THRESHOLD", 2.0 / _TEST_K)
    # AttentionPoolingMapper's n_topics default is bound to N_SURVIVING_CLUSTERS at import.
    monkeypatch.setattr(mods.s2.AttentionPoolingMapper.__init__, "__defaults__", (512, _TEST_K))
    monkeypatch.setattr(mods.s2, "MAPPER_EPOCHS", 2)


def _pilot_dec_stop(messages):
    prefix = "Stopping K=6 DEC at epoch "
    stops = [m for m in messages if m.startswith(prefix)]
    assert len(stops) == 1
    epoch, reason = stops[0][len(prefix):-1].split(": ")
    return int(epoch), reason


@pytest.mark.parametrize("seed,cfg,stop_reason,expect_unmapped", [
    (42, _CFG_A, "stability criterion", False),
    (7, _CFG_B, "epoch ceiling", True),
])
def test_port_reproduces_fixed_snapshot_pilot_bit_for_bit(seed, cfg, stop_reason, expect_unmapped,
                                                          percept_on_synthetic_data, tmp_path, monkeypatch):
    env = percept_on_synthetic_data
    out = _run_port(env, seed, cfg)

    snap = _load(f"snapshot_pilot_for_h2h_percept_test_{seed}", _SNAPSHOT_PILOT)
    _configure_pilot(env, seed, monkeypatch, cfg)
    monkeypatch.setattr(snap, "load_module", lambda name, path: env.mods.s2)
    monkeypatch.setattr(snap, "SNAPSHOT_PATH", tmp_path / "snapshot.npz")
    monkeypatch.setattr(snap, "REPORT_PATH", tmp_path / "snapshot.md")
    monkeypatch.setattr(snap, "TOLERANCE", 1.0)   # the 0.5925 real-data sanity gate cannot apply here
    env.messages.clear()
    snap.main()
    with np.load(tmp_path / "snapshot.npz", allow_pickle=True) as npz:
        ref = {key: npz[key] for key in ("train_embedding", "heldout_embedding", "train_topic", "heldout_topic")}

    assert out.train_embedding.dtype == np.float32 and out.heldout_embedding.dtype == np.float32
    assert np.array_equal(out.train_embedding, ref["train_embedding"])
    assert np.array_equal(out.heldout_embedding, ref["heldout_embedding"])
    label_ids = out.info["label_ids"]
    assert np.array_equal(label_ids, np.unique(ref["train_topic"]))
    assert np.array_equal(label_ids[out.train_labels], ref["train_topic"])
    assert np.array_equal(out.train_labels, np.searchsorted(label_ids, ref["train_topic"]))
    mapped = np.isin(ref["heldout_topic"], label_ids)
    assert out.info["heldout_unmapped"] == int((~mapped).sum())
    assert np.array_equal(label_ids[out.heldout_native][mapped], ref["heldout_topic"][mapped])
    assert (out.info["heldout_unmapped"] > 0) == expect_unmapped
    if expect_unmapped:   # fallback = the best-scoring populated center under the pilot's soft assignment
        centers = torch.from_numpy(out.info["surviving_centers"])
        q = env.mods.base.soft_assignments(torch.from_numpy(ref["heldout_embedding"]), centers).numpy()
        assert np.array_equal(label_ids[out.heldout_native][~mapped], label_ids[q[~mapped][:, label_ids].argmax(1)])
    assert out.info["n_train_topics"] == len(label_ids) == len(np.unique(out.train_labels))
    assert out.info["n_initial"] == 6
    assert (out.info["dec_epochs"], out.info["dec_stop_reason"]) == _pilot_dec_stop(env.messages)
    assert out.info["dec_stop_reason"] == stop_reason and out.info["dec_epochs"] > 1
    assert out.info["seconds"] >= 0.0


def test_port_reproduces_symmetric_sweep_stage1_targets(percept_on_synthetic_data, monkeypatch):
    """Against the reference flow itself, fit_stage1_and_get_targets(s2): the
    port's latents + surviving centers rebuild the pilot's own multi-hot
    Stage-2 targets exactly (via the pilot's multi_hot_targets)."""
    env = percept_on_synthetic_data
    seed = 42
    out = _run_port(env, seed)

    sym = _load("symsweep_pilot_for_h2h_percept_test", _SYMSWEEP_PILOT)
    _configure_pilot(env, seed, monkeypatch)
    _device, _train_patches, train_targets, _heldout_patches, heldout_targets = sym.fit_stage1_and_get_targets(
        env.mods.s2)

    centers = torch.from_numpy(out.info["surviving_centers"])
    rebuilt = {split: env.mods.s2.multi_hot_targets(torch.nn.Identity(), centers, torch.from_numpy(embedding), "cpu")
               for split, embedding in (("train", out.train_embedding), ("heldout", out.heldout_embedding))}
    assert torch.equal(rebuilt["train"], train_targets)
    assert torch.equal(rebuilt["heldout"], heldout_targets)


def test_encoder_load_replay_is_what_makes_the_port_match(percept_on_synthetic_data, monkeypatch):
    """Sensitivity check: without replaying the pilot's two affect-encoder
    loads (which draw from the torch RNG), the autoencoder init differs."""
    env = percept_on_synthetic_data
    loads_before = _FakeAffectEncoder.loads
    faithful = _run_port(env, 42)
    assert _FakeAffectEncoder.loads - loads_before == 2
    monkeypatch.setattr(h2h_percept, "_replay_affect_encoder_loads", lambda base: None)
    without_replay = _run_port(env, 42)
    assert not np.array_equal(faithful.train_embedding, without_replay.train_embedding)
