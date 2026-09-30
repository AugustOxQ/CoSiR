"""Buddy Stage 1 for the matched-topic-count head-to-head (spec R7).

`impl="pilot"` is a line-by-line port of the frozen Attention-h1 snapshot
pilot, `src/test/20260923_artelingo_buddy_analysis/
run_attention_h1_embedding_snapshot_pilot.py::main` (lines 177-380): it calls
the pilot's own helpers on `PilotModules.arch` and applies knobs by setting
that module's constants, so with the default config and the full held-out set
as plateau monitor it reproduces the pilot's snapshot bit-for-bit (V2).
Comments cite the pilot line numbers each block ports.

`impl="harness"` is the §6i harness Stage 1 (`stage1.train_stage1`) fed from
the store.
"""
import os
import time
import weakref
from dataclasses import dataclass

# Pilot line 35: deterministic cuBLAS (the DAS6 wrappers export it too).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from scripts.buddy_percept_sweep.cache import FixedInputs  # noqa: E402
from scripts.buddy_percept_sweep.h2h_store import H2HStore  # noqa: E402
from scripts.buddy_percept_sweep.h2h_types import Stage1Output  # noqa: E402
from scripts.buddy_percept_sweep.pilot_metrics import PilotModules  # noqa: E402
from scripts.buddy_percept_sweep.stage1 import ParameterizedLearnedStudent, train_stage1  # noqa: E402

PILOT_HEADS = ("mlp128", "attn1", "attn4")
HARNESS_HEADS = ("mlp128", "attn1", "attn4", "attn")

# (tag, id(store), device) -> (weakref to store, content_edges, affect_edges).
# The weakref guards against a later store reusing a freed store's id.
_TEACHER_EDGES: dict = {}


@dataclass
class BuddyStage1Config:
    impl: str = "pilot"              # "pilot" | "harness"
    heads: str = "attn1"             # pilot: mlp128|attn1|attn4 ; harness: mlp128|attn1|attn4|attn
    num_heads: int = 1               # harness only
    d_shared: int = 32
    content_pca_dim: int = 50
    lr: float = 1e-3
    batch_size: int = 1024
    temperature: float = 0.1         # pilot only (arch.TEMPERATURE)
    max_epochs: int = 200
    plateau_window: int = 5          # pilot only
    plateau_rel_improvement: float = 0.01  # pilot only
    noise_std: float = 0.0           # harness only
    lambda_affect: float = 1.0       # harness only
    weight_decay: float = 0.0        # harness only
    teacher_graph_K: int = 20        # harness only


def pilot_constants(cfg: BuddyStage1Config) -> dict:
    """The arch-module constants (run_learned_student_arch_sweep_pilot.py
    names) that the pilot impl sets from `cfg`."""
    if cfg.impl != "pilot":
        raise ValueError(f"pilot_constants needs impl='pilot', got {cfg.impl!r}")
    if cfg.heads not in PILOT_HEADS:
        raise ValueError(f"pilot impl heads must be one of {PILOT_HEADS}, got {cfg.heads!r}")
    return {"D_SHARED": cfg.d_shared, "LEARNING_RATE": cfg.lr, "BATCH_SIZE": cfg.batch_size,
            "TEMPERATURE": cfg.temperature, "MAX_EPOCHS": cfg.max_epochs,
            "PLATEAU_WINDOW": cfg.plateau_window,
            "PLATEAU_REL_IMPROVEMENT": cfg.plateau_rel_improvement,
            "CONTENT_PCA_DIM": cfg.content_pca_dim}


def monitor_subsample(n_monitor: int, seed: int, edge_sample: int, rank_sample: int):
    """Pilot lines 318-324: recall-node draw, then effective-rank-node draw,
    from one generator; capped at the monitor size for subsets."""
    rng = np.random.default_rng(seed)
    sampled_nodes = rng.choice(n_monitor, size=min(edge_sample, n_monitor), replace=False)
    rank_nodes = rng.choice(n_monitor, size=min(rank_sample, n_monitor), replace=False)
    return sampled_nodes, rank_nodes


def _determinism_state() -> tuple:
    return (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled())


def _restore_determinism(state: tuple) -> None:
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = state[0], state[1]
    torch.use_deterministic_algorithms(state[2], warn_only=state[3])


def _pilot_teacher_edges(store: H2HStore, pilot: PilotModules, device: str):
    """Pilot lines 247-257. Independent of every knob and seed, so built
    once per store per process."""
    key = ("pilot_teacher", id(store), device)
    cached = _TEACHER_EDGES.get(key)
    if cached is not None and cached[0]() is store:
        return cached[1], cached[2]
    _img_graph, _txt_graph, content_teacher_graph = pilot.pipeline.build_buddy_graphs(
        store.train_img, store.train_txt, K=pilot.pipeline.K, alpha=pilot.pipeline.ALPHA, device=device,
        connect_components=True,
    )
    affect_teacher_graph = pilot.single_modality.build_single_modality_graph(
        "train-affect-teacher", store.train_affect28, pilot.pipeline, pilot.affect_pilot, device,
        expected_nodes=len(store.train_paintings),
    )
    content_edges = pilot.arch.upper_triangle_edges(content_teacher_graph)
    affect_edges = pilot.arch.upper_triangle_edges(affect_teacher_graph)
    for stale in [k for k, v in _TEACHER_EDGES.items() if v[0]() is None]:
        del _TEACHER_EDGES[stale]
    _TEACHER_EDGES[key] = (weakref.ref(store), content_edges, affect_edges)
    return content_edges, affect_edges


def _fit_pilot(cfg: BuddyStage1Config, store: H2HStore, seed: int, monitor_idx: np.ndarray,
               pilot: PilotModules, device_name: str) -> Stage1Output:
    constants = pilot_constants(cfg)
    arch = pilot.arch
    device = torch.device(device_name)   # the pilot passes `device` to torch and str(device) to graph helpers
    start = time.monotonic()
    saved_constants = {name: getattr(arch, name) for name in constants}
    saved_determinism = _determinism_state()
    try:
        # Lines 178-184.
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        for name, value in constants.items():
            setattr(arch, name, value)
        if pilot.cca_audit is not None:
            arch.cca_audit = pilot.cca_audit   # line 204 (evaluate_checkpoint reads it as a global)

        # Lines 240-245; content_features and the float64 affect are precomputed in the store.
        pca = PCA(n_components=arch.CONTENT_PCA_DIM, random_state=seed)
        content_train = pca.fit_transform(store.train_content_raw).astype(np.float32)
        content_heldout = pca.transform(store.heldout_content_raw).astype(np.float32)

        # Lines 247-258.
        content_edges, affect_edges = _pilot_teacher_edges(store, pilot, str(device))

        # Lines 260-268.
        train_content_t = torch.as_tensor(content_train, dtype=torch.float32, device=device)
        train_affect_t = torch.as_tensor(store.train_affect28, dtype=torch.float32, device=device)
        heldout_content_t = torch.as_tensor(content_heldout, dtype=torch.float32, device=device)
        heldout_affect_t = torch.as_tensor(store.heldout_affect28, dtype=torch.float32, device=device)

        torch.manual_seed(seed)
        np.random.seed(seed)
        model = arch.LearnedStudent(cfg.heads).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=arch.LEARNING_RATE)

        # Lines 270-313 (epoch-0 embeddings, Leiden and AMIs) are reporting only
        # and draw from no global RNG: omitted.

        # Lines 316-338, the plateau monitor restricted to held-out rows `monitor_idx`.
        trajectory = [{"epoch": 0, "content_recall": float("nan"), "affect_recall": float("nan")}]
        plateau_count, stop_reason = 0, f"reached MAX_EPOCHS={arch.MAX_EPOCHS}"
        sampled_nodes, rank_nodes = monitor_subsample(
            len(monitor_idx), seed, arch.EDGE_SAMPLE_SIZE, arch.EFFECTIVE_RANK_SAMPLE_SIZE)
        monitor_content = content_heldout[monitor_idx]
        monitor_affect = store.heldout_affect28[monitor_idx]
        monitor_content_graph = pilot.single_modality.build_single_modality_graph(
            "held-out-content-reference", monitor_content, pilot.heldout_pipeline, pilot.affect_pilot,
            str(device), expected_nodes=len(monitor_idx),
        )
        monitor_affect_graph = pilot.single_modality.build_single_modality_graph(
            "held-out-affect-reference", monitor_affect, pilot.heldout_pipeline, pilot.affect_pilot,
            str(device), expected_nodes=len(monitor_idx),
        )
        monitor_content_t = torch.as_tensor(monitor_content, dtype=torch.float32, device=device)
        monitor_affect_t = torch.as_tensor(monitor_affect, dtype=torch.float32, device=device)
        epoch_0_diag = arch.evaluate_checkpoint(
            model, monitor_content_t, monitor_affect_t, monitor_content_graph,
            monitor_affect_graph, sampled_nodes, rank_nodes, pilot.single_modality,
            pilot.heldout_pipeline, pilot.affect_pilot, str(device),
        )
        trajectory[0].update(epoch_0_diag)

        # Lines 339-380.
        epochs_run = 0
        for epoch in range(1, arch.MAX_EPOCHS + 1):
            epochs_run = epoch
            model.train()
            epoch_rng = np.random.default_rng(seed + epoch)
            content_pairs = arch.sample_positive_pairs(content_edges, epoch_rng)
            affect_pairs = arch.sample_positive_pairs(affect_edges, epoch_rng)
            content_embeddings, remapped_content_pairs = arch.content_batch_embeddings(
                model, train_content_t, train_affect_t, content_pairs, device
            )
            content_loss = arch.symmetric_infonce(content_embeddings, remapped_content_pairs, device)
            affect_embeddings, _mixing_weights = model(train_content_t, train_affect_t)
            affect_loss = arch.symmetric_infonce(affect_embeddings, affect_pairs, device)
            total_loss = content_loss + affect_loss
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
            if epoch % arch.CHECKPOINT_EVERY:
                continue
            diagnostics = arch.evaluate_checkpoint(
                model, monitor_content_t, monitor_affect_t, monitor_content_graph,
                monitor_affect_graph, sampled_nodes, rank_nodes, pilot.single_modality,
                pilot.heldout_pipeline, pilot.affect_pilot, str(device),
            )
            checkpoint = {"epoch": epoch, **diagnostics}
            trajectory.append(checkpoint)
            previous = trajectory[-2]
            content_plateau = (
                arch.relative_improvement(checkpoint["content_recall"], previous["content_recall"])
                < arch.PLATEAU_REL_IMPROVEMENT
            )
            affect_plateau = (
                arch.relative_improvement(checkpoint["affect_recall"], previous["affect_recall"])
                < arch.PLATEAU_REL_IMPROVEMENT
            )
            plateau_count = plateau_count + 1 if content_plateau and affect_plateau else 0
            if plateau_count >= arch.PLATEAU_WINDOW:
                stop_reason = f"both recalls plateaued for {arch.PLATEAU_WINDOW} consecutive checkpoints"
                break

        # Lines 384-389, held-out = all rows regardless of the monitor.
        model.eval()
        with torch.no_grad():
            train_embedding, _ = model(train_content_t, train_affect_t)
            heldout_embedding, _ = model(heldout_content_t, heldout_affect_t)
        train_embedding_np = train_embedding.cpu().numpy().astype(np.float32, copy=False)
        heldout_embedding_np = heldout_embedding.cpu().numpy().astype(np.float32, copy=False)
    finally:
        for name, value in saved_constants.items():
            setattr(arch, name, value)
        _restore_determinism(saved_determinism)
    return Stage1Output(train_embedding_np, heldout_embedding_np, None, None,
                        info={"stop_reason": stop_reason, "epochs_run": epochs_run,
                              "seconds": time.monotonic() - start, "trajectory": trajectory})


def _fit_harness(cfg: BuddyStage1Config, store: H2HStore, seed: int) -> Stage1Output:
    """The §6i Stage 1: `pipeline.run_trial` lines 65-87 on store inputs cast
    to float32 like `real_data.load_real_raw_inputs`, with the PCA seeded by
    the run seed instead of `cache.FixedInputCache`'s fixed 42."""
    if cfg.heads not in HARNESS_HEADS:
        raise ValueError(f"harness impl heads must be one of {HARNESS_HEADS}, got {cfg.heads!r}")
    start = time.monotonic()
    pca = PCA(n_components=cfg.content_pca_dim, random_state=seed)
    fixed_inputs = FixedInputs(
        train_content=pca.fit_transform(store.train_content_raw.astype(np.float32)).astype(np.float32),
        train_affect=store.train_affect28.astype(np.float32),
        heldout_content=pca.transform(store.heldout_content_raw.astype(np.float32)).astype(np.float32),
        heldout_affect=store.heldout_affect28.astype(np.float32),
        train_emotion=list(store.train_emotion), heldout_emotion=list(store.heldout_emotion),
        train_genre=store.train_genre, heldout_genre=store.heldout_genre,
        train_patches=store.train_patches, heldout_patches=store.heldout_patches,
        content_pca_dim=cfg.content_pca_dim,
    )
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # The harness student's attention variants differ only by num_heads.
    student = ParameterizedLearnedStudent(
        heads="mlp128" if cfg.heads == "mlp128" else "attn1", num_heads=cfg.num_heads,
        d_shared=cfg.d_shared, content_dim=fixed_inputs.train_content.shape[1],
        affect_dim=fixed_inputs.train_affect.shape[1],
    )
    train_embedding = train_stage1(
        student, fixed_inputs, lr=cfg.lr, noise_std=cfg.noise_std, lambda_affect=cfg.lambda_affect,
        batch_size=cfg.batch_size, weight_decay=cfg.weight_decay, seed=seed,
        max_epochs=cfg.max_epochs, teacher_graph_K=cfg.teacher_graph_K,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # train_stage1's device
    student.eval()
    with torch.no_grad():
        heldout_embedding, _ = student(
            torch.as_tensor(fixed_inputs.heldout_content, dtype=torch.float32, device=device),
            torch.as_tensor(fixed_inputs.heldout_affect, dtype=torch.float32, device=device),
        )
    return Stage1Output(np.asarray(train_embedding, dtype=np.float32),
                        heldout_embedding.cpu().numpy().astype(np.float32, copy=False), None, None,
                        info={"stop_reason": f"fixed schedule, max_epochs={cfg.max_epochs}",
                              "epochs_run": cfg.max_epochs, "seconds": time.monotonic() - start})


def fit_buddy_stage1(cfg: BuddyStage1Config, store: H2HStore, seed: int,
                     monitor_idx: np.ndarray, pilot: PilotModules, device: str) -> Stage1Output:
    """Buddy Stage 1 embeddings for train and ALL held-out paintings.

    `monitor_idx` (held-out rows the pilot's plateau rule watches) and
    `pilot` are used by impl="pilot" only; impl="harness" trains on
    `train_stage1`'s own device choice, as the §6i harness does."""
    if cfg.impl == "pilot":
        return _fit_pilot(cfg, store, seed, np.asarray(monitor_idx), pilot, device)
    if cfg.impl == "harness":
        return _fit_harness(cfg, store, seed)
    raise ValueError(f"impl must be 'pilot' or 'harness', got {cfg.impl!r}")
