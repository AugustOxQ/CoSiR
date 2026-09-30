"""PercepT Stage 1 for the matched-topic-count head-to-head (spec R7).

A port of the §6g fixed pilot's Stage-1 flow,
`src/test/20260927_deep_stage_analysis/run_percept_mapper_symmetric_sweep_pilot.py
::fit_stage1_and_get_targets` (lines 58-98), with the native topic assignment
of `run_percept_fixed_snapshot_pilot.py::main` (lines 107-112). It calls the
pilots' own functions on the fixed Stage-2 pilot module `s2`
(`src/test/20260922_percept_topic_pipeline/run_percept_stage2_fixed_pilot.py`),
its `s2.base` (`run_percept_stage1_pilot.py`) and `s2.sweep`
(`run_percept_stage1_cluster_count_sweep_pilot.py`), and applies knobs by
setting the module constants those functions read at call time, restored
afterwards; the pilot files are never edited. With the default config,
k_target=40 and seed 42 it runs the §6g flow. Comments cite the lines each
block ports (`sym` = the symmetric-sweep pilot, `snap` = the fixed-snapshot
pilot).
"""
import importlib.util
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scripts.buddy_percept_sweep.h2h_store import H2HStore
from scripts.buddy_percept_sweep.h2h_types import Stage1Output

_FIXED_STAGE2_PATH = (Path(__file__).resolve().parents[2]
                      / "src/test/20260922_percept_topic_pipeline/run_percept_stage2_fixed_pilot.py")
_MODS = None

# cudnn.deterministic, cudnn.benchmark, deterministic algorithms, warn_only:
# the PyTorch defaults. Neither the fixed pilot nor the flows that drive it set
# any of them (sym lines 146-156 say so), so the refit runs on default kernels.
_PILOT_KERNELS = (False, False, False, False)


@dataclass
class PerceptStage1Config:
    pretrain_epochs: int = 100
    pretrain_lr: float = 1e-3
    dec_lr: float = 1e-4
    lambda_balance: float = 1000.0
    lambda_reconstruction: float = 1.0
    n_initial_factor: float = 1.5      # N_initial = max(k_target, round(factor * k_target))
    stability_threshold: float = 1e-3
    max_dec_epochs: int = 500


def load_percept_modules() -> SimpleNamespace:
    """The fixed Stage-2 pilot as `s2`, with `base = s2.base` and
    `sweep = s2.sweep` (the instances s2's functions read), loaded once per
    process. Loading it installs the pilot's own `sweep.prune_centers` fix."""
    global _MODS
    if _MODS is None:
        spec = importlib.util.spec_from_file_location("percept_stage2_fixed_for_h2h", str(_FIXED_STAGE2_PATH))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not import the fixed Stage-2 pilot from {_FIXED_STAGE2_PATH}")
        s2 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(s2)
        _MODS = SimpleNamespace(s2=s2, base=s2.base, sweep=s2.sweep)
    return _MODS


def n_initial_for(cfg: PerceptStage1Config, k_target: int) -> int:
    return max(k_target, int(round(cfg.n_initial_factor * k_target)))


def percept_constants(cfg: PerceptStage1Config, k_target: int) -> dict:
    """Each knob under the module whose global is read at call time:
    `s2.train_dec_until_stable_fixed` reads s2.LAMBDA_BALANCE and
    s2.LAMBDA_RECONSTRUCTION, and sweep.DEC_LEARNING_RATE, sweep.MAX_DEC_EPOCHS,
    sweep.STABILITY_THRESHOLD (s2 lines 106-156); `base.pretrain_autoencoder`
    reads base.PRETRAIN_EPOCHS and base.PRETRAIN_LEARNING_RATE (base lines
    151-173); the flow itself reads s2.N_INITIAL_CLUSTERS and
    s2.N_SURVIVING_CLUSTERS (sym lines 96-98)."""
    return {
        "s2": {"N_INITIAL_CLUSTERS": n_initial_for(cfg, k_target), "N_SURVIVING_CLUSTERS": k_target,
               "LAMBDA_BALANCE": cfg.lambda_balance, "LAMBDA_RECONSTRUCTION": cfg.lambda_reconstruction},
        "base": {"PRETRAIN_EPOCHS": cfg.pretrain_epochs, "PRETRAIN_LEARNING_RATE": cfg.pretrain_lr},
        "sweep": {"DEC_LEARNING_RATE": cfg.dec_lr, "MAX_DEC_EPOCHS": cfg.max_dec_epochs,
                  "STABILITY_THRESHOLD": cfg.stability_threshold},
    }


def relabel_native(train_topic: np.ndarray, heldout_topic: np.ndarray, heldout_q: np.ndarray):
    """Native train topic ids -> 0..n-1 by sorted unique id; the same map for
    held-out. A held-out node whose native (argmax) center has no train
    member takes the best-scoring center in `heldout_q` that has one: the
    nearest populated surviving center, since the DEC soft assignment
    decreases with latent distance.
    Returns (train_labels, heldout_native, label_ids, n_heldout_unmapped)."""
    label_ids = np.unique(train_topic)
    unmapped = ~np.isin(heldout_topic, label_ids)
    if unmapped.any():
        heldout_topic = heldout_topic.copy()
        heldout_topic[unmapped] = label_ids[heldout_q[np.ix_(unmapped, label_ids)].argmax(axis=1)]
    return (np.searchsorted(label_ids, train_topic), np.searchsorted(label_ids, heldout_topic),
            label_ids, int(unmapped.sum()))


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [percept-stage1] {message}", flush=True)


def _kernel_state() -> tuple:
    return (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled())


def _set_kernel_state(state: tuple) -> None:
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = state[0], state[1]
    torch.use_deterministic_algorithms(state[2], warn_only=state[3])


def _replay_affect_encoder_loads(base) -> None:
    """Sym lines 86-87 run `base.extract_affect_embedding_nodes` for train and
    held-out between the seeding and `build_autoencoder`. Each call loads
    AutoTokenizer and AutoModel(MODEL_NAME) (base lines 69-70); loading
    RobertaModel from the GoEmotions classification checkpoint newly
    initializes the missing pooler from the global torch RNG, which the
    autoencoder init and the pretraining permutations then draw from. The
    embeddings themselves come from the store; the two loads are replayed
    only for their RNG draws."""
    for _split in ("train", "held-out"):
        base.AutoTokenizer.from_pretrained(base.MODEL_NAME)
        base.AutoModel.from_pretrained(base.MODEL_NAME)


def fit_percept_stage1(cfg: PerceptStage1Config, store: H2HStore, seed: int, k_target: int,
                       mods: SimpleNamespace, device: str) -> Stage1Output:
    """PercepT latent embeddings for train and ALL held-out paintings, plus
    native topic labels (train relabeled 0..n-1, held-out under the same map)."""
    if k_target < 1:
        raise ValueError(f"k_target must be >= 1, got {k_target}")
    s2 = mods.s2
    base, sweep = s2.base, s2.sweep
    modules = {"s2": s2, "base": base, "sweep": sweep}
    constants = percept_constants(cfg, k_target)
    train_h, heldout_h = store.train_percept_h, store.heldout_percept_h
    if train_h.shape[1] != heldout_h.shape[1]:   # s2 lines 361-362
        raise RuntimeError(f"Train/held-out fused dimensions differ: {train_h.shape[1]} vs {heldout_h.shape[1]}.")

    start = time.monotonic()
    saved = [(modules[key], name, getattr(modules[key], name)) for key, values in constants.items() for name in values]
    saved += [(module, "SEED", module.SEED) for module in modules.values()]
    saved_kernels = _kernel_state()
    try:
        for key, values in constants.items():
            for name, value in values.items():
                setattr(modules[key], name, value)
        for module in modules.values():
            module.SEED = seed
        _set_kernel_state(_PILOT_KERNELS)

        # Sym lines 64-67.
        np.random.seed(s2.SEED)
        torch.manual_seed(s2.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s2.SEED)
        # Sym lines 69-89: data loading and input fusion (precomputed in the
        # store) plus the two affect-encoder loads' RNG draws.
        _replay_affect_encoder_loads(base)

        # Sym lines 91-98.
        train_inputs = torch.from_numpy(train_h)
        encoder, decoder = base.build_autoencoder(train_h.shape[1])
        encoder.to(device)
        decoder.to(device)
        base.pretrain_autoencoder(encoder, decoder, train_inputs, device, _log)
        centers = sweep.initialize_cluster_centers(encoder, train_inputs, device, s2.N_INITIAL_CLUSTERS, s2.SEED)
        _dec_losses, dec_stop_reason, dec_epochs = s2.train_dec_until_stable_fixed(
            encoder, decoder, centers, train_inputs, device, _log, s2.N_INITIAL_CLUSTERS)
        surviving_centers, _surviving_indices = s2.prune_centers_fixed(centers, s2.N_SURVIVING_CLUSTERS)

        # Snap lines 107-112 (the held-out soft assignment is kept for the
        # unmapped-topic fallback; its argmax is the pilot's heldout_topic).
        encoder.eval()
        with torch.no_grad():
            train_latent = encoder(train_inputs.to(device))
            train_topic = base.soft_assignments(train_latent, surviving_centers).argmax(dim=1).cpu().numpy()
            heldout_latent = encoder(torch.from_numpy(heldout_h).to(device))
            heldout_q = base.soft_assignments(heldout_latent, surviving_centers)
            heldout_topic = heldout_q.argmax(dim=1).cpu().numpy()
            heldout_q = heldout_q.cpu().numpy()
        n_initial = s2.N_INITIAL_CLUSTERS
        # Snap lines 164/169.
        train_embedding = train_latent.detach().cpu().numpy().astype(np.float32)
        heldout_embedding = heldout_latent.detach().cpu().numpy().astype(np.float32)
        surviving_centers_np = surviving_centers.cpu().numpy().astype(np.float32)
    finally:
        for module, name, value in reversed(saved):
            setattr(module, name, value)
        _set_kernel_state(saved_kernels)

    train_labels, heldout_native, label_ids, n_unmapped = relabel_native(train_topic, heldout_topic, heldout_q)
    return Stage1Output(
        train_embedding, heldout_embedding, train_labels, heldout_native,
        info={"n_initial": n_initial, "dec_epochs": dec_epochs, "dec_stop_reason": dec_stop_reason,
              "n_train_topics": int(len(label_ids)), "heldout_unmapped": n_unmapped,
              "seconds": time.monotonic() - start,
              # Row j = the pilot's native topic j (its multi-hot target column);
              # relabeled topic i is pilot topic label_ids[i].
              "surviving_centers": surviving_centers_np, "label_ids": label_ids},
    )
