"""Shared constants and result types for the matched-topic-count buddy vs
PercepT head-to-head harness."""
from dataclasses import dataclass
from typing import Optional

import numpy as np

EVAL_TRANSFER_K = 20
SEARCH_SEEDS = (1001, 1002)
STRESS_SEEDS = (42, 7, 123, 2024)
TEST_SEEDS = (11, 23, 57, 101, 211)


@dataclass
class Stage1Output:
    train_embedding: np.ndarray            # (N_train, D) float32
    heldout_embedding: np.ndarray          # (N_heldout, D) float32
    train_labels: Optional[np.ndarray]     # PercepT: native train topic ids; buddy: None
    heldout_native: Optional[np.ndarray]   # PercepT: native held-out topic ids; buddy: None
    info: dict                             # stop_reason, epochs_run, seconds, ...


@dataclass
class H2HSplit:
    val_idx: np.ndarray    # int64, sorted, indices into held-out paintings
    test_idx: np.ndarray   # int64, sorted, disjoint from val_idx, union = all
    digest: str            # sha1 hex of val_idx.tobytes() + test_idx.tobytes()
