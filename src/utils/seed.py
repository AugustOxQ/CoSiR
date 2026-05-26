import torch
import random
import numpy as np


def setup_seed(seed: int = 42):
    """Set global random seeds for reproducibility across Python, NumPy, and PyTorch."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
