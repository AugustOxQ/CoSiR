import torch
import random
import numpy as np


def setup_seed(seed: int = 42):
    """
    设置全局随机种子的工具函数。

    Args:
        seed (int): 用于设置随机种子的整数，默认为42。

    该函数会设置Python的random、numpy、torch（包括cuda）等常用库的随机种子，
    并设置torch.backends.cudnn.deterministic为True以保证实验的可复现性。
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
