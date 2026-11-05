"""
fst.reproducibility
===================
Single function that seeds every relevant random-number generator so that
experiments are fully reproducible across runs.
"""

import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Seed Python, NumPy, and PyTorch (CPU + all GPUs) for reproducibility.

    Also sets ``torch.backends.cudnn.deterministic = True`` and disables
    the non-deterministic cuDNN auto-tuner.  Note that deterministic mode
    may reduce throughput slightly on some hardware.

    Parameters
    ----------
    seed : int
        The integer seed value.  Per-fold training typically passes
        ``cfg["seed"] + fold`` so each fold uses a distinct but
        deterministic seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
