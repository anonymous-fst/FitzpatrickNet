"""
fst.sampling
============
Sampling utilities shared by both training pipelines.

Classes
-------
BalancedBatchSampler
    Yields batches with an equal number of samples per class, cycling
    through per-class pools with replacement when a pool is exhausted.

Functions
---------
mixup_batch         Apply Mixup interpolation to a training batch.
get_prov_weights    Return per-sample provenance loss-weight arrays.
"""

import random

import numpy as np
import torch
import torch.utils.data
import pandas as pd


# ── Balanced batch sampler ────────────────────────────────────────────────────

class BalancedBatchSampler(torch.utils.data.Sampler):
    """
    Yield batches that contain an equal number of samples per class.

    Each batch contains ``batch_size // num_classes`` samples from every
    class.  If ``batch_size`` is not divisible by ``num_classes``, the
    remainder is drawn from a randomly chosen class.

    Parameters
    ----------
    labels      : (N,) int array of hard labels.
    batch_size  : total samples per batch.
    num_classes : number of distinct classes.
    """

    def __init__(self, labels: np.ndarray, batch_size: int, num_classes: int = 6):
        self.labels        = labels
        self.batch_size    = batch_size
        self.num_classes   = num_classes
        self.class_indices = {
            c: np.where(labels == c)[0].tolist()
            for c in range(num_classes)
        }
        self.n_batches = len(labels) // batch_size

    def __iter__(self):
        class_pools = {
            c: np.random.permutation(idxs).tolist()
            for c, idxs in self.class_indices.items()
            if idxs
        }
        active    = list(class_pools.keys())
        per_class = self.batch_size // self.num_classes
        remainder = self.batch_size  % self.num_classes

        for _ in range(self.n_batches):
            batch = []
            for c in active:
                if len(class_pools[c]) < per_class:
                    class_pools[c] = np.random.permutation(
                        self.class_indices[c]).tolist()
                batch.extend(class_pools[c][:per_class])
                class_pools[c] = class_pools[c][per_class:]
            if remainder:
                fc = random.choice(active)
                if len(class_pools[fc]) < remainder:
                    class_pools[fc] = np.random.permutation(
                        self.class_indices[fc]).tolist()
                batch.extend(class_pools[fc][:remainder])
                class_pools[fc] = class_pools[fc][remainder:]
            random.shuffle(batch)
            yield from batch

    def __len__(self) -> int:
        return self.n_batches * self.batch_size


# ── Mixup ─────────────────────────────────────────────────────────────────────

def mixup_batch(
    images:      torch.Tensor,
    soft_labels: torch.Tensor,
    hard_labels: torch.Tensor,
    prov_weights: torch.Tensor,
    alpha:       float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply Mixup to a training batch.

    A single mixing coefficient ``λ ~ Beta(α, α)`` is sampled and applied
    to images, soft labels, and provenance weights.  Hard labels are
    returned unchanged because they are used only for focal-weight lookup;
    the mixed soft labels carry the actual supervision signal.

    Parameters
    ----------
    images       : (B, C, H, W) image tensor.
    soft_labels  : (B, K) soft label tensor.
    hard_labels  : (B,)   integer hard label tensor (pass-through).
    prov_weights : (B,)   provenance weight tensor.
    alpha        : Beta distribution concentration parameter.

    Returns
    -------
    Tuple of (mixed_images, mixed_soft_labels, hard_labels, mixed_prov_weights).
    """
    lam = np.random.beta(alpha, alpha)
    B   = images.size(0)
    idx = torch.randperm(B, device=images.device)

    mixed_images      = lam * images       + (1 - lam) * images[idx]
    mixed_soft_labels = lam * soft_labels  + (1 - lam) * soft_labels[idx]
    mixed_prov        = lam * prov_weights + (1 - lam) * prov_weights[idx]

    return mixed_images, mixed_soft_labels, hard_labels, mixed_prov


# ── Provenance weight helper ──────────────────────────────────────────────────

def get_prov_weights(df: "pd.DataFrame", cfg: dict) -> np.ndarray:
    """
    Return per-sample provenance loss weights as a float32 array.

    Returns an all-ones array when ``use_provenance_weighting`` is False
    or when the provenance column is absent from *df*.

    Parameters
    ----------
    df  : DataFrame containing a provenance column (e.g. ``"dataset"``).
    cfg : experiment config dict.

    Returns
    -------
    (N,) float32 array.
    """
    N = len(df)
    if not cfg.get("use_provenance_weighting", False):
        return np.ones(N, dtype=np.float32)

    prov_col = cfg.get("provenance_col", "dataset")
    if prov_col not in df.columns:
        return np.ones(N, dtype=np.float32)

    prov_map = cfg.get("provenance_loss_weights", {})
    weights  = np.ones(N, dtype=np.float32)
    for i, ds in enumerate(df[prov_col].values):
        key = str(ds).strip().lower()
        if key in prov_map:
            weights[i] = float(prov_map[key])
    return weights
