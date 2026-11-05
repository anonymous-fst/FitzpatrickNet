"""
fst.soft_labels
===============
Triangular soft-label generation for ordinal FST classification.

The triangular scheme places mass ``1 - 2α`` on the true class and ``α``
on each immediate neighbour, with boundary classes absorbing the
neighbour mass that would fall outside [0, K-1].  Labels are then
row-normalised to sum to 1.

Public API
----------
make_soft_labels            Uniform alpha across all samples
make_soft_labels_adaptive   Per-sample alpha looked up from a dataset map
anneal_alpha                Linear alpha schedule over training epochs
build_soft_labels           Dispatcher used by dataloaders (reads cfg)
"""

import numpy as np
import pandas as pd


# ── Core label constructors ───────────────────────────────────────────────────

def make_soft_labels(
    hard_labels: np.ndarray,
    num_classes: int,
    alpha: float = 0.15,
) -> np.ndarray:
    """
    Build triangular soft labels with a uniform smoothing parameter.

    Parameters
    ----------
    hard_labels : (N,) int array
        Ground-truth class indices in [0, num_classes - 1].
    num_classes : int
        Total number of classes (6 for FST I–VI).
    alpha : float
        Smoothing mass placed on each immediate neighbour.
        Must satisfy ``0 < alpha < 0.5``.

    Returns
    -------
    soft : (N, num_classes) float32 array, rows sum to 1.
    """
    N           = len(hard_labels)
    soft        = np.zeros((N, num_classes), dtype=np.float32)
    hard_labels = np.array(hard_labels, dtype=np.int64)

    for i, c in enumerate(hard_labels):
        soft[i, c] = 1.0 - 2 * alpha
        if c > 0:
            soft[i, c - 1] = alpha
        else:
            soft[i, c] += alpha                 # absorb left-boundary mass
        if c < num_classes - 1:
            soft[i, c + 1] = alpha
        else:
            soft[i, c] += alpha                 # absorb right-boundary mass

    return (soft / soft.sum(axis=1, keepdims=True)).astype(np.float32)


def make_soft_labels_adaptive(
    hard_labels:   np.ndarray,
    datasets:      np.ndarray,
    num_classes:   int,
    alpha_map:     dict,
    default_alpha: float = 0.15,
) -> np.ndarray:
    """
    Build triangular soft labels with a per-sample alpha drawn from
    *alpha_map* keyed on the sample's dataset provenance string.

    Samples whose dataset key is absent from *alpha_map* fall back to
    *default_alpha*.

    Parameters
    ----------
    hard_labels   : (N,) int array   — ground-truth class indices.
    datasets      : (N,) str array   — dataset name per sample (lower-cased).
    num_classes   : int
    alpha_map     : dict[str, float] — dataset → smoothing alpha.
    default_alpha : float            — fallback when key not in alpha_map.

    Returns
    -------
    soft : (N, num_classes) float32 array, rows sum to 1.
    """
    N           = len(hard_labels)
    soft        = np.zeros((N, num_classes), dtype=np.float32)
    hard_labels = np.array(hard_labels, dtype=np.int64)

    for i, (c, ds) in enumerate(zip(hard_labels, datasets)):
        alpha      = float(alpha_map.get(str(ds).strip().lower(), default_alpha))
        soft[i, c] = 1.0 - 2 * alpha
        if c > 0:
            soft[i, c - 1] = alpha
        else:
            soft[i, c] += alpha
        if c < num_classes - 1:
            soft[i, c + 1] = alpha
        else:
            soft[i, c] += alpha

    return (soft / soft.sum(axis=1, keepdims=True)).astype(np.float32)


# ── Alpha schedule ────────────────────────────────────────────────────────────

def anneal_alpha(
    epoch:       int,
    total_epochs: int,
    alpha_start: float = 0.20,
    alpha_end:   float = 0.10,
) -> float:
    """
    Linear interpolation from *alpha_start* (epoch 0) to *alpha_end*
    (epoch ``total_epochs - 1``).

    Returns *alpha_end* immediately when ``total_epochs <= 1``.
    """
    if total_epochs <= 1:
        return alpha_end
    return alpha_start + (epoch / (total_epochs - 1)) * (alpha_end - alpha_start)


# ── Dataloader dispatcher ─────────────────────────────────────────────────────

def build_soft_labels(
    df:    "pd.DataFrame",
    cfg:   dict,
    alpha: float | None = None,
) -> np.ndarray:
    """
    Build soft labels for a DataFrame, respecting cfg flags.

    Dispatch order
    --------------
    1. If ``use_adaptive_smoothing`` is True and the provenance column
       exists, use ``make_soft_labels_adaptive``.  If *alpha* is given
       (annealing), the alpha_map is scaled proportionally so the
       annealing signal is preserved across all datasets.
    2. Otherwise use ``make_soft_labels`` with *alpha* (or the cfg
       default ``soft_label_alpha``).
    3. If ``use_soft_labels`` is False, return one-hot labels.

    Parameters
    ----------
    df    : DataFrame with a ``label`` column (0-indexed).
    cfg   : experiment config dict.
    alpha : optional override (used during alpha annealing).

    Returns
    -------
    (N, num_classes) float32 array.
    """
    labels = df["label"].values.astype(int)

    if cfg.get("use_adaptive_smoothing", False) and cfg.get("use_soft_labels", True):
        prov_col = cfg.get("provenance_col", "dataset")
        if prov_col in df.columns:
            datasets  = df[prov_col].fillna("unknown").values
            alpha_map = cfg.get("adaptive_alpha_map", {})
            if alpha is not None:
                default_a = cfg.get("soft_label_alpha", 0.15)
                scale     = alpha / (default_a + 1e-8)
                alpha_map = {k: min(v * scale, 0.45) for k, v in alpha_map.items()}
            return make_soft_labels_adaptive(
                labels, datasets, cfg["num_classes"],
                alpha_map, cfg.get("soft_label_alpha", 0.15),
            )

    a = alpha if alpha is not None else cfg.get("soft_label_alpha", 0.15)
    if cfg.get("use_soft_labels", True):
        return make_soft_labels(labels, cfg["num_classes"], a)
    return np.eye(cfg["num_classes"])[labels].astype(np.float32)
