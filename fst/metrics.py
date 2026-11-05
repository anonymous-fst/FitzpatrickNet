"""
fst.metrics
===========
Comprehensive evaluation metrics for ordinal FST classification.

Constants
---------
AGG_KEYS    Ordered list of aggregate metric keys written to CSV.
PC_KEYS     Per-class metric keys written to CSV.

Functions
---------
compute_extended_metrics    Full per-fold metric dict including per-class breakdown.
bootstrap_ci                Percentile bootstrap 95 % CIs across fold scores.
compute_patient_metrics     Patient-level accuracy, confidence, and consistency.
quadratic_weighted_kappa    QWK wrapper with graceful NaN fallback.
expected_calibration_error  Top-label or classwise ECE.
mc_dropout_ece              ECE computed from MC-Dropout mean probabilities.
"""

from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    cohen_kappa_score,
)
from scipy.stats import spearmanr


# ── Metric key lists ──────────────────────────────────────────────────────────

AGG_KEYS: list[str] = [
    "accuracy", "precision", "recall", "specificity", "f1",
    "auroc", "concordance_pm1", "adjacent_precision", "spearman_rho",
    "qwk", "ece_top_label", "ece_classwise",
]

PC_KEYS: list[str] = [
    "accuracy", "precision", "recall", "specificity", "f1", "auroc",
]


# ── Individual metric helpers ─────────────────────────────────────────────────

def quadratic_weighted_kappa(
    labels:      np.ndarray,
    preds:       np.ndarray,
    num_classes: int = 6,
) -> float:
    """Quadratic weighted kappa with NaN fallback on degenerate inputs."""
    try:
        return float(cohen_kappa_score(
            labels, preds,
            weights="quadratic",
            labels=list(range(num_classes)),
        ))
    except Exception:
        return float("nan")


def expected_calibration_error(
    labels: np.ndarray,
    probs:  np.ndarray,
    n_bins: int = 15,
    mode:   str = "top_label",
) -> float:
    """
    Expected Calibration Error (ECE).

    Parameters
    ----------
    labels : (N,) int array of ground-truth class indices.
    probs  : (N, K) predicted probability array.
    n_bins : number of equal-width confidence bins.
    mode   : ``"top_label"`` (standard ECE) or ``"classwise"`` (mean
             over per-class OvR ECE).

    Returns
    -------
    float ECE in [0, 1].
    """
    if mode == "top_label":
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        correct     = (predictions == labels).astype(float)
        bin_edges   = np.linspace(0.0, 1.0, n_bins + 1)
        ece         = 0.0
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (confidences > lo) & (confidences <= hi)
            if mask.sum() == 0:
                continue
            ece += mask.sum() * abs(correct[mask].mean() - confidences[mask].mean())
        return float(ece / max(len(labels), 1))

    elif mode == "classwise":
        K         = probs.shape[1]
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece_per   = []
        for k in range(K):
            y_bin      = (labels == k).astype(float)
            y_prob     = probs[:, k]
            class_ece  = 0.0
            for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
                mask = (y_prob > lo) & (y_prob <= hi)
                if mask.sum() == 0:
                    continue
                class_ece += mask.sum() * abs(y_bin[mask].mean() - y_prob[mask].mean())
            ece_per.append(class_ece / max(len(labels), 1))
        return float(np.mean(ece_per))

    else:
        raise ValueError(f"Unknown ECE mode '{mode}'. Expected 'top_label' or 'classwise'.")


def mc_dropout_ece(
    mean_probs: np.ndarray,
    labels:     np.ndarray,
    n_bins:     int = 15,
) -> dict[str, float]:
    """
    Compute both top-label and classwise ECE from MC-Dropout mean probabilities.

    Returns
    -------
    dict with keys ``"mc_ece_top_label"`` and ``"mc_ece_classwise"``.
    """
    return {
        "mc_ece_top_label": expected_calibration_error(
            labels, mean_probs, n_bins=n_bins, mode="top_label"),
        "mc_ece_classwise": expected_calibration_error(
            labels, mean_probs, n_bins=n_bins, mode="classwise"),
    }


# ── Main metric function ──────────────────────────────────────────────────────

def compute_extended_metrics(
    labels:      np.ndarray,
    preds:       np.ndarray,
    probs:       np.ndarray,
    num_classes: int,
    class_names: list[str],
) -> dict:
    """
    Compute the full evaluation metric suite for one fold or test split.

    Aggregate metrics
    -----------------
    accuracy, precision (macro), recall (macro), specificity (macro),
    F1 (macro), AUROC (macro OvR), concordance_pm1 (off-by-one accuracy),
    adjacent_precision (macro precision on FST II–V), spearman_rho,
    QWK, ECE (top-label and classwise).

    Per-class metrics
    -----------------
    Stored under ``metrics["per_class"][class_name]`` for each class:
    precision, recall (= sensitivity), accuracy (= sensitivity), specificity,
    F1, AUROC.

    Parameters
    ----------
    labels      : (N,) int ground-truth class indices.
    preds       : (N,) int predicted class indices.
    probs       : (N, K) predicted probability array.
    num_classes : total number of classes.
    class_names : list of K human-readable class name strings.

    Returns
    -------
    dict containing all aggregate and per-class metrics.
    """
    metrics: dict = {}

    # ── Aggregate ─────────────────────────────────────────────
    metrics["accuracy"]  = accuracy_score(labels, preds)
    metrics["precision"] = precision_score(labels, preds, average="macro", zero_division=0)
    metrics["recall"]    = recall_score(labels, preds, average="macro", zero_division=0)
    metrics["f1"]        = f1_score(labels, preds, average="macro", zero_division=0)

    cm_full = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    specs   = []
    for i in range(num_classes):
        tp = cm_full[i, i]
        fp = cm_full[:, i].sum() - tp
        fn = cm_full[i, :].sum() - tp
        tn = cm_full.sum() - tp - fp - fn
        specs.append(tn / (tn + fp + 1e-8))
    metrics["specificity"] = float(np.mean(specs))

    try:
        metrics["auroc"] = roc_auc_score(
            labels, probs, multi_class="ovr", average="macro")
    except Exception:
        metrics["auroc"] = np.nan

    metrics["concordance_pm1"] = float((np.abs(labels - preds) <= 1).mean())

    adj_mask = np.isin(labels, [1, 2, 3, 4])
    metrics["adjacent_precision"] = (
        precision_score(labels[adj_mask], preds[adj_mask],
                        average="macro", zero_division=0)
        if adj_mask.sum() > 0 else np.nan
    )

    try:
        rho, pval = spearmanr(labels, preds)
        metrics["spearman_rho"]  = float(rho)
        metrics["spearman_pval"] = float(pval)
    except Exception:
        metrics["spearman_rho"]  = np.nan
        metrics["spearman_pval"] = np.nan

    metrics["qwk"]           = quadratic_weighted_kappa(labels, preds, num_classes)
    metrics["ece_top_label"] = expected_calibration_error(labels, probs, mode="top_label")
    metrics["ece_classwise"] = expected_calibration_error(labels, probs, mode="classwise")

    # ── Per-class ─────────────────────────────────────────────
    cm        = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    per_class = {}
    for i, name in enumerate(class_names):
        tp   = cm[i, i]
        fp   = cm[:, i].sum() - tp
        fn   = cm[i, :].sum() - tp
        tn   = cm.sum() - tp - fp - fn
        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
        prec = tp / (tp + fp + 1e-8)
        f1_i = 2 * prec * sens / (prec + sens + 1e-8)
        try:
            auroc_i = roc_auc_score((labels == i).astype(int), probs[:, i])
        except Exception:
            auroc_i = np.nan
        per_class[name] = {
            "precision":   prec,
            "recall":      sens,
            "accuracy":    sens,
            "specificity": spec,
            "f1":          f1_i,
            "auroc":       auroc_i,
        }
    metrics["per_class"] = per_class
    return metrics


# ── Bootstrap CIs ─────────────────────────────────────────────────────────────

def bootstrap_ci(
    fold_metrics_list: list[dict],
    metric_keys:       list[str],
    n_bootstrap:       int   = 1000,
    ci_level:          float = 0.95,
    seed:              int   = 42,
) -> pd.DataFrame:
    """
    Percentile bootstrap confidence intervals over fold-level scores.

    For each metric key, resample the fold scores with replacement
    *n_bootstrap* times and report the percentile CI.

    Parameters
    ----------
    fold_metrics_list : list of per-fold metric dicts.
    metric_keys       : which keys to compute CIs for.
    n_bootstrap       : number of bootstrap resamples.
    ci_level          : coverage level (default 0.95 → 95 % CI).
    seed              : RNG seed for reproducibility.

    Returns
    -------
    DataFrame with columns: metric, mean, std, ci_lower, ci_upper, ci_level.
    """
    rng  = np.random.default_rng(seed)
    rows = []
    for key in metric_keys:
        scores = np.array([
            m[key] for m in fold_metrics_list
            if not np.isnan(m.get(key, np.nan))
        ])
        if len(scores) == 0:
            rows.append({
                "metric": key, "mean": np.nan, "std": np.nan,
                "ci_lower": np.nan, "ci_upper": np.nan, "ci_level": ci_level,
            })
            continue
        boot_means = np.array([
            rng.choice(scores, size=len(scores), replace=True).mean()
            for _ in range(n_bootstrap)
        ])
        alpha = (1.0 - ci_level) / 2.0
        rows.append({
            "metric":   key,
            "mean":     float(scores.mean()),
            "std":      float(scores.std()),
            "ci_lower": float(np.percentile(boot_means, 100 * alpha)),
            "ci_upper": float(np.percentile(boot_means, 100 * (1 - alpha))),
            "ci_level": ci_level,
        })
    return pd.DataFrame(rows)


# ── Patient-level metrics ─────────────────────────────────────────────────────

def compute_patient_metrics(
    labels:      np.ndarray,
    preds:       np.ndarray,
    probs:       np.ndarray,
    patient_ids: list[str],
    num_classes: int,
) -> pd.DataFrame:
    """
    Aggregate per-image predictions to the patient level.

    Only patients with ≥ 2 images in the split are included.
    Singleton patients (``__unique_*`` IDs) are excluded.

    Columns in the returned DataFrame
    ----------------------------------
    patient_id, n_images, accuracy, mean_confidence, entropy,
    norm_entropy, consistency.

    Parameters
    ----------
    labels      : (N,) int ground-truth class indices.
    preds       : (N,) int predicted class indices.
    probs       : (N, K) predicted probability array.
    patient_ids : list of N patient ID strings.
    num_classes : total number of classes (used to normalise entropy).

    Returns
    -------
    DataFrame (one row per multi-image patient).
    """
    records: dict = defaultdict(lambda: {"labels": [], "preds": [], "probs": []})
    for i, pid in enumerate(patient_ids):
        if not pid.startswith("__unique_"):
            records[pid]["labels"].append(labels[i])
            records[pid]["preds"].append(preds[i])
            records[pid]["probs"].append(probs[i])

    rows = []
    for pid, data in records.items():
        if len(data["labels"]) < 2:
            continue
        lbls      = np.array(data["labels"])
        prds      = np.array(data["preds"])
        prbs      = np.array(data["probs"])
        mean_prob = prbs.mean(axis=0)
        H         = -np.sum(mean_prob * np.log(mean_prob + 1e-8))
        H_max     = np.log(num_classes)
        norm_H    = H / H_max
        rows.append({
            "patient_id":      pid,
            "n_images":        len(lbls),
            "accuracy":        float((lbls == prds).mean()),
            "mean_confidence": float(prbs[np.arange(len(lbls)), lbls].mean()),
            "entropy":         float(H),
            "norm_entropy":    float(norm_H),
            "consistency":     float(1.0 - norm_H),
        })
    return pd.DataFrame(rows)
