#!/usr/bin/env python3
"""
ensemble_fold_averaging.py
==========================
End-to-end fold-level ensemble pipeline for multimodal ViT/CNN experiments.

For each experiment group (identified by base name), the script:
  1. Discovers seed-specific folders (default seeds: 42–46).
  2. Loads the best fold model for each of the 5 folds within each seed.
  3. Runs inference on the held-out test set and averages probabilities
     across folds → per-seed fold ensemble.
  4. Computes a comprehensive metric suite for each fold individually and
     for the fold ensemble.
  5. Aggregates all metrics across seeds (mean ± std).
  6. Saves prediction CSVs, metric summaries, confusion matrices, and
     ROC curves — no separate post-hoc plotting step required.

Architecture routing
--------------------
  ViT archs  (vit_small, eva02_small)    → train_eval_vit_ita module
  CNN archs  (resnet18, efficientnet_b0) → train_eval_cnn_ita module

ITA normalisation
-----------------
  ViT : fixed z-score via cfg["ita_mean"] / cfg["ita_std"], handled
        internally by SkinDatasetViTITA.
  CNN : per-fold ITANormalizer saved as
        <model_dir>/models/ita_normalizer_fold{N}.json during training;
        loaded here and passed to cnn_te.build_test_dataloader.

Usage
-----
    python ensemble_fold_averaging.py \
        --results_dirs  results/multimodal_vit_ita results/multimodal_cnn_ita \
        --test_csv      data_split/test_dataset_metadata_extended_balanced.csv \
        [--seeds 42 43 44 45 46] \
        [--force]

Output layout (written inside each seed's existing directory)
-------------------------------------------------------------
    <seed_dir>/
        metrics/
            test_predictions_all_folds_ensemble.csv
            ensemble_summary.json
        plots/
            fold_ensemble_comparison.png   # fold vs. ensemble bar chart
            cm_fold_ensemble.png / .csv / _normalised.csv
            roc_fold_ensemble.png

Aggregated results across all seeds are written to:
    <results_dir>/<base_name>_seed_ensemble/
        metrics/
            aggregated_ensemble_metrics.csv
            per_seed_ensemble_metrics.csv
            seed_ensemble_summary.json      ← sentinel; re-run skipped unless --force
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc,
)
from sklearn.preprocessing import label_binarize
from scipy.stats import spearmanr

import train_eval_vit_ita as vit_te
import train_eval_cnn_ita as cnn_te


# ── Constants ─────────────────────────────────────────────────────────────────

CLASS_NAMES  = [f"FST {i+1}" for i in range(6)]
_COLORS      = ["#F6D5B0", "#E8B88A", "#C68642", "#8D5524", "#4A2912", "#2C1A0E"]
_LINE_STYLES = ["-", "-", "--", "--", "-.", ":"]
_LINE_WIDTHS = [2.0, 2.0, 2.0, 2.0, 2.5, 2.5]

# Written after a successful run — used as the skip-logic sentinel.
SENTINEL_FILENAME = "seed_ensemble_summary.json"

# Architectures that belong to each training module.
_VIT_ARCHS = {"vit_small", "eva02_small"}
_CNN_ARCHS  = {"resnet18", "efficientnet_b0"}


# ── Architecture routing ──────────────────────────────────────────────────────

def is_cnn_experiment(cfg: dict) -> bool:
    return cfg.get("arch", "").lower() in _CNN_ARCHS


def get_te_module(cfg: dict):
    """Return vit_te or cnn_te based on cfg['arch']."""
    arch = cfg.get("arch", "").lower()
    if arch in _VIT_ARCHS:
        return vit_te
    if arch in _CNN_ARCHS:
        return cnn_te
    raise ValueError(
        f"Unknown architecture '{arch}'. "
        f"Expected one of: {sorted(_VIT_ARCHS | _CNN_ARCHS)}"
    )


# ── CNN ITA normaliser loader ─────────────────────────────────────────────────

def load_cnn_ita_normalizer(model_dir: str, fold: int):
    """
    Load the per-fold ITANormalizer saved during CNN training.

    Search order:
      1. <model_dir>/models/ita_normalizer_fold{fold}.json  (exact match)
      2. Any ita_normalizer_fold*.json in models/            (nearest fallback)
      3. Identity normalizer (mean=0, std=1) with a warning
    """
    primary = os.path.join(model_dir, "models", f"ita_normalizer_fold{fold}.json")
    if os.path.exists(primary):
        norm = cnn_te.ITANormalizer()
        with open(primary) as f:
            norm.load_state_dict(json.load(f))
        return norm, primary

    fallbacks = sorted(
        glob.glob(os.path.join(model_dir, "models", "ita_normalizer_fold*.json"))
    )
    if fallbacks:
        norm = cnn_te.ITANormalizer()
        with open(fallbacks[0]) as f:
            norm.load_state_dict(json.load(f))
        print(
            f"  [WARN] ita_normalizer_fold{fold}.json not found — "
            f"using {os.path.basename(fallbacks[0])} as fallback"
        )
        return norm, fallbacks[0]

    print(
        f"  [WARN] No ITA normalizer found in {model_dir}/models/. "
        f"Using identity (mean=0, std=1). ITA values will NOT be properly "
        f"normalised — predictions may be degraded."
    )
    return cnn_te.ITANormalizer(), None


# ── Skip-logic ────────────────────────────────────────────────────────────────

def ensemble_already_done(results_dir: str, base_name: str) -> bool:
    """
    Return True if the aggregated sentinel file already exists, indicating
    this experiment group has been fully processed in a previous run.
    """
    ensemble_dir_name = re.sub(r'_?seed\d+', '', base_name) + "_seed_ensemble"
    sentinel = os.path.join(
        results_dir, ensemble_dir_name, "metrics", SENTINEL_FILENAME
    )
    return os.path.exists(sentinel)


# ── Directory / config utilities ──────────────────────────────────────────────

def extract_seed_from_path(path: str):
    match = re.search(r'seed(\d+)', path)
    return int(match.group(1)) if match else None


def get_base_experiment_name(path: str) -> str:
    return re.sub(r'_?seed\d+', '', os.path.basename(path))


def find_experiment_groups(results_dir: str, seeds: list) -> dict:
    """
    Scan results_dir for experiment folders and group by base name (seed suffix
    removed).  Returns {base_name: {seed: full_path}} for groups that have
    every requested seed present.
    """
    all_dirs = [d for d in glob.glob(os.path.join(results_dir, "*"))
                if os.path.isdir(d)]

    groups: dict = defaultdict(list)
    for d in all_dirs:
        seed = extract_seed_from_path(d)
        if seed in seeds:
            groups[get_base_experiment_name(d)].append((seed, d))

    return {
        base: dict(seed_dirs)
        for base, seed_dirs in groups.items()
        if sorted(s for s, _ in seed_dirs) == sorted(seeds)
    }


def load_config(model_dir: str) -> dict:
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    with open(config_path) as f:
        return json.load(f)


def get_best_fold_model(model_dir: str, fold: int):
    matches = glob.glob(
        os.path.join(model_dir, "models", f"best_fold{fold}_*.pth")
    )
    return matches[0] if matches else None


# ── ITA column verification ───────────────────────────────────────────────────

def check_ita_availability(test_df: pd.DataFrame) -> bool:
    """Report ITA column coverage and return True if any valid values exist."""
    ita_col = "ITA_value_spectral_spatial"
    print(f"\n  ITA Column Check:")
    print(f"  {'─'*70}")

    if ita_col not in test_df.columns:
        print(f"  [WARN] Column '{ita_col}' not found in test CSV")
        print(f"         Available columns: {list(test_df.columns)[:10]}...")
        print(f"         Will impute 0.0 (normalised) for all samples")
        return False

    n_total   = len(test_df)
    n_valid   = int(test_df[ita_col].notna().sum())
    n_missing = n_total - n_valid
    pct_valid = (n_valid / n_total * 100) if n_total > 0 else 0

    print(f"  [OK] Column '{ita_col}' found")
    print(f"       Valid ITA values : {n_valid}/{n_total} ({pct_valid:.1f}%)")

    if n_valid > 0:
        v = test_df[ita_col].dropna()
        print(f"       Mean / Std      : {v.mean():.2f} / {v.std():.2f}")
        print(f"       Min  / Max      : {v.min():.2f} / {v.max():.2f}")

    if n_missing > 0:
        print(f"       Missing         : {n_missing} samples → imputed as 0.0")

    return n_valid > 0


# ── Prediction generation ─────────────────────────────────────────────────────

def generate_fold_predictions(model_dir, fold, test_df, cfg, device):
    """
    Load one fold model and run inference on test_df.
    Returns (predictions, probabilities, labels) or (None, None, None) on failure.
    """
    model_path = get_best_fold_model(model_dir, fold)
    if not model_path:
        print(f"  [WARN] No model found for fold {fold} in {model_dir}")
        return None, None, None

    if is_cnn_experiment(cfg):
        model = cnn_te.build_model(cfg, device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        normalizer, _ = load_cnn_ita_normalizer(model_dir, fold)
        test_loader   = cnn_te.build_test_dataloader(test_df, cfg, normalizer)
    else:
        vit_te.resolve_image_size(cfg)
        model = vit_te.build_model(cfg, device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        test_loader = vit_te.build_test_dataloader(test_df, cfg)

    fold_probs, fold_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 8:
                imgs, _, hard_lbls, ita, has_ita, _, _, _ = batch
            else:
                imgs, _, hard_lbls, ita, has_ita, _, _ = batch

            logits = model(imgs.to(device), ita.to(device), has_ita.to(device))
            fold_probs.append(F.softmax(logits, dim=1).cpu().numpy())
            fold_labels.extend(hard_lbls.cpu().numpy())

    fold_probs  = np.concatenate(fold_probs, axis=0)
    fold_labels = np.array(fold_labels)
    return fold_probs.argmax(axis=1), fold_probs, fold_labels


def generate_predictions_for_seed(seed_model_dir, test_df, cfg, device):
    """
    Run inference for all 5 folds of one seed.
    Returns (all_fold_probs, all_labels, fold_predictions_dict).
    """
    all_fold_probs: list  = []
    all_labels            = None
    fold_preds_dict: dict = {}

    for fold in range(1, 6):
        print(f"    Fold {fold}...", end=" ", flush=True)
        preds, probs, labels = generate_fold_predictions(
            seed_model_dir, fold, test_df, cfg, device
        )
        if probs is None:
            print("[SKIP]")
            continue
        all_fold_probs.append(probs)
        all_labels = labels
        fold_preds_dict[fold] = {
            "predictions":   preds,
            "probabilities": probs,
            "labels":        labels,
        }
        print("[OK]")

    return all_fold_probs, all_labels, fold_preds_dict


def create_ensemble_predictions(all_fold_probs):
    """Average probabilities across folds → ensemble predictions."""
    if not all_fold_probs:
        return None, None
    avg_probs = np.mean(all_fold_probs, axis=0)
    return avg_probs.argmax(axis=1), avg_probs


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_extended_metrics(labels, preds, probs, num_classes, class_names):
    """
    Compute a comprehensive metric suite for a single set of predictions.

    Metrics returned
    ----------------
    accuracy, precision (macro), recall (macro), f1 (macro), specificity (macro),
    AUROC (macro OvR), off_by_one_accuracy, spearman_rho, qwk
    """
    metrics = {
        "accuracy":  float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, average="macro",
                                           zero_division=0)),
        "recall":    float(recall_score(labels, preds, average="macro",
                                        zero_division=0)),
        "f1":        float(f1_score(labels, preds, average="macro",
                                    zero_division=0)),
    }

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
        metrics["auroc"] = float(
            roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        )
    except Exception:
        metrics["auroc"] = float("nan")

    metrics["off_by_one_accuracy"] = float((np.abs(labels - preds) <= 1).mean())

    try:
        rho, _ = spearmanr(labels, preds)
        metrics["spearman_rho"] = float(rho)
    except Exception:
        metrics["spearman_rho"] = float("nan")

    try:
        metrics["qwk"] = float(
            cohen_kappa_score(labels, preds, weights="quadratic")
        )
    except Exception:
        metrics["qwk"] = float("nan")

    return metrics


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_fold_ensemble_comparison(fold_metrics_list, ensemble_metrics,
                                   output_dir, seed):
    """
    2×3 bar-chart grid comparing individual fold metrics to the fold ensemble.
    Saved as fold_ensemble_comparison.png inside output_dir.
    """
    metrics_to_plot = ["accuracy", "precision", "recall", "f1", "auroc", "qwk"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    folds = [f"Fold {i+1}" for i in range(len(fold_metrics_list))]
    x_pos = np.arange(len(folds))

    for idx, metric in enumerate(metrics_to_plot):
        ax              = axes[idx]
        individual_vals = [m.get(metric, float("nan")) for m in fold_metrics_list]
        ensemble_val    = ensemble_metrics.get(metric, float("nan"))

        valid_vals = [v for v in individual_vals if not np.isnan(v)]
        if not valid_vals:
            ax.text(0.5, 0.5, f"No data for {metric}", ha="center", va="center")
            continue

        bars = ax.bar(x_pos[:len(valid_vals)], valid_vals, alpha=0.7,
                      color="steelblue")

        if not np.isnan(ensemble_val):
            ax.axhline(ensemble_val, color="red", linestyle="--",
                       linewidth=2.5, label="Ensemble")
            ax.text(len(valid_vals) - 0.5, ensemble_val + 0.01,
                    f"{ensemble_val:.3f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color="red")

        for bar, val in zip(bars, valid_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x_pos[:len(valid_vals)])
        ax.set_xticklabels(folds[:len(valid_vals)], rotation=45, ha="right")
        ax.set_ylabel(metric.capitalize())
        ax.set_title(metric.upper(), fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)

        if idx == 0 and not np.isnan(ensemble_val):
            ax.legend(loc="lower right", fontsize=9)

    fig.suptitle(
        f"Fold Ensemble (Seed {seed}): Individual vs. Logit-Averaged",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(
        os.path.join(output_dir, "fold_ensemble_comparison.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)
    print(f"    [OK] Saved comparison chart → {output_dir}/fold_ensemble_comparison.png")


def plot_confusion_matrix(labels, preds, class_names, tag, save_dir):
    """
    Normalised confusion matrix with raw counts and percentages annotated.
    Saves <save_dir>/cm_<tag>.png, cm_<tag>.csv, cm_<tag>_normalised.csv.
    """
    cm      = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm, annot=False, fmt="", cmap="Blues", vmin=0.0, vmax=1.0,
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5, linecolor="grey",
    )

    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color("grey")
        spine.set_linewidth(0.5)

    ax.tick_params(axis="both", which="major", labelsize=14)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            tc = "white" if cm_norm[i, j] > 0.55 else "black"
            ax.text(j + 0.5, i + 0.38, f"{cm[i, j]}", ha="center", va="center",
                    fontsize=14, fontweight="bold", color=tc)
            ax.text(j + 0.5, i + 0.65, f"({cm_norm[i, j]*100:.1f}%)",
                    ha="center", va="center", fontsize=12, color=tc)

    ax.set_xlabel("Predicted", fontsize=16)
    ax.set_ylabel("True", fontsize=16)

    cbar = ax.collections[0].colorbar
    cbar.set_label("Proportion of Ground Truth Class", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"cm_{tag}.png"), dpi=120,
                bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        os.path.join(save_dir, f"cm_{tag}.csv")
    )
    pd.DataFrame(cm_norm.round(4), index=class_names, columns=class_names).to_csv(
        os.path.join(save_dir, f"cm_{tag}_normalised.csv")
    )
    print(f"    [OK] Saved confusion matrix → {save_dir}/cm_{tag}.png")


def plot_roc_curve(labels, probs, class_names, tag, save_dir):
    """
    One-vs-rest ROC curves for all classes on a single axes.
    Saves <save_dir>/roc_<tag>.png.
    """
    y_bin = label_binarize(labels, classes=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(9, 7))

    for i, (name, color, ls, lw) in enumerate(
        zip(class_names, _COLORS, _LINE_STYLES, _LINE_WIDTHS)
    ):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=lw,
                label=f"{name} (AUC={auc(fpr, tpr):.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate", fontsize=16)
    ax.set_ylabel("True Positive Rate", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.legend(loc="lower right", fontsize=14)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"roc_{tag}.png"), dpi=150)
    plt.close(fig)
    print(f"    [OK] Saved ROC curve     → {save_dir}/roc_{tag}.png")


# ── Per-experiment processing ─────────────────────────────────────────────────

def process_experiment_group(base_name, seed_dirs, test_df, device):
    """
    Process one experiment group across all requested seeds.

    For each seed:
      - Verifies all 5 fold models are present.
      - Runs inference and creates the fold ensemble.
      - Computes per-fold and ensemble metrics.
      - Saves prediction CSV, ensemble_summary.json, comparison chart,
        confusion matrix, and ROC curve into the seed's own directory.

    Returns {seed: result_dict} for seeds that completed successfully.
    """
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: {base_name}")
    print(f"{'='*70}")

    seed_results: dict = {}

    for seed in sorted(seed_dirs.keys()):
        model_dir = seed_dirs[seed]
        print(f"\n  Seed {seed}: {os.path.basename(model_dir)}")

        # ── Verify models/ directory ──────────────────────────
        models_dir = os.path.join(model_dir, "models")
        if not os.path.isdir(models_dir):
            print(f"  [SKIP] models/ directory not found")
            continue

        # ── Verify all 5 fold models are present ──────────────
        missing_folds = [
            f for f in range(1, 6)
            if not glob.glob(os.path.join(models_dir, f"best_fold{f}_*.pth"))
        ]
        if missing_folds:
            print(f"  [SKIP] Missing fold model(s): {missing_folds} "
                  f"— skipping seed {seed}")
            continue
        print(f"    [OK] All 5 fold models present")

        # ── Load config ───────────────────────────────────────
        try:
            cfg = load_config(model_dir)
        except Exception as e:
            print(f"  [ERROR] Failed to load config: {e}")
            continue

        class_names = get_te_module(cfg).CLASS_NAMES

        # ── Run fold inference ────────────────────────────────
        print(f"    Loading fold models...")
        all_fold_probs, all_labels, fold_preds_dict = generate_predictions_for_seed(
            model_dir, test_df, cfg, device
        )

        if not all_fold_probs:
            print(f"  [SKIP] No valid fold predictions for seed {seed}")
            continue

        # ── Create fold ensemble ──────────────────────────────
        print(f"    Creating fold ensemble...")
        ensemble_preds, ensemble_probs = create_ensemble_predictions(all_fold_probs)

        # ── Compute metrics ───────────────────────────────────
        fold_metrics_list = []
        for fold, fold_data in sorted(fold_preds_dict.items()):
            fm = compute_extended_metrics(
                fold_data["labels"], fold_data["predictions"],
                fold_data["probabilities"], cfg["num_classes"], class_names,
            )
            fm["fold"] = fold
            fold_metrics_list.append(fm)

        ensemble_metrics = compute_extended_metrics(
            all_labels, ensemble_preds, ensemble_probs,
            cfg["num_classes"], class_names,
        )

        # ── Output directories ────────────────────────────────
        metrics_dir = os.path.join(model_dir, "metrics")
        plots_dir   = os.path.join(model_dir, "plots")
        os.makedirs(metrics_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        # ── Save prediction CSV ───────────────────────────────
        pred_df = test_df.copy().reset_index(drop=True)
        pred_df["prediction"] = ensemble_preds
        pred_df["true_label"] = all_labels
        for i in range(cfg["num_classes"]):
            pred_df[f"prob_{class_names[i]}"] = ensemble_probs[:, i]

        pred_csv = os.path.join(
            metrics_dir, "test_predictions_all_folds_ensemble.csv"
        )
        pred_df.to_csv(pred_csv, index=False)
        print(f"    [OK] Saved predictions → {os.path.basename(pred_csv)}")

        # ── Save summary JSON ─────────────────────────────────
        mean_fold_acc = float(np.mean([m["accuracy"] for m in fold_metrics_list]))
        improvement   = float(ensemble_metrics["accuracy"] - mean_fold_acc)
        summary = {
            "seed":                    seed,
            "individual_fold_metrics": fold_metrics_list,
            "ensemble_metrics":        ensemble_metrics,
            "mean_fold_accuracy":      mean_fold_acc,
            "improvement_accuracy":    improvement,
            "improvement_pct":         (
                improvement / mean_fold_acc * 100 if mean_fold_acc > 0 else 0.0
            ),
        }

        json_path = os.path.join(metrics_dir, "ensemble_summary.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"    [OK] Saved summary    → {os.path.basename(json_path)}")

        # ── Generate plots ────────────────────────────────────
        tag = "fold_ensemble"
        plot_fold_ensemble_comparison(fold_metrics_list, ensemble_metrics,
                                      plots_dir, seed)
        plot_confusion_matrix(all_labels, ensemble_preds, class_names,
                              tag, plots_dir)
        plot_roc_curve(all_labels, ensemble_probs, class_names,
                       tag, plots_dir)

        seed_results[seed] = {
            "config":           cfg,
            "fold_metrics":     fold_metrics_list,
            "ensemble_metrics": ensemble_metrics,
            "summary":          summary,
        }

    return seed_results


# ── Cross-seed aggregation ────────────────────────────────────────────────────

def aggregate_seed_results(base_name, seed_results):
    """
    Collect ensemble metrics across all seeds and compute mean ± std.
    Prints a formatted summary table and returns the aggregated data dict.
    """
    print(f"\n  {'─'*70}")
    print(f"  AGGREGATING ACROSS SEEDS")
    print(f"  {'─'*70}")

    if not seed_results:
        print(f"  [ERROR] No valid seed results to aggregate")
        return None

    aggregated = {k: [] for k in [
        "accuracy", "precision", "recall", "f1", "auroc",
        "qwk", "off_by_one_accuracy", "specificity", "spearman_rho",
    ]}
    improvements = {k: [] for k in [
        "mean_fold_accuracy", "improvement_accuracy", "improvement_pct",
    ]}

    for seed in sorted(seed_results):
        ens_m = seed_results[seed]["ensemble_metrics"]
        summ  = seed_results[seed]["summary"]

        for key in aggregated:
            val = ens_m.get(key, float("nan"))
            if not np.isnan(val):
                aggregated[key].append(val)

        for key in improvements:
            val = float(summ.get(key, float("nan")))
            if not np.isnan(val):
                improvements[key].append(val)

    print(f"\n  Ensemble Performance (Mean ± Std across {len(seed_results)} seeds):")
    print(f"  {'─'*70}")
    print(f"  {'Metric':<25} | {'Mean':<10} | {'Std Dev':<10}")
    print(f"  {'-'*70}")
    for metric, values in aggregated.items():
        if values:
            print(f"  {metric:<25} | {np.mean(values):<10.4f} | "
                  f"{np.std(values):<10.4f}")

    print(f"\n  Improvement Metrics (Mean ± Std):")
    print(f"  {'─'*70}")
    print(f"  {'Metric':<25} | {'Mean':<10} | {'Std Dev':<10}")
    print(f"  {'-'*70}")
    for metric, values in improvements.items():
        if values:
            mv, sv = np.mean(values), np.std(values)
            if "pct" in metric:
                print(f"  {metric:<25} | {mv:+10.2f}% | {sv:<10.2f}%")
            else:
                print(f"  {metric:<25} | {mv:<10.4f} | {sv:<10.4f}")

    return {"aggregated_metrics": aggregated, "improvement_metrics": improvements}


def save_aggregated_results(base_name, results_dir, aggregated_data, seed_results):
    """
    Write cross-seed aggregated results to a dedicated <base>_seed_ensemble/
    directory inside results_dir.  The sentinel JSON is written last so that
    an interrupted run does not leave a false-positive skip marker.
    """
    ensemble_dir_name = re.sub(r'_?seed\d+', '', base_name) + "_seed_ensemble"
    ensemble_dir      = os.path.join(results_dir, ensemble_dir_name)
    metrics_dir       = os.path.join(ensemble_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # Aggregated metrics CSV
    agg_records = [
        {
            "metric":  metric,
            "mean":    float(np.mean(values)),
            "std":     float(np.std(values)),
            "n_seeds": len(values),
        }
        for metric, values in aggregated_data["aggregated_metrics"].items()
        if values
    ]
    pd.DataFrame(agg_records).to_csv(
        os.path.join(metrics_dir, "aggregated_ensemble_metrics.csv"), index=False
    )

    # Per-seed metrics CSV
    seed_records = []
    for seed in sorted(seed_results):
        result = seed_results[seed]
        row    = {"seed": seed}
        row.update(result["ensemble_metrics"])
        row.update({
            f"improvement_{k}": v
            for k, v in result["summary"].items()
            if k.startswith("improvement") or k == "mean_fold_accuracy"
        })
        seed_records.append(row)
    pd.DataFrame(seed_records).to_csv(
        os.path.join(metrics_dir, "per_seed_ensemble_metrics.csv"), index=False
    )

    # Sentinel / summary JSON — written last
    sentinel = {
        "experiment": base_name,
        "n_seeds":    len(seed_results),
        "seeds":      sorted(seed_results.keys()),
        "aggregated_metrics": {
            k: {
                "mean":     float(np.mean(v)) if v else float("nan"),
                "std":      float(np.std(v))  if v else float("nan"),
                "n_values": len(v),
            }
            for k, v in aggregated_data["aggregated_metrics"].items()
        },
        "improvement_metrics": {
            k: {
                "mean":     float(np.mean(v)) if v else float("nan"),
                "std":      float(np.std(v))  if v else float("nan"),
                "n_values": len(v),
            }
            for k, v in aggregated_data["improvement_metrics"].items()
        },
    }

    with open(os.path.join(metrics_dir, SENTINEL_FILENAME), "w") as f:
        json.dump(sentinel, f, indent=2)

    print(f"\n  [OK] Aggregated results saved → {ensemble_dir}")
    return ensemble_dir


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Fold-level ensemble pipeline for multimodal ViT/CNN experiments. "
            "Averages probabilities across the 5 cross-validation fold models "
            "within each seed, then aggregates metrics across seeds."
        )
    )
    p.add_argument(
        "--results_dirs",
        nargs="+",
        required=True,
        help="One or more directories containing seed-specific experiment folders.",
    )
    p.add_argument(
        "--test_csv",
        required=True,
        help="Path to the held-out test CSV (must contain a 'fitzpatrick' column).",
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45, 46],
        help="Seed values to look for (default: 42 43 44 45 46).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-run ensemble even if the sentinel output file already exists. "
            "Without this flag, completed experiment groups are skipped."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*70}")
    print(f"  FOLD-LEVEL ENSEMBLE PIPELINE")
    print(f"{'='*70}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device   : {device}")
    print(f"  Test CSV : {args.test_csv}")
    print(f"  Seeds    : {args.seeds}")
    print(f"  Force    : {args.force}\n")

    # ── Load test data ────────────────────────────────────────
    print(f"  Loading test data...")
    test_df = pd.read_csv(args.test_csv, low_memory=False)
    test_df["label"] = test_df["fitzpatrick"].astype(int) - 1
    print(f"  Test set : {len(test_df)} samples")
    print(f"  Label distribution:\n"
          f"{test_df['label'].value_counts().sort_index().to_string()}")

    check_ita_availability(test_df)

    all_ensemble_dirs: list = []
    skipped_groups:    list = []

    for results_dir in args.results_dirs:
        if not os.path.exists(results_dir):
            print(f"\n[WARN] Results directory not found: {results_dir}")
            continue

        print(f"\n{'='*70}")
        print(f"  SEARCHING: {results_dir}")
        print(f"{'='*70}")

        experiment_groups = find_experiment_groups(results_dir, args.seeds)

        if not experiment_groups:
            print(f"  [WARN] No complete experiment groups found "
                  f"(need all seeds: {args.seeds})")
            continue

        print(f"  Found {len(experiment_groups)} complete experiment group(s):\n")
        for base_name in sorted(experiment_groups):
            print(f"    • {base_name}")

        for base_name in sorted(experiment_groups):
            seed_dirs = experiment_groups[base_name]

            # ── Skip-logic ────────────────────────────────────
            if not args.force and ensemble_already_done(results_dir, base_name):
                print(
                    f"\n  [SKIP] '{base_name}' — already complete "
                    f"(sentinel '{SENTINEL_FILENAME}' found). "
                    f"Pass --force to re-run."
                )
                skipped_groups.append(base_name)
                continue

            # ── Process ───────────────────────────────────────
            seed_results = process_experiment_group(
                base_name, seed_dirs, test_df, device
            )

            if not seed_results:
                print(f"  [SKIP] No valid results for {base_name}")
                continue

            # ── Aggregate and save ────────────────────────────
            aggregated_data = aggregate_seed_results(base_name, seed_results)

            if aggregated_data:
                ensemble_dir = save_aggregated_results(
                    base_name, results_dir, aggregated_data, seed_results
                )
                all_ensemble_dirs.append(ensemble_dir)

    # ── Final summary ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'='*70}")

    if all_ensemble_dirs:
        print(f"\n  Ensemble directories created ({len(all_ensemble_dirs)}):")
        for d in all_ensemble_dirs:
            print(f"    • {d}")

    if skipped_groups:
        print(f"\n  Skipped — already done ({len(skipped_groups)}):")
        for g in skipped_groups:
            print(f"    • {g}")

    print()


if __name__ == "__main__":
    main()
