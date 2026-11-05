#!/usr/bin/env python3
"""
ensemble_logit_averaging.py
===========================
Load test predictions from multiple seed-trained models, average logits,
compute ensemble metrics, and compare against individual models.

Expected directory layout
--------------------------
  <model_dir>/
      metrics/
          test_predictions.csv   # columns: true_label, prob_FST 1, ..., prob_FST 6
"""

import os
import sys
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, cohen_kappa_score,
)


# ── Utilities ─────────────────────────────────────────────────────────────────

def load_model_predictions(model_dir, fold=1):
    """
    Load test predictions from a single model's results directory.
    Expects: <model_dir>/metrics/test_predictions.csv
    """
    pred_file = os.path.join(model_dir, "metrics", "test_predictions.csv")
    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"Predictions not found: {pred_file}")

    df = pd.read_csv(pred_file)

    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    if not prob_cols:
        raise ValueError(f"No probability columns found in {pred_file}")

    probs  = df[prob_cols].values.astype(np.float32)
    labels = df["true_label"].values.astype(int)

    return {
        "probs":     probs,
        "labels":    labels,
        "pred_file": pred_file,
        "n_samples": len(df),
    }


def compute_metrics(labels, probs, class_names=None):
    """Compute comprehensive metrics from labels and probability predictions."""
    preds = probs.argmax(axis=1)

    if class_names is None:
        class_names = [f"FST {i+1}" for i in range(probs.shape[1])]

    metrics = {
        "accuracy":  float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall":    float(recall_score(labels, preds, average="macro", zero_division=0)),
        "f1":        float(f1_score(labels, preds, average="macro", zero_division=0)),
        "qwk":       float(cohen_kappa_score(labels, preds, weights="quadratic")),
    }

    try:
        metrics["auroc"] = float(
            roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        )
    except Exception:
        metrics["auroc"] = np.nan

    return metrics, preds


def ensemble_logit_averaging(all_probs_list):
    """
    Average probabilities across multiple models.

    Parameters
    ----------
    all_probs_list : list of (n_samples, n_classes) arrays

    Returns
    -------
    (n_samples, n_classes) averaged and renormalised probabilities
    """
    stacked        = np.stack(all_probs_list, axis=0)
    ensemble_probs = stacked.mean(axis=0)
    ensemble_probs = ensemble_probs / (ensemble_probs.sum(axis=1, keepdims=True) + 1e-8)
    return ensemble_probs


def plot_comparison(individual_metrics_list, ensemble_metrics, output_dir, seeds):
    """Plot individual vs. ensemble metrics."""
    os.makedirs(output_dir, exist_ok=True)

    metrics_to_plot = ["accuracy", "precision", "recall", "f1", "auroc", "qwk"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]

        individual_vals = [m[metric] for m in individual_metrics_list]
        ensemble_val    = ensemble_metrics[metric]

        x_pos = np.arange(len(seeds))
        bars  = ax.bar(x_pos, individual_vals, alpha=0.7,
                       label="Individual seeds", color="steelblue")

        ax.axhline(ensemble_val, color="red", linestyle="--",
                   linewidth=2.5, label="Ensemble (avg logits)")

        for i, (bar, val) in enumerate(zip(bars, individual_vals)):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

        ax.text(len(seeds) - 0.5, ensemble_val + 0.01,
                f"{ensemble_val:.3f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="red")

        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"Seed {s}" for s in seeds], rotation=45, ha="right")
        ax.set_ylabel(metric.capitalize())
        ax.set_title(metric.upper(), fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        if idx == 0:
            ax.legend(loc="lower right", fontsize=9)

    fig.suptitle(
        "Seed Ensemble: Individual vs. Logit-Averaged Predictions",
        fontsize=14, fontweight="bold"
    )
    fig.tight_layout()
    out_path = os.path.join(output_dir, "seed_ensemble_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Comparison plot saved → {out_path}")


def plot_confusion_matrix(labels, preds, class_names, tag, output_dir):
    """Plot normalised confusion matrix."""
    cm      = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=False, fmt="", cmap="Blues", vmin=0.0, vmax=1.0,
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, linewidths=0.5, linecolor="grey")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            tc = "white" if cm_norm[i, j] > 0.55 else "black"
            ax.text(j+0.5, i+0.38, f"{cm[i,j]}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=tc)
            ax.text(j+0.5, i+0.65, f"({cm_norm[i,j]*100:.1f}%)",
                    ha="center", va="center", fontsize=8, color=tc)

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"Confusion Matrix — {tag}", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"cm_{tag}.png"),
                dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ensemble logit averaging & metrics computation"
    )
    parser.add_argument(
        "--model_dirs",
        nargs="+",
        required=True,
        help="List of model directories (one per seed)."
    )
    parser.add_argument(
        "--output_dir",
        default="results/seed_ensemble_results",
        help="Output directory for ensemble results."
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help="Fold number (used for naming only)."
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print("  SEED ENSEMBLE — LOGIT AVERAGING")
    print(f"{'='*70}\n")

    print(f"Loading predictions from {len(args.model_dirs)} models...\n")

    all_probs               = []
    individual_metrics_list = []
    seeds                   = []
    labels_ref              = None
    class_names             = [f"FST {i+1}" for i in range(6)]

    for model_dir in args.model_dirs:
        if not os.path.isdir(model_dir):
            print(f"  [WARN] Directory not found: {model_dir}")
            continue

        try:
            result = load_model_predictions(model_dir, fold=args.fold)
            probs  = result["probs"]
            labels = result["labels"]

            if labels_ref is None:
                labels_ref = labels
            elif not np.array_equal(labels_ref, labels):
                print(f"  [WARN] Labels mismatch in {model_dir} — skipping")
                continue

            all_probs.append(probs)

            metrics, preds = compute_metrics(labels, probs, class_names)
            individual_metrics_list.append(metrics)

            seed_str = model_dir.split("seed")[-1].split("_")[0]
            seeds.append(seed_str)

            print(
                f"  ✓ Seed {seed_str:>3s} | "
                f"Acc={metrics['accuracy']:.4f} | "
                f"F1={metrics['f1']:.4f} | "
                f"AUROC={metrics['auroc']:.4f}"
            )

        except Exception as e:
            print(f"  [ERROR] {model_dir}: {e}")
            continue

    if len(all_probs) < 2:
        print(f"\n[ERROR] Need at least 2 models, found {len(all_probs)}")
        return 1

    print(f"\n  Loaded {len(all_probs)} models successfully\n")

    print("Computing ensemble via logit averaging...")
    ensemble_probs   = ensemble_logit_averaging(all_probs)
    ensemble_metrics, ensemble_preds = compute_metrics(
        labels_ref, ensemble_probs, class_names
    )

    print(f"\n{'─'*70}")
    print("  INDIVIDUAL MODELS vs. ENSEMBLE")
    print(f"{'─'*70}\n")

    print(
        f"  {'Seed':<8} {'Accuracy':<12} {'Precision':<12} "
        f"{'Recall':<12} {'F1':<12} {'AUROC':<12}"
    )
    print(f"  {'-'*70}")

    for seed, metrics in zip(seeds, individual_metrics_list):
        print(
            f"  {seed:<8} {metrics['accuracy']:<12.4f} "
            f"{metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
            f"{metrics['f1']:<12.4f} {metrics['auroc']:<12.4f}"
        )

    print(f"  {'-'*70}")
    print(
        f"  {'ENSEMBLE':<8} {ensemble_metrics['accuracy']:<12.4f} "
        f"{ensemble_metrics['precision']:<12.4f} "
        f"{ensemble_metrics['recall']:<12.4f} "
        f"{ensemble_metrics['f1']:<12.4f} "
        f"{ensemble_metrics['auroc']:<12.4f}"
    )
    print(f"  {'-'*70}\n")

    individual_accs      = [m["accuracy"] for m in individual_metrics_list]
    mean_individual_acc  = np.mean(individual_accs)
    std_individual_acc   = np.std(individual_accs)
    ensemble_acc         = ensemble_metrics["accuracy"]
    improvement          = ensemble_acc - mean_individual_acc

    print(f"  Individual accuracy   : {mean_individual_acc:.4f} ± {std_individual_acc:.4f}")
    print(f"  Ensemble accuracy     : {ensemble_acc:.4f}")
    print(
        f"  Improvement           : {improvement:+.4f} "
        f"({improvement/mean_individual_acc*100:+.2f}%)\n"
    )

    results_summary = {
        "n_models":              len(all_probs),
        "seeds":                 seeds,
        "individual_metrics":    individual_metrics_list,
        "ensemble_metrics":      ensemble_metrics,
        "improvement_accuracy":  float(improvement),
        "improvement_pct":       float(improvement / mean_individual_acc * 100),
    }

    with open(os.path.join(args.output_dir, "ensemble_summary.json"), "w") as f:
        json.dump(results_summary, f, indent=2)

    ensemble_df = pd.DataFrame({
        "true_label":          labels_ref,
        "ensemble_prediction": ensemble_preds,
    })
    for i, name in enumerate(class_names):
        ensemble_df[f"ensemble_prob_{name}"] = ensemble_probs[:, i]
    ensemble_df.to_csv(
        os.path.join(args.output_dir, "ensemble_predictions.csv"), index=False
    )

    pd.DataFrame(individual_metrics_list).to_csv(
        os.path.join(args.output_dir, "individual_metrics.csv"), index=False
    )

    plot_comparison(individual_metrics_list, ensemble_metrics, args.output_dir, seeds)
    plot_confusion_matrix(
        labels_ref, ensemble_preds, class_names, "ensemble", args.output_dir
    )

    print(f"  Results saved → {args.output_dir}/")
    print(f"    - ensemble_summary.json")
    print(f"    - ensemble_predictions.csv")
    print(f"    - individual_metrics.csv")
    print(f"    - seed_ensemble_comparison.png")
    print(f"    - cm_ensemble.png")
    print(f"\n{'='*70}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
