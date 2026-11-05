"""
fst.plotting
============
All visualisation utilities shared by both training pipelines.

Constants
---------
CLASS_NAMES     Default FST class name strings.
_COLORS         Skin-tone palette (FST I → VI, light → dark).
_LINE_STYLES    Line styles for per-class ROC/PR curves.
_LINE_WIDTHS    Line widths for per-class ROC/PR curves.

Functions
---------
save_curves                     Per-fold and combined training curves (CSV + PNG).
plot_confusion_matrix           Normalised CM heat-map with raw counts.
plot_roc_curve                  Multi-class OvR ROC curves.
plot_pr_curve                   Multi-class OvR precision-recall curves.
plot_calibration_curve          Per-class reliability diagrams.
plot_metric_bars                Aggregate and per-class metric bar charts.
plot_patient_scatter            Patient consistency vs. accuracy scatter.
save_timing                     Timing records to CSV.
plot_curriculum_stage_comparison
                                Stage 1 vs Stage 2 bar chart.
plot_curriculum_learning_curves
                                Combined learning curves across curriculum stages.
save_final_model_curves         Final-model training loss/accuracy curves.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve

from fst.metrics import AGG_KEYS, PC_KEYS


# ── Palette / style constants ─────────────────────────────────────────────────

CLASS_NAMES: list[str] = [f"FST {i+1}" for i in range(6)]

_COLORS:      list[str]   = ["#F6D5B0", "#E8B88A", "#C68642", "#8D5524", "#4A2912", "#2C1A0E"]
_LINE_STYLES: list[str]   = ["-", "-", "--", "--", "-.", ":"]
_LINE_WIDTHS: list[float] = [2.0, 2.0, 2.0, 2.0, 2.5, 2.5]


# ── Training curves ───────────────────────────────────────────────────────────

def save_curves(
    train_losses_all:  list | None,
    val_losses_all:    list | None,
    train_accs_all:    list | None,
    val_accs_all:      list | None,
    lr_histories_all:  list | None,
    save_dir:          str,
    *,
    fold:         int | None  = None,
    train_losses: list | None = None,
    val_losses:   list | None = None,
    train_accs:   list | None = None,
    val_accs:     list | None = None,
    lr_history:   list | None = None,
) -> None:
    """
    Save training curves either as a single-fold CSV or as a combined
    multi-fold PNG + CSV.

    Single-fold mode
    ----------------
    Pass ``fold``, ``train_losses``, ``val_losses``, ``train_accs``,
    ``val_accs``, and ``lr_history``.  Writes one CSV file.

    Multi-fold mode
    ---------------
    Pass the ``*_all`` list-of-lists.  Writes three PNGs (loss, accuracy,
    LR schedule) with one subplot per fold.
    """
    # ── Single-fold CSV ───────────────────────────────────────
    if fold is not None and train_losses is not None:
        pd.DataFrame({
            "epoch":      range(1, len(train_losses) + 1),
            "train_loss": train_losses,
            "val_loss":   val_losses,
            "train_acc":  train_accs,
            "val_acc":    val_accs,
            "lr":         lr_history,
        }).to_csv(os.path.join(save_dir, f"curves_fold{fold}.csv"), index=False)
        return

    # ── Multi-fold PNGs ───────────────────────────────────────
    n_folds    = len(train_losses_all)
    max_epochs = max(len(l) for l in train_losses_all)

    def _fig(data_pairs, ylabel, title, filename, ymin, ymax,
             labels=("Train", "Val"), colors=("steelblue", "tomato")):
        fig, axes = plt.subplots(1, n_folds, figsize=(5 * n_folds, 4),
                                  sharey=True, sharex=True)
        if n_folds == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            if len(data_pairs[i]) == 2:
                ax.plot(data_pairs[i][0], color=colors[0], label=labels[0], linewidth=1.5)
                ax.plot(data_pairs[i][1], color=colors[1], label=labels[1], linewidth=1.5)
                if i == 0:
                    ax.legend(fontsize=8)
            else:
                ax.plot(data_pairs[i][0], color=colors[0], linewidth=1.5)
            ax.set_title(f"Fold {i+1}", fontsize=10)
            ax.set_xlabel("Epoch", fontsize=9)
            ax.set_ylim(ymin, ymax)
            ax.set_xlim(0, max_epochs)
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel(ylabel, fontsize=10)
        fig.suptitle(title, fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches="tight")
        plt.close(fig)

    all_tl = [v for f in train_losses_all for v in f]
    all_vl = [v for f in val_losses_all   for v in f]
    all_ta = [v for f in train_accs_all   for v in f]
    all_va = [v for f in val_accs_all     for v in f]
    all_lr = [v for f in lr_histories_all for v in f]

    _fig([(train_losses_all[i], val_losses_all[i]) for i in range(n_folds)],
         "Loss", "Training & Validation Loss — All Folds", "loss_all_folds.png",
         min(min(all_tl), min(all_vl)) * 0.95,
         max(max(all_tl), max(all_vl)) * 1.05)
    _fig([(train_accs_all[i], val_accs_all[i]) for i in range(n_folds)],
         "Accuracy", "Training & Validation Accuracy — All Folds",
         "accuracy_all_folds.png",
         min(min(all_ta), min(all_va)) * 0.95,
         min(1.0, max(max(all_ta), max(all_va)) * 1.05))
    _fig([(lr_histories_all[i],) for i in range(n_folds)],
         "Learning Rate", "LR Schedule — All Folds", "lr_all_folds.png",
         min(all_lr) * 0.9, max(all_lr) * 1.1,
         labels=("LR",), colors=("steelblue",))
    print(f"  Combined fold curves saved → {save_dir}")


# ── Confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(
    labels:      np.ndarray,
    preds:       np.ndarray,
    class_names: list[str],
    tag:         str,
    save_dir:    str,
) -> None:
    """
    Save a normalised confusion matrix heat-map with raw counts overlaid.

    Outputs: ``cm_{tag}.png``, ``cm_{tag}.csv``, ``cm_{tag}_normalised.csv``.
    """
    cm      = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=False, fmt="", cmap="Blues", vmin=0.0, vmax=1.0,
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, linewidths=0.5, linecolor="grey")

    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color("grey")
        spine.set_linewidth(0.5)

    ax.tick_params(axis="both", which="major", labelsize=14)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            tc = "white" if cm_norm[i, j] > 0.55 else "black"
            ax.text(j + 0.5, i + 0.38, f"{cm[i,j]}",
                    ha="center", va="center", fontsize=14, fontweight="bold", color=tc)
            ax.text(j + 0.5, i + 0.65, f"({cm_norm[i,j]*100:.1f}%)",
                    ha="center", va="center", fontsize=12, color=tc)

    ax.set_xlabel("Predicted", fontsize=16)
    ax.set_ylabel("True", fontsize=16)
    cbar = ax.collections[0].colorbar
    cbar.set_label("Proportion of Ground Truth Class", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"cm_{tag}.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        os.path.join(save_dir, f"cm_{tag}.csv"))
    pd.DataFrame(cm_norm.round(4), index=class_names, columns=class_names).to_csv(
        os.path.join(save_dir, f"cm_{tag}_normalised.csv"))


# ── ROC curve ─────────────────────────────────────────────────────────────────

def plot_roc_curve(
    labels:      np.ndarray,
    probs:       np.ndarray,
    class_names: list[str],
    tag:         str,
    save_dir:    str,
) -> None:
    """Save multi-class OvR ROC curves as ``roc_{tag}.png``."""
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


# ── PR curve ──────────────────────────────────────────────────────────────────

def plot_pr_curve(
    labels:      np.ndarray,
    probs:       np.ndarray,
    class_names: list[str],
    tag:         str,
    save_dir:    str,
) -> None:
    """Save multi-class OvR precision-recall curves as ``pr_{tag}.png``."""
    y_bin = label_binarize(labels, classes=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, (name, color, ls, lw) in enumerate(
        zip(class_names, _COLORS, _LINE_STYLES, _LINE_WIDTHS)
    ):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], probs[:, i])
        ax.plot(rec, prec, color=color, linestyle=ls, linewidth=lw,
                label=f"{name} (AP={auc(rec, prec):.3f})")
    ax.set_xlabel("Recall", fontsize=16)
    ax.set_ylabel("Precision", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.legend(loc="lower left", fontsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"pr_{tag}.png"), dpi=150)
    plt.close(fig)


# ── Calibration / reliability diagram ────────────────────────────────────────

def plot_calibration_curve(
    labels:      np.ndarray,
    probs:       np.ndarray,
    class_names: list[str],
    tag:         str,
    save_dir:    str,
    n_bins:      int = 10,
) -> None:
    """
    Save per-class OvR reliability diagrams as ``calibration_{tag}.png``
    and ``calibration_{tag}.csv``.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    axes      = axes.flatten()
    records   = []

    for i, (name, color) in enumerate(zip(class_names, _COLORS)):
        y_bin  = (labels == i).astype(int)
        y_prob = probs[:, i]
        try:
            frac_pos, mean_pred = calibration_curve(
                y_bin, y_prob, n_bins=n_bins, strategy="uniform")
        except ValueError:
            frac_pos  = np.array([0.0, 1.0])
            mean_pred = np.array([0.0, 1.0])

        ax = axes[i]
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.plot(mean_pred, frac_pos, color=color, marker="o", linewidth=2)
        ax.fill_between(mean_pred, frac_pos, mean_pred, alpha=0.15, color=color)
        ece = np.mean(np.abs(frac_pos - mean_pred))
        ax.set_title(f"{name}  (ECE={ece:.3f})", fontsize=9)
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        for mp, fp in zip(mean_pred, frac_pos):
            records.append({
                "class":             name,
                "mean_predicted":    round(float(mp), 4),
                "fraction_positive": round(float(fp), 4),
                "tag":               tag,
            })

    fig.suptitle(f"Reliability Diagrams — {tag}", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"calibration_{tag}.png"),
                dpi=120, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(records).to_csv(
        os.path.join(save_dir, f"calibration_{tag}.csv"), index=False)


# ── Metric bar charts ─────────────────────────────────────────────────────────

def plot_metric_bars(
    fold_metrics_list: list[dict],
    class_names:       list[str],
    save_dir:          str,
) -> None:
    """
    Save aggregate and per-class metric bar charts (mean ± std across folds).

    Outputs
    -------
    ``bar_aggregate.png``
        One bar per aggregate metric key in AGG_KEYS.
    ``bar_perclass_{metric}.png``
        One bar per class for each metric in PC_KEYS.
    """
    agg_labels = {
        "accuracy":          "Accuracy",
        "precision":         "Precision\n(Macro)",
        "recall":            "Recall\n(Macro)",
        "specificity":       "Specificity\n(Macro)",
        "f1":                "F1\n(Macro)",
        "auroc":             "AUROC\n(Macro)",
        "concordance_pm1":   "±1\nConcordance",
        "adjacent_precision":"Adjacent\nPrecision",
        "spearman_rho":      "Spearman\nρ",
        "qwk":               "QWK",
        "ece_top_label":     "ECE\n(top-label)",
        "ece_classwise":     "ECE\n(classwise)",
    }
    agg_data = {
        k: [m[k] for m in fold_metrics_list if not np.isnan(m.get(k, np.nan))]
        for k in AGG_KEYS
    }

    fig, ax = plt.subplots(figsize=(18, 6))
    x     = np.arange(len(AGG_KEYS))
    means = [np.mean(agg_data[k]) if agg_data[k] else 0 for k in AGG_KEYS]
    stds  = [np.std(agg_data[k])  if agg_data[k] else 0 for k in AGG_KEYS]
    bars  = ax.bar(x, means, yerr=stds, capsize=5, color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([agg_labels[k] for k in AGG_KEYS], fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.12)
    ax.set_title("Aggregate Metrics — Mean ± Std (CV)")
    ax.grid(axis="y", alpha=0.4)
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.02,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "bar_aggregate.png"),
                dpi=120, bbox_inches="tight")
    plt.close(fig)

    for metric in PC_KEYS:
        means_pc, stds_pc = [], []
        for name in class_names:
            vals = [
                m["per_class"][name].get(metric, np.nan)
                for m in fold_metrics_list
                if not np.isnan(m["per_class"][name].get(metric, np.nan))
            ]
            means_pc.append(np.mean(vals) if vals else 0)
            stds_pc.append(np.std(vals)   if vals else 0)
        fig, ax = plt.subplots(figsize=(10, 5))
        x    = np.arange(len(class_names))
        bars = ax.bar(x, means_pc, yerr=stds_pc, capsize=5,
                      color=_COLORS, alpha=0.9, edgecolor="grey", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, fontsize=10)
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f"Per-Class {metric.capitalize()} — Mean ± Std (CV)")
        ax.set_ylim(0, 1.12)
        ax.grid(axis="y", alpha=0.4)
        for bar, mean, std in zip(bars, means_pc, stds_pc):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + std + 0.02,
                    f"{mean:.3f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"bar_perclass_{metric}.png"),
                    dpi=120, bbox_inches="tight")
        plt.close(fig)

    print(f"  Metric bar charts saved → {save_dir}")


# ── Patient scatter ───────────────────────────────────────────────────────────

def plot_patient_scatter(
    patient_df: pd.DataFrame,
    save_dir:   str,
    tag:        str,
) -> None:
    """
    Scatter plot of mean confidence vs. prediction stability, coloured
    by per-patient accuracy.

    Saves ``patient_scatter_{tag}.png`` and ``patient_metrics_{tag}.csv``.
    Silently skips if *patient_df* is empty (no multi-image patients).
    """
    if patient_df.empty:
        print("  [WARN] No multi-image patients — skipping scatter.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        patient_df["mean_confidence"], patient_df["consistency"],
        c=patient_df["accuracy"], cmap="RdYlGn", vmin=0, vmax=1,
        alpha=0.75, edgecolors="grey", linewidths=0.5, s=60,
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Per-patient Accuracy", fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    ax.set_xlabel("Mean Confidence  [mean p(true class)]", fontsize=14)
    ax.set_ylabel(r"Prediction Stability  $[1 - H / \log K]$", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"patient_scatter_{tag}.png"), dpi=120)
    plt.close(fig)
    patient_df.to_csv(
        os.path.join(save_dir, f"patient_metrics_{tag}.csv"), index=False)


# ── Timing ────────────────────────────────────────────────────────────────────

def save_timing(timing_records: list[dict], save_dir: str) -> None:
    """Write timing records to ``timing.csv`` in *save_dir*."""
    pd.DataFrame(timing_records).to_csv(
        os.path.join(save_dir, "timing.csv"), index=False)


# ── Curriculum plots ──────────────────────────────────────────────────────────

def plot_curriculum_stage_comparison(
    pretrain_metrics_list: list[dict],
    finetune_metrics_list: list[dict],
    save_dir:              str,
) -> None:
    """
    Grouped bar chart comparing Stage 1 (pre-train) vs Stage 2 (fine-tune)
    validation performance across folds.

    Saves ``curriculum_stage_comparison.png``.
    """
    os.makedirs(save_dir, exist_ok=True)
    keys_to_plot = ["accuracy", "f1", "auroc", "concordance_pm1"]
    labels_map   = {
        "accuracy":        "Accuracy",
        "f1":              "F1 (Macro)",
        "auroc":           "AUROC",
        "concordance_pm1": "±1 Concordance",
    }
    s1_means, s1_stds, s2_means, s2_stds = [], [], [], []
    for k in keys_to_plot:
        s1_vals = [m[k] for m in pretrain_metrics_list if not np.isnan(m.get(k, np.nan))]
        s2_vals = [m[k] for m in finetune_metrics_list if not np.isnan(m.get(k, np.nan))]
        s1_means.append(np.mean(s1_vals) if s1_vals else 0)
        s1_stds.append(np.std(s1_vals)   if s1_vals else 0)
        s2_means.append(np.mean(s2_vals) if s2_vals else 0)
        s2_stds.append(np.std(s2_vals)   if s2_vals else 0)

    x     = np.arange(len(keys_to_plot))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, s1_means, width, yerr=s1_stds, capsize=5,
                   label="Stage 1 — Noisy Pre-train",
                   color="#7bafd4", alpha=0.85, edgecolor="grey")
    bars2 = ax.bar(x + width / 2, s2_means, width, yerr=s2_stds, capsize=5,
                   label="Stage 2 — Clinical Fine-tune",
                   color="#e07b54", alpha=0.85, edgecolor="grey")
    for bars, means, stds in [(bars1, s1_means, s1_stds), (bars2, s2_means, s2_stds)]:
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + std + 0.015,
                    f"{mean:.3f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([labels_map[k] for k in keys_to_plot], fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title(
        "Curriculum Learning: Stage 1 vs Stage 2 Validation Performance\n"
        "(Mean ± Std across folds)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "curriculum_stage_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Curriculum stage comparison plot saved → {save_dir}")


def plot_curriculum_learning_curves(
    pretrain_curves_list: list[dict],
    finetune_curves_list: list[dict],
    save_dir:             str,
    n_folds:              int,
) -> None:
    """
    Per-fold learning curves spanning both curriculum stages.

    Blue = Stage 1 (pre-train), Red = Stage 2 (fine-tune).
    A vertical dashed line marks the stage boundary.

    Saves ``curriculum_curves_fold{k}.png`` for k in 1…n_folds.
    """
    os.makedirs(save_dir, exist_ok=True)
    for fold_idx in range(n_folds):
        pc     = pretrain_curves_list[fold_idx]
        fc     = finetune_curves_list[fold_idx]
        offset = len(pc["train_losses"])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, s1_tr, s1_va, s2_tr, s2_va, ylabel in [
            (axes[0], pc["train_losses"], pc["val_losses"],
             fc["train_losses"], fc["val_losses"], "Loss"),
            (axes[1], pc["train_accs"],   pc["val_accs"],
             fc["train_accs"],   fc["val_accs"],   "Accuracy"),
        ]:
            ax.plot(s1_tr, color="steelblue", label="S1 Train", linewidth=1.8)
            ax.plot(s1_va, color="steelblue", linestyle="--", label="S1 Val", linewidth=1.8)
            ax.plot(range(offset, offset + len(s2_tr)), s2_tr,
                    color="tomato", label="S2 Train", linewidth=1.8)
            ax.plot(range(offset, offset + len(s2_va)), s2_va,
                    color="tomato", linestyle="--", label="S2 Val", linewidth=1.8)
            ax.axvline(offset - 0.5, color="black", linestyle=":",
                       linewidth=1.5, alpha=0.7, label="Stage boundary")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.set_title(f"Fold {fold_idx+1} — {ylabel}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"Curriculum Learning Curves — Fold {fold_idx+1}\n"
            f"(Blue=Stage 1 noisy pre-train, Red=Stage 2 clinical fine-tune)",
            fontsize=11, fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(
            os.path.join(save_dir, f"curriculum_curves_fold{fold_idx+1}.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)
    print(f"  Curriculum learning curves saved → {save_dir}")


# ── Final model curves ────────────────────────────────────────────────────────

def save_final_model_curves(
    train_losses: list[float],
    train_accs:   list[float],
    save_dir:     str,
) -> None:
    """
    Save final-model training loss and accuracy curves.

    Outputs: ``final_model_curves.png``, ``final_model_curves.csv``.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(train_losses, color="steelblue")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Final Model — Training Loss")
    axes[0].grid(True)
    axes[1].plot(train_accs, color="seagreen")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Final Model — Training Accuracy")
    axes[1].grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "final_model_curves.png"), dpi=120)
    plt.close(fig)
    pd.DataFrame({
        "epoch":      range(1, len(train_losses) + 1),
        "train_loss": train_losses,
        "train_acc":  train_accs,
    }).to_csv(os.path.join(save_dir, "final_model_curves.csv"), index=False)
