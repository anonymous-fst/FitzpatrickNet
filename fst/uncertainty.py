"""
fst.uncertainty
===============
Monte-Carlo Dropout uncertainty quantification.

Functions
---------
mc_dropout_predict      Run T stochastic forward passes and return
                        mean probabilities, predictive entropy, and
                        mutual information (epistemic uncertainty).
plot_uncertainty_boxplot
                        Per-class box-plot comparing predictive entropy
                        for correctly vs. incorrectly classified samples.
"""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ── MC-Dropout inference ──────────────────────────────────────────────────────

def mc_dropout_predict(
    model:      "torch.nn.Module",
    dataloader: "torch.utils.data.DataLoader",
    device:     "torch.device",
    n_passes:   int = 20,
) -> dict:
    """
    Run *n_passes* stochastic forward passes with dropout active.

    The model is set to ``train()`` mode so that dropout layers remain
    active, then restored to ``eval()`` after inference.

    Supports both 7-element and 8-element batch tuples (the 8-element
    form includes a provenance-weight tensor as the last element).

    Parameters
    ----------
    model      : trained model with dropout layers.
    dataloader : validation or test DataLoader.
    device     : target compute device.
    n_passes   : number of stochastic forward passes T.

    Returns
    -------
    dict with keys:
        ``"mean_probs"``         — (N, K) mean predicted probabilities.
        ``"predictive_entropy"`` — (N,)   H[E[p]] = total uncertainty.
        ``"mutual_information"`` — (N,)   I = H[E[p]] - E[H[p]]
                                          (epistemic uncertainty).
    """
    model.train()                          # keep dropout active
    all_pass_probs: list[np.ndarray] = []

    with torch.no_grad():
        for _ in range(n_passes):
            pass_probs: list[np.ndarray] = []
            for batch in dataloader:
                # Support 7-element (legacy) and 8-element (current) batches
                if len(batch) == 8:
                    imgs, _, _, ita, has_ita, _, _, _ = batch
                else:
                    imgs, _, _, ita, has_ita, _, _ = batch

                imgs    = imgs.to(device)
                ita     = ita.to(device)
                has_ita = has_ita.to(device)
                probs   = F.softmax(model(imgs, ita, has_ita), dim=1)
                pass_probs.append(probs.cpu().numpy())
            all_pass_probs.append(np.concatenate(pass_probs, axis=0))

    stacked    = np.stack(all_pass_probs, axis=0)   # (T, N, K)
    mean_probs = stacked.mean(axis=0)               # (N, K)

    # Predictive entropy: H[E[p]]
    pred_entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-8), axis=1)

    # Per-pass entropy: E[H[p]]
    per_pass_entropy = -np.sum(stacked * np.log(stacked + 1e-8), axis=2)

    # Mutual information (epistemic uncertainty)
    mutual_info = pred_entropy - per_pass_entropy.mean(axis=0)

    model.eval()
    return {
        "mean_probs":         mean_probs,
        "predictive_entropy": pred_entropy,
        "mutual_information": mutual_info,
    }


# ── Visualisation ─────────────────────────────────────────────────────────────

# FST skin-tone palette (light → dark)
_COLORS = ["#F6D5B0", "#E8B88A", "#C68642", "#8D5524", "#4A2912", "#2C1A0E"]


def plot_uncertainty_boxplot(
    unc_df:      pd.DataFrame,
    class_names: list[str],
    tag:         str,
    save_dir:    str,
    n_passes:    int = 20,
) -> None:
    """
    2 × 3 grid of box-plots comparing predictive entropy for correctly
    vs. incorrectly classified samples, one panel per FST class.

    Parameters
    ----------
    unc_df      : DataFrame with columns ``true_label``, ``pred_label``,
                  ``predictive_entropy``.
    class_names : list of K class name strings.
    tag         : string appended to the output filename.
    save_dir    : directory in which to save the PNG.
    n_passes    : number of MC-Dropout passes (shown in the title).
    """
    fig, axes  = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
    axes       = axes.flatten()
    global_max = unc_df["predictive_entropy"].quantile(0.99)

    for i, (name, color) in enumerate(zip(class_names, _COLORS)):
        ax          = axes[i]
        class_df    = unc_df[unc_df["true_label"] == i]
        correct_e   = class_df[class_df["pred_label"] == i]["predictive_entropy"].values
        incorrect_e = class_df[class_df["pred_label"] != i]["predictive_entropy"].values

        bp = ax.boxplot(
            [correct_e, incorrect_e],
            labels=[
                f"Correct\n(n={len(correct_e)})",
                f"Incorrect\n(n={len(incorrect_e)})",
            ],
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
        )
        for patch, c in zip(bp["boxes"], ["#4CAF50", "#F44336"]):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)

        for j, (d, c) in enumerate(zip([correct_e, incorrect_e],
                                        ["#4CAF50", "#F44336"])):
            if len(d):
                jitter = np.random.uniform(-0.1, 0.1, size=len(d))
                ax.scatter(np.full(len(d), j + 1) + jitter, d,
                           color=c, alpha=0.25, s=8, zorder=3)

        ax.axhline(np.log(len(class_names)), color="grey",
                   linestyle=":", linewidth=1, alpha=0.5)
        ax.set_title(name, fontsize=10, fontweight="bold", pad=4)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
        ax.set_ylabel("Predictive Entropy H[E[p]]", fontsize=8)
        ax.set_ylim(0, global_max * 1.05)
        ax.grid(axis="y", alpha=0.3)

        for j, d in enumerate([correct_e, incorrect_e]):
            if len(d):
                ax.text(j + 1, np.median(d) + global_max * 0.02,
                        f"{np.median(d):.3f}",
                        ha="center", va="bottom", fontsize=7, fontweight="bold")

    fig.suptitle(
        f"MC-Dropout Predictive Entropy by FST Class — {tag}\n"
        f"(Correct vs Incorrect, T={n_passes} passes)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(save_dir, f"uncertainty_entropy_boxplot_{tag}.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)
