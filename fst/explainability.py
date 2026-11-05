"""
fst.explainability
==================
Grad-CAM explainability utilities shared by both training pipelines.

The ViT-specific Attention Rollout implementation lives in
``train_eval_vit_ita.py`` because it depends on ViT attention block
internals that are not present in the CNN backbone.

Classes
-------
MinimalGradCAM
    Hook-based Grad-CAM that works with any model exposing a
    ``get_gradcam_layer()`` method.

Functions
---------
save_three_gradcam_files    Save panel, heatmap, and overlay PNGs for one sample.
generate_and_save_gradcam   Collect class-balanced samples and run Grad-CAM.
"""

import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ── MinimalGradCAM ────────────────────────────────────────────────────────────

class MinimalGradCAM:
    """
    Hook-based Grad-CAM compatible with both CNN and ViT backbones.

    The target layer is obtained by calling ``model.get_gradcam_layer()``,
    which each architecture implements differently:

    * **CNN** (ResNet-18 / EfficientNet-B0): hooks the last conv layer,
      producing a spatial activation map directly.
    * **ViT**: hooks the LayerNorm before the last self-attention block.
      The patch sequence (excluding the [CLS] token) is reshaped into a
      2-D grid before interpolation.

    Parameters
    ----------
    model : trained model with a ``get_gradcam_layer()`` method.
    """

    def __init__(self, model: "torch.nn.Module"):
        self.model        = model
        self.target_layer = model.get_gradcam_layer()
        self.gradients    = None
        self.activations  = None
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(
        self,
        input_tensor:   "torch.Tensor",
        ita_tensor:     "torch.Tensor",
        has_ita_tensor: "torch.Tensor",
        target_class:   int | None = None,
    ) -> np.ndarray:
        """
        Compute a normalised Grad-CAM saliency map.

        For ViT activations the patch sequence dimension is reshaped into
        a square grid (CLS token at index 0 is discarded).  For CNN
        activations the standard spatial pooling path is used.

        Parameters
        ----------
        input_tensor   : (1, C, H, W) image tensor on the model device.
        ita_tensor     : (1, 1) normalised ITA value.
        has_ita_tensor : (1, 1) ITA presence mask.
        target_class   : class index to explain; defaults to argmax.

        Returns
        -------
        (H, W) float32 array in [0, 1].
        """
        self.model.eval()
        inp = input_tensor.clone().requires_grad_(True)
        with torch.enable_grad():
            output = self.model(inp, ita_tensor, has_ita_tensor)
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            self.model.zero_grad()
            output[0, target_class].backward()

        grads = self.gradients.cpu().numpy()[0]
        acts  = self.activations.cpu().numpy()[0]

        # ── ViT path: sequence of patch tokens ────────────────
        # Shape: (seq_len, embed_dim) — drop CLS token at index 0
        if grads.ndim == 2:
            patch_grads = grads[1:, :]
            patch_acts  = acts[1:, :]
            weights     = patch_grads.mean(axis=0)          # (embed_dim,)
            cam_flat    = np.maximum((patch_acts * weights).sum(axis=1), 0)
            n_side      = int(cam_flat.shape[0] ** 0.5)
            cam         = cam_flat.reshape(n_side, n_side)

        # ── CNN path: spatial feature map ─────────────────────
        # Shape: (channels, H_feat, W_feat)
        else:
            weights = grads.mean(axis=(1, 2))               # (channels,)
            cam     = np.maximum((weights[:, None, None] * acts).sum(axis=0), 0)

        cam_t = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0).float()
        cam_r = F.interpolate(
            cam_t,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear", align_corners=False,
        )
        cam = cam_r.squeeze().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# ── File-save helpers ─────────────────────────────────────────────────────────

def save_three_gradcam_files(
    img_show:   np.ndarray,
    cam:        np.ndarray,
    true_label: str,
    pred_label: str,
    save_dir:   str,
    sample_id:  str,
    act_conc:   float,
    iou:        float,
) -> None:
    """
    Save three Grad-CAM output files for one sample:

    * ``{sample_id}_panel.png``   — side-by-side: original / heatmap / overlay.
    * ``{sample_id}_heatmap.png`` — heatmap only.
    * ``{sample_id}_overlay.png`` — overlay only.

    Parameters
    ----------
    img_show   : (H, W, 3) float32 image in [0, 1].
    cam        : (H, W) float32 saliency map in [0, 1].
    true_label : ground-truth class name string.
    pred_label : predicted class name string.
    save_dir   : output directory.
    sample_id  : filename prefix.
    act_conc   : activation concentration metric (fraction of CAM mass
                 inside the segmentation mask).
    iou        : IoU between thresholded CAM and segmentation mask.
    """
    correct_str = "✓" if true_label == pred_label else "✗"
    fig, axes   = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={"wspace": 0.05})
    fig.suptitle(
        f"True: {true_label}   Pred: {pred_label} {correct_str}   "
        f"Act. Conc.: {act_conc:.2f}   IoU: {iou:.2f}",
        fontsize=11, y=1.02, va="bottom",
    )
    axes[0].imshow(img_show); axes[0].axis("off"); axes[0].set_title("Original", pad=4)
    axes[1].imshow(cam, cmap="jet", vmin=0, vmax=1)
    axes[1].axis("off"); axes[1].set_title("Activation Map", pad=4)
    axes[2].imshow(img_show)
    axes[2].imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    axes[2].axis("off"); axes[2].set_title("Overlay", pad=4)
    fig.savefig(os.path.join(save_dir, f"{sample_id}_panel.png"),
                dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    for suffix, draw in [
        ("_heatmap.png", lambda ax: ax.imshow(cam, cmap="jet", vmin=0, vmax=1)),
        ("_overlay.png", lambda ax: (ax.imshow(img_show),
                                      ax.imshow(cam, cmap="jet", alpha=0.45))),
    ]:
        fig, ax = plt.subplots(figsize=(5, 5))
        draw(ax)
        ax.axis("off")
        fig.savefig(os.path.join(save_dir, f"{sample_id}{suffix}"),
                    dpi=150, bbox_inches="tight", pad_inches=0.0)
        plt.close(fig)


# ── Main Grad-CAM generation function ────────────────────────────────────────

def generate_and_save_gradcam(
    model:       "torch.nn.Module",
    dataloader:  "torch.utils.data.DataLoader",
    class_names: list[str],
    save_dir:    str,
    device:      "torch.device",
    cfg:         dict,
    mean_arr:    np.ndarray,
    std_arr:     np.ndarray,
    prefix:      str = "fold",
    pred_labels: np.ndarray | None = None,
    true_labels: np.ndarray | None = None,
) -> None:
    """
    Collect a class-balanced sample of images, run Grad-CAM, and save
    all output files plus a metrics CSV.

    The function collects ``cfg["gradcam_n_samples"]`` images spread as
    evenly as possible across classes, then generates one Grad-CAM
    visualisation per image.

    Parameters
    ----------
    model       : trained model.
    dataloader  : DataLoader to collect samples from.
    class_names : list of K class name strings.
    save_dir    : output directory (created if absent).
    device      : compute device.
    cfg         : experiment config dict (reads ``gradcam_n_samples``,
                  ``num_classes``).
    mean_arr    : (3,) ImageNet normalisation mean (for de-normalisation).
    std_arr     : (3,) ImageNet normalisation std (for de-normalisation).
    prefix      : filename prefix (e.g. ``"fold1"``, ``"test"``).
    pred_labels : (N,) predicted labels aligned with the dataset indices,
                  used to annotate each panel with the model's prediction.
    true_labels : unused (kept for API symmetry with callers).
    """
    grad_cam = MinimalGradCAM(model)
    os.makedirs(save_dir, exist_ok=True)
    metrics_list = []
    n_needed     = cfg["gradcam_n_samples"]

    # ── Collect class-balanced samples ────────────────────────
    class_to_samples: dict = defaultdict(list)
    for batch in dataloader:
        if len(batch) == 8:
            imgs_b, _, hard_b, ita_b, has_ita_b, _, idxs_b, _ = batch
        else:
            imgs_b, _, hard_b, ita_b, has_ita_b, _, idxs_b = batch
        for i in range(imgs_b.size(0)):
            lbl = int(hard_b[i].item())
            class_to_samples[lbl].append((
                imgs_b[i], ita_b[i:i+1], has_ita_b[i:i+1],
                int(idxs_b[i].item()),
            ))
        if (len(class_to_samples) == cfg["num_classes"]
                and sum(len(v) for v in class_to_samples.values()) >= n_needed):
            break

    selected = []
    class_keys = sorted(class_to_samples.keys())
    i = 0
    while len(selected) < n_needed:
        cls = class_keys[i % len(class_keys)]
        if class_to_samples[cls]:
            img_t, ita_t, has_ita_t, ds_idx = class_to_samples[cls].pop(0)
            selected.append((img_t, ita_t, has_ita_t, cls, ds_idx))
        i += 1
        if all(len(v) == 0 for v in class_to_samples.values()):
            break

    # ── Generate and save ─────────────────────────────────────
    dev = next(model.parameters()).device
    for idx, (img_t, ita_t, has_ita_t, true_lbl, ds_idx) in enumerate(selected):
        img_inp  = img_t.unsqueeze(0).to(dev)
        cam      = grad_cam.generate(img_inp, ita_t.to(dev), has_ita_t.to(dev))
        img_np   = img_t.numpy()
        mask_np  = (np.abs(img_np).sum(axis=0) > 1e-5).astype(float)
        act_conc = np.sum(cam * mask_np) / (np.sum(cam) + 1e-8)
        cam_bin  = (cam > 0.5).astype(float)
        iou      = (np.logical_and(cam_bin, mask_np).sum()
                    / (np.logical_or(cam_bin, mask_np).sum() + 1e-8))
        pred_lbl = (int(pred_labels[ds_idx])
                    if pred_labels is not None and ds_idx < len(pred_labels)
                    else None)
        metrics_list.append({
            "sample_idx":               idx + 1,
            "dataset_idx":              ds_idx,
            "true_label":               class_names[true_lbl],
            "pred_label":               class_names[pred_lbl] if pred_lbl is not None else "N/A",
            "correct":                  (pred_lbl == true_lbl) if pred_lbl is not None else None,
            "activation_concentration": round(float(act_conc), 4),
            "iou":                      round(float(iou), 4),
        })
        img_show = np.clip(std_arr * img_np.transpose(1, 2, 0) + mean_arr, 0, 1)
        save_three_gradcam_files(
            img_show, cam,
            class_names[true_lbl],
            class_names[pred_lbl] if pred_lbl is not None else "N/A",
            save_dir, f"{prefix}_gradcam_{idx+1}",
            act_conc, iou,
        )

    pd.DataFrame(metrics_list).to_csv(
        os.path.join(save_dir, f"{prefix}_gradcam_metrics.csv"), index=False)
    print(f"  Grad-CAM saved → {save_dir}  ({len(metrics_list)} samples)")
