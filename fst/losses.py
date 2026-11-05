"""
fst.losses
==========
Loss functions shared by both the CNN and ViT training pipelines.

Classes
-------
FocalKLLoss
    KL-divergence loss with optional focal weighting, class-frequency
    weighting, and per-sample provenance weighting.

Functions
---------
compute_class_weights       Inverse-frequency class weights as a tensor.
patient_consistency_loss    KL penalty encouraging consistent predictions
                            across images of the same patient.
supervised_contrastive_loss Supervised contrastive loss using patient
                            identity and class label as positivity signal.
"""

from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Primary loss ──────────────────────────────────────────────────────────────

class FocalKLLoss(nn.Module):
    """
    Per-sample KL divergence with three optional modifiers applied
    multiplicatively in order:

    1. **Focal weighting** — down-weights easy samples via
       ``(1 - p_true)^γ``, identical to the focal-loss formulation but
       applied to the KL term rather than cross-entropy.
    2. **Class-frequency weighting** — scales each sample by the
       inverse frequency of its hard label.
    3. **Provenance weighting** — scales each sample by a per-dataset
       scalar supplied by the dataloader.

    Parameters
    ----------
    class_weights     : (K,) float tensor or None.
    focal_gamma       : focal exponent γ (default 2.0).
    use_focal         : whether to apply focal weighting.
    use_class_weights : whether to apply class-frequency weighting.
    """

    def __init__(
        self,
        class_weights:     torch.Tensor | None = None,
        focal_gamma:       float = 2.0,
        use_focal:         bool  = True,
        use_class_weights: bool  = True,
    ):
        super().__init__()
        self.class_weights     = class_weights
        self.focal_gamma       = focal_gamma
        self.use_focal         = use_focal
        self.use_class_weights = use_class_weights
        self.kl_loss           = nn.KLDivLoss(reduction="none")

    def forward(
        self,
        logits:       torch.Tensor,
        soft_targets: torch.Tensor,
        hard_labels:  torch.Tensor,
        prov_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits       : (B, K) raw model outputs.
        soft_targets : (B, K) soft label distribution (rows sum to 1).
        hard_labels  : (B,)   integer ground-truth class indices.
        prov_weights : (B,)   per-sample provenance scalars, or None.

        Returns
        -------
        Scalar mean loss.
        """
        log_probs = F.log_softmax(logits, dim=1)
        probs     = torch.exp(log_probs)
        kl_per    = self.kl_loss(log_probs, soft_targets).sum(dim=1)

        if self.use_focal:
            p_t    = (probs.gather(1, hard_labels.unsqueeze(1))
                          .squeeze(1).clamp(1e-8, 1.0))
            kl_per = ((1.0 - p_t) ** self.focal_gamma) * kl_per

        if self.use_class_weights and self.class_weights is not None:
            cw     = self.class_weights.to(logits.device)
            kl_per = cw[hard_labels] * kl_per

        if prov_weights is not None:
            kl_per = prov_weights.to(logits.device) * kl_per

        return kl_per.mean()


# ── Class-weight helper ───────────────────────────────────────────────────────

def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Compute inverse-frequency class weights normalised to mean 1.

    Parameters
    ----------
    labels      : (N,) int array of hard labels.
    num_classes : total number of classes.

    Returns
    -------
    (num_classes,) float32 tensor.
    """
    counts  = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts  = np.maximum(counts, 1)
    weights = 1.0 / counts
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


# ── Auxiliary losses ──────────────────────────────────────────────────────────

def patient_consistency_loss(
    logits:      torch.Tensor,
    patient_ids: list[str],
    weight:      float = 0.1,
) -> torch.Tensor:
    """
    Penalise inconsistent predictions across images from the same patient.

    For each patient group (≥ 2 images in the batch), compute the KL
    divergence of each image's predicted distribution from the group mean.
    The loss is the weighted mean of these per-group KL terms.

    Singleton patients (``__unique_*`` IDs) are excluded.

    Parameters
    ----------
    logits      : (B, K) raw model outputs.
    patient_ids : list of B patient ID strings.
    weight      : scalar multiplier applied to the final loss term.

    Returns
    -------
    Scalar tensor (0.0 if no multi-image patients are present in the batch).
    """
    probs      = F.softmax(logits, dim=1)
    log_probs  = F.log_softmax(logits, dim=1)
    pid_to_idx: dict[str, list[int]] = defaultdict(list)

    for i, pid in enumerate(patient_ids):
        if not pid.startswith("__unique_"):
            pid_to_idx[pid].append(i)

    kl_terms = []
    for pid, indices in pid_to_idx.items():
        if len(indices) < 2:
            continue
        idx_t   = torch.tensor(indices, device=logits.device)
        p_group = probs[idx_t]
        lp_grp  = log_probs[idx_t]
        p_mean  = p_group.mean(dim=0, keepdim=True)
        kl      = (p_group * (lp_grp - p_mean.log().clamp(-100))).sum(dim=1)
        kl_terms.append(kl.mean())

    if not kl_terms:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    return weight * torch.stack(kl_terms).mean()


def supervised_contrastive_loss(
    logits:      torch.Tensor,
    patient_ids: list[str],
    hard_labels: torch.Tensor,
    weight:      float = 0.05,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Supervised contrastive loss using patient identity and class label
    as the positivity criterion.

    Two samples are considered a positive pair when they share the same
    patient ID (excluding singletons) **or** the same hard label.

    Parameters
    ----------
    logits      : (B, K) raw model outputs.
    patient_ids : list of B patient ID strings.
    hard_labels : (B,) integer ground-truth class indices.
    weight      : scalar multiplier (set to 0.0 to disable).
    temperature : softmax temperature for the similarity matrix.

    Returns
    -------
    Scalar tensor.
    """
    if weight == 0.0:
        return torch.tensor(0.0, device=logits.device)

    probs  = F.normalize(F.softmax(logits, dim=1), dim=1)
    B      = probs.size(0)
    sim    = torch.mm(probs, probs.t()) / temperature

    pos_mask = torch.zeros(B, B, dtype=torch.bool, device=logits.device)
    for i in range(B):
        for j in range(B):
            if i == j:
                continue
            same_patient = (
                not patient_ids[i].startswith("__unique_")
                and patient_ids[i] == patient_ids[j]
            )
            if same_patient or hard_labels[i] == hard_labels[j]:
                pos_mask[i, j] = True

    exp_sim   = torch.exp(sim)
    diag_mask = ~torch.eye(B, dtype=torch.bool, device=logits.device)
    denom     = (exp_sim * diag_mask).sum(dim=1)
    pos_count = pos_mask.sum(dim=1).clamp(min=1)
    loss_per  = -(
        (sim * pos_mask).sum(dim=1) / pos_count
        - torch.log(denom + 1e-8)
    )
    return weight * loss_per.mean()
