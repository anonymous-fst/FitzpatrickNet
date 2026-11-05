#!/usr/bin/env python3
"""
train_eval_cnn_ita.py
=====================
Multimodal CNN with late ITA fusion for Fitzpatrick Skin Type (FST)
classification.

Shared logic (soft labels, losses, metrics, plotting, uncertainty,
explainability) is imported from the fst package.  Only CNN-specific
code lives here:

  - ITANormalizer          (data-driven z-score, serialisable)
  - SkinDatasetITA         (explicit normalizer injection)
  - ITAProjector           (MLP + learned missing token)
  - ResNet18ITA            (backbone + ITA fusion head)
  - EfficientNetB0ITA      (backbone + ITA fusion head)
  - build_model            (factory)
  - freeze_cnn_layers / unfreeze_all / build_finetune_optimizer
  - build_scheduler        (step | cosine)
  - build_dataloaders / build_test_dataloader
  - load_fold_df / load_curriculum_fold_df
  - _run_one_epoch / train_one_fold / train_curriculum_one_fold
  - evaluate
  - run_cv / run_curriculum_cv
  - run_held_out_test / run_curriculum_held_out_test
  - run_optuna
  - parse_args / apply_args_to_cfg / main

Architectures: ResNet-18 | EfficientNet-B0
"""

# ============================================================
# 0. IMPORTS
# ============================================================
import os, sys, copy, time, random, warnings, argparse, json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as tv_models
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

from PIL import Image
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("[WARN] optuna not installed — Optuna mode disabled.")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
Image.MAX_IMAGE_PIXELS = None

# ── fst package imports ───────────────────────────────────────────────────────
from fst.config import (
    DATA_SPLIT_VARIANTS,
    fold_csv_template      as _fold_csv_tmpl_from_cfg,
    test_csv_path          as _test_csv_from_cfg,
)
from fst.reproducibility import set_seed
from fst.soft_labels import (
    make_soft_labels,
    make_soft_labels_adaptive,
    anneal_alpha,
    build_soft_labels      as _build_soft_labels,
)
from fst.losses import (
    FocalKLLoss,
    compute_class_weights,
    patient_consistency_loss,
    supervised_contrastive_loss,
)
from fst.sampling import (
    BalancedBatchSampler,
    mixup_batch,
    get_prov_weights       as _get_prov_weights,
)
from fst.metrics import (
    AGG_KEYS, PC_KEYS,
    compute_extended_metrics,
    bootstrap_ci,
    compute_patient_metrics,
    expected_calibration_error,
    mc_dropout_ece,
    quadratic_weighted_kappa,
)
from fst.uncertainty import (
    mc_dropout_predict,
    plot_uncertainty_boxplot,
)
from fst.plotting import (
    CLASS_NAMES, _COLORS,
    save_curves            as _save_curves,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve,
    plot_calibration_curve,
    plot_metric_bars,
    plot_patient_scatter,
    save_timing,
    plot_curriculum_stage_comparison,
    plot_curriculum_learning_curves,
    save_final_model_curves as _save_final_model_curves,
)
from fst.explainability import (
    MinimalGradCAM,
    generate_and_save_gradcam,
)


# ============================================================
# 1. GLOBAL CONFIGURATION
# ============================================================
CFG = dict(
    # ── Paths ────────────────────────────────────────────────
    fold_csv_dir   = "data_split",
    fold_csv_tmpl  = "train_val_dataset_metadata_extended_fold_{fold}.csv",
    test_csv_path  = "data_split/test_dataset_metadata_extended.csv",
    test_csv_path_balanced = "data_split/test_dataset_metadata_extended_balanced.csv",
    output_root    = "results/multimodal_cnn_ita",
    debug_root     = "results/debug/multimodal_cnn_ita",
    optuna_root    = "results/multimodal_cnn_ita/optuna",

    # ── Experiment identity ───────────────────────────────────
    arch           = "resnet18",
    experiment_tag = "baseline",

    # ── Data split variant ────────────────────────────────────
    data_split_variant = "standard",

    # ── ITA fusion ───────────────────────────────────────────
    ita_col        = "ITA_value_spectral_spatial",
    ita_mean       = 20.0,
    ita_std        = 25.0,
    ita_proj_dim   = 8,
    use_ita_fusion = True,

    # ── Training ─────────────────────────────────────────────
    num_folds      = 5,
    num_epochs     = 50,
    batch_size     = 32,
    learning_rate  = 5e-5,
    weight_decay   = 3e-4,

    # ── Scheduler ────────────────────────────────────────────
    scheduler        = "cosine",
    cosine_T0_epochs = 5,
    cosine_T_mult    = 2,
    cosine_eta_min   = 1e-6,
    scheduler_step   = 7,
    scheduler_gamma  = 0.1,

    num_workers    = 4,
    seed           = 42,

    # ── Early stopping ────────────────────────────────────────
    patience       = 8,
    min_delta      = 2e-4,

    # ── Soft labels ───────────────────────────────────────────
    use_soft_labels      = True,
    soft_label_alpha     = 0.15,
    use_alpha_annealing  = True,
    alpha_start          = 0.20,
    alpha_end            = 0.10,

    # ── Loss components ───────────────────────────────────────
    use_focal_loss    = True,
    focal_gamma       = 2.0,
    use_class_weights = True,

    # ── Sampling ─────────────────────────────────────────────
    use_weighted_sampler       = True,
    use_balanced_batch_sampler = False,
    use_minority_augment       = True,
    minority_classes           = [0, 5],

    # ── Mixup ─────────────────────────────────────────────────
    use_mixup    = False,
    mixup_alpha  = 0.4,
    mixup_prob   = 0.5,

    # ── Exposure augmentation ─────────────────────────────────
    use_exposure_augment  = False,
    exposure_aug_prob     = 0.3,

    # ── Patient consistency ───────────────────────────────────
    use_patient_consistency = True,
    patient_kl_weight       = 0.1,
    use_contrastive         = False,
    contrastive_weight      = 0.05,

    # ── Uncertainty ───────────────────────────────────────────
    use_mc_dropout    = True,
    mc_dropout_passes = 20,
    mc_dropout_rate   = 0.4,

    # ── TTA ───────────────────────────────────────────────────
    use_tta = True,

    # ── Grad-CAM ─────────────────────────────────────────────
    gradcam_n_samples     = 10,
    save_all_test_gradcam = False,

    # ── Quick mode ────────────────────────────────────────────
    quick_mode              = False,
    quick_fold              = 1,
    quick_mc_dropout_passes = 5,

    # ── Optuna ───────────────────────────────────────────────
    optuna_n_trials = 30,
    optuna_fold     = 1,

    # ── Debug ────────────────────────────────────────────────
    debug_mode      = False,
    debug_n_samples = 64,
    debug_epochs    = 2,

    # ── Misc ─────────────────────────────────────────────────
    num_classes = 6,
    image_size  = 224,
    resume      = False,

    # ── Provenance-based loss weighting ──────────────────────
    use_provenance_weighting = False,
    provenance_col           = "dataset",
    provenance_loss_weights  = {
        "tla_i4i":        1.00,
        "eh_web":         0.60,
        "pad-ufes-20":    0.85,
        "passion":        0.85,
        "scin":           0.60,
        "fitzpatrick17k": 0.40,
    },

    # ── Adaptive label smoothing ──────────────────────────────
    use_adaptive_smoothing   = False,
    adaptive_alpha_map       = {
        "tla_i4i":        0.05,
        "eh_web":         0.18,
        "pad-ufes-20":    0.10,
        "passion":        0.10,
        "scin":           0.18,
        "fitzpatrick17k": 0.22,
    },

    # ── Curriculum ───────────────────────────────────────────
    curriculum_dataset_col        = "dataset",
    curriculum_pretrain_datasets  = ["fitzpatrick17k", "scin"],
    curriculum_finetune_datasets  = ["pad-ufes-20", "passion", "eh_web", "tla_i4i"],
    curriculum_finetune_epochs    = 20,
    curriculum_freeze_layers      = 2,
    curriculum_finetune_lr        = 5e-5,
    curriculum_finetune_wd        = 3e-4,
    curriculum_finetune_scheduler = "cosine",
    curriculum_finetune_alpha       = 0.08,
    curriculum_finetune_alpha_start = 0.12,
    curriculum_finetune_alpha_end   = 0.05,
    curriculum_finetune_patience  = 7,
    curriculum_run_test           = True,

    debug_curriculum_pretrain_epochs = 2,
    debug_curriculum_finetune_epochs = 2,
    debug_curriculum_n_samples       = 64,
)


# ============================================================
# 2. OUTPUT DIRECTORY BUILDER
# ============================================================
def build_experiment_name(cfg: dict) -> str:
    parts = [cfg["arch"]]
    variant = cfg.get("data_split_variant", "standard")
    if variant != "standard":
        parts.append(f"split_{variant}")
    if cfg.get("quick_mode", False):
        parts.append(f"quick_fold{cfg.get('quick_fold', 1)}")
    parts.append("ita"         if cfg["use_ita_fusion"]           else "no_ita")
    parts.append("softlabel"   if cfg["use_soft_labels"]          else "no_softlabel")
    parts.append("focal"       if cfg["use_focal_loss"]           else "no_focal")
    parts.append("clswt"       if cfg["use_class_weights"]        else "no_clswt")
    parts.append("wsampler"    if cfg["use_weighted_sampler"]     else "no_wsampler")
    parts.append("minaug"      if cfg["use_minority_augment"]     else "no_minaug")
    parts.append("mixup"       if cfg.get("use_mixup", False)     else "no_mixup")
    parts.append("exposureaug" if cfg.get("use_exposure_augment", False) else "no_exposureaug")
    parts.append("patcons"     if cfg["use_patient_consistency"]  else "no_patcons")
    if cfg.get("use_contrastive", False):
        parts.append("contrastive")
    parts.append("mcdrop"      if cfg["use_mc_dropout"]           else "no_mcdrop")
    if cfg.get("use_provenance_weighting", False):
        parts.append("provwt")
    if cfg.get("use_adaptive_smoothing", False):
        parts.append("adapsmooth")
    parts.append(cfg["experiment_tag"])
    return "_".join(parts)


def make_dirs(cfg: dict, mode: str = "train") -> dict:
    if mode in ("debug", "debug_curriculum"):
        root = Path(cfg["debug_root"])
    elif mode == "optuna":
        root = Path(cfg["optuna_root"])
    else:
        root = Path(cfg["output_root"])

    exp_name = build_experiment_name(cfg)
    base     = root / exp_name

    dirs = {
        "base":        base,
        "models":      base / "models",
        "plots":       base / "plots",
        "metrics":     base / "metrics",
        "gradcam":     base / "gradcam",
        "uncertainty": base / "uncertainty",
        "curriculum_pretrain": base / "curriculum_pretrain",
        "curriculum_finetune": base / "curriculum_finetune",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    for stage_key in ("curriculum_pretrain", "curriculum_finetune"):
        for sub in ("models", "plots", "metrics", "gradcam", "uncertainty"):
            (dirs[stage_key] / sub).mkdir(parents=True, exist_ok=True)

    with open(base / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    return {k: str(v) for k, v in dirs.items()}


# ============================================================
# 3. ITA NORMALISER  (CNN-specific: data-driven, serialisable)
# ============================================================
class ITANormalizer:
    """
    Fits mean/std on training ITA values and applies z-score normalisation.
    Unlike the ViT pipeline (which uses fixed cfg["ita_mean"]/cfg["ita_std"]),
    the CNN pipeline fits a normalizer per fold and saves it alongside the
    model checkpoint so it can be reloaded at inference time.
    """
    def __init__(self):
        self.mean_: float = 0.0
        self.std_:  float = 1.0

    def fit(self, values: np.ndarray) -> "ITANormalizer":
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            self.mean_, self.std_ = 0.0, 1.0
        else:
            self.mean_ = float(np.mean(valid))
            self.std_  = max(float(np.std(valid)), 1e-6)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.where(np.isnan(values), np.nan,
                        (values - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        return self.fit(values).transform(values)

    def state_dict(self) -> dict:
        return {"mean": self.mean_, "std": self.std_}

    def load_state_dict(self, d: dict):
        self.mean_, self.std_ = d["mean"], d["std"]


def extract_ita_array(df: pd.DataFrame, ita_col: str) -> np.ndarray:
    if ita_col not in df.columns:
        return np.full(len(df), np.nan, dtype=np.float32)
    return pd.to_numeric(df[ita_col], errors="coerce").values.astype(np.float32)


# ============================================================
# 4. DATASET  (CNN-specific: explicit ITANormalizer injection)
# ============================================================
class SkinDatasetITA(Dataset):
    """
    PyTorch Dataset for CNN training.

    Unlike SkinDatasetViTITA (which normalises ITA inline using fixed
    cfg["ita_mean"]/cfg["ita_std"]), this dataset receives pre-normalised
    ITA values from an ITANormalizer that was fit on the training split.
    This allows the same normalizer to be applied consistently at test time.
    """
    PATIENT_COLS = ["patient_id", "subject_id", "case_id", "refno"]

    def __init__(self, df, soft_labels, ita_normed, prov_weights,
                 is_train=True, cfg=None):
        self.df           = df.reset_index(drop=True)
        self.soft_labels  = soft_labels
        self.ita_normed   = ita_normed
        self.prov_weights = prov_weights
        self.is_train     = is_train
        self.cfg          = cfg or CFG
        self.patient_ids  = self._resolve_patient_ids()

    def _resolve_patient_ids(self):
        ids = []
        for idx, row in self.df.iterrows():
            pid = None
            for col in self.PATIENT_COLS:
                if col in self.df.columns:
                    val = row[col]
                    if pd.notna(val) and str(val).strip() not in ("", "nan"):
                        pid = str(val).strip()
                        break
            ids.append(pid if pid is not None else f"__unique_{idx}__")
        return ids

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row       = self.df.iloc[idx]
        img_path  = str(row["image_path_512"]).strip()
        mask_path = row.get("mask_path_512", None)

        image = Image.open(img_path).convert("RGB")
        has_mask = (mask_path is not None
                    and str(mask_path).strip() not in ("", "nan", "None")
                    and os.path.exists(str(mask_path).strip()))
        mask = Image.open(str(mask_path).strip()).convert("L") if has_mask else None

        hard_label  = int(row["label"])
        is_minority = (self.cfg["use_minority_augment"]
                       and hard_label in self.cfg["minority_classes"])

        image, mask = self._apply_transforms(image, mask, is_minority)

        image_tensor = TF.to_tensor(image)
        mask_tensor  = (TF.to_tensor(mask) if mask is not None
                        else torch.ones(1, image_tensor.shape[1],
                                           image_tensor.shape[2]))
        normalize    = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std =[0.229, 0.224, 0.225])
        image_tensor = normalize(image_tensor) * mask_tensor

        ita_raw      = self.ita_normed[idx]
        has_real_ita = not np.isnan(ita_raw)
        ita_val      = float(ita_raw) if has_real_ita else 0.0

        ita_tensor  = torch.tensor([ita_val],                      dtype=torch.float32)
        has_ita_t   = torch.tensor([1.0 if has_real_ita else 0.0], dtype=torch.float32)
        soft_label  = torch.tensor(self.soft_labels[idx],          dtype=torch.float32)
        prov_weight = torch.tensor(self.prov_weights[idx],         dtype=torch.float32)

        return (image_tensor, soft_label, hard_label,
                ita_tensor, has_ita_t, self.patient_ids[idx], idx, prov_weight)

    def _apply_transforms(self, image, mask, is_minority):
        sz = self.cfg["image_size"]
        if self.is_train:
            image = TF.resize(image, (int(sz * 1.14), int(sz * 1.14)))
            if mask:
                mask = TF.resize(mask, (int(sz * 1.14), int(sz * 1.14)),
                                  interpolation=transforms.InterpolationMode.NEAREST)

            i, j, h, w = transforms.RandomResizedCrop.get_params(
                image, scale=(0.8, 1.0), ratio=(0.9, 1.1))
            image = TF.crop(image, i, j, h, w)
            if mask: mask = TF.crop(mask, i, j, h, w)

            if random.random() > 0.5:
                image = TF.hflip(image)
                if mask: mask = TF.hflip(mask)

            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle)
            if mask:
                mask = TF.rotate(mask, angle,
                                  interpolation=transforms.InterpolationMode.NEAREST)

            if is_minority:
                shear = random.uniform(-10, 10)
                image = TF.affine(image, angle=0, translate=[0, 0],
                                   scale=1.0, shear=shear)
                if mask:
                    mask = TF.affine(mask, angle=0, translate=[0, 0],
                                      scale=1.0, shear=shear,
                                      interpolation=transforms.InterpolationMode.NEAREST)

            image = transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.05)(image)
            image = TF.adjust_gamma(image, random.uniform(0.7, 1.4))

            if self.cfg.get("use_exposure_augment", False):
                if random.random() < self.cfg.get("exposure_aug_prob", 0.3):
                    mode = random.choice(["over", "under"])
                    if mode == "over":
                        image = TF.adjust_brightness(image, random.uniform(1.4, 2.0))
                        image = TF.adjust_contrast(image,   random.uniform(0.6, 1.0))
                        image = TF.adjust_gamma(image,      random.uniform(0.4, 0.65))
                    else:
                        image = TF.adjust_brightness(image, random.uniform(0.3, 0.6))
                        image = TF.adjust_contrast(image,   random.uniform(0.7, 1.2))
                        image = TF.adjust_gamma(image,      random.uniform(1.5, 2.5))

            image = TF.resize(image, (sz, sz))
            if mask:
                mask = TF.resize(mask, (sz, sz),
                                  interpolation=transforms.InterpolationMode.NEAREST)
        else:
            image = TF.resize(image, (sz, sz))
            if mask:
                mask = TF.resize(mask, (sz, sz),
                                  interpolation=transforms.InterpolationMode.NEAREST)
        return image, mask


# ============================================================
# 5. ITA PROJECTOR MODULE  (CNN-specific)
# ============================================================
class ITAProjector(nn.Module):
    def __init__(self, proj_dim: int = 8, hidden: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, proj_dim),
        )
        self.missing_token = nn.Parameter(torch.zeros(1, proj_dim))

    def forward(self, ita, has_ita):
        mlp_out     = self.mlp(ita)
        missing_exp = self.missing_token.expand(ita.size(0), -1)
        return has_ita * mlp_out + (1.0 - has_ita) * missing_exp


# ============================================================
# 6. BACKBONE MODELS  (CNN-specific)
# ============================================================
class ResNet18ITA(nn.Module):
    def __init__(self, num_classes=6, dropout_rate=0.4,
                 use_ita=True, ita_proj_dim=8):
        super().__init__()
        self.use_ita  = use_ita
        self.backbone = tv_models.resnet18(
            weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()
        feat_dim = 512
        if use_ita:
            self.ita_proj = ITAProjector(proj_dim=ita_proj_dim, hidden=16)
            fused_dim     = feat_dim + ita_proj_dim
        else:
            self.ita_proj = None
            fused_dim     = feat_dim
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(fused_dim, 256), nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, ita, has_ita):
        feats = self.backbone(x)
        if self.use_ita:
            feats = torch.cat([feats, self.ita_proj(ita, has_ita)], dim=1)
        return self.classifier(feats)

    def get_gradcam_layer(self):
        return self.backbone.layer4[-1].conv2


class EfficientNetB0ITA(nn.Module):
    def __init__(self, num_classes=6, dropout_rate=0.4,
                 use_ita=True, ita_proj_dim=8):
        super().__init__()
        self.use_ita  = use_ita
        self.backbone = tv_models.efficientnet_b0(
            weights=tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        feat_dim                 = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        if use_ita:
            self.ita_proj = ITAProjector(proj_dim=ita_proj_dim, hidden=16)
            fused_dim     = feat_dim + ita_proj_dim
        else:
            self.ita_proj = None
            fused_dim     = feat_dim
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(fused_dim, 256), nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, ita, has_ita):
        feats = self.backbone(x)
        if self.use_ita:
            feats = torch.cat([feats, self.ita_proj(ita, has_ita)], dim=1)
        return self.classifier(feats)

    def get_gradcam_layer(self):
        return self.backbone.features[-1][0]


def build_model(cfg, device):
    arch = cfg["arch"].lower()
    dr   = cfg["mc_dropout_rate"] if cfg["use_mc_dropout"] else 0.0
    pdim = cfg.get("ita_proj_dim", 8)
    use  = cfg["use_ita_fusion"]
    if arch == "resnet18":
        model = ResNet18ITA(cfg["num_classes"], dr, use, pdim)
    elif arch == "efficientnet_b0":
        model = EfficientNetB0ITA(cfg["num_classes"], dr, use, pdim)
    else:
        raise ValueError(
            f"Unknown arch: '{arch}'. Supported: resnet18, efficientnet_b0")
    return model.to(device)


# ============================================================
# 7. SCHEDULER FACTORY
# ============================================================
def build_scheduler(optimizer, cfg, steps_per_epoch, scheduler_name=None):
    name = (scheduler_name or cfg.get("scheduler", "cosine")).lower()
    if name == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg["scheduler_step"],
            gamma    =cfg["scheduler_gamma"])
    elif name == "cosine":
        T_0 = cfg.get("cosine_T0_epochs", 5) * steps_per_epoch
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0    =max(1, T_0),
            T_mult =cfg.get("cosine_T_mult", 2),
            eta_min=cfg.get("cosine_eta_min", 1e-6))
    raise ValueError(f"Unknown scheduler '{name}'.")


# ============================================================
# 8. CNN LAYER FREEZING UTILITIES
# ============================================================
def freeze_cnn_layers(model: nn.Module, n_layers: int, cfg: dict) -> dict:
    frozen = {}
    arch   = cfg["arch"].lower()
    if n_layers == 0:
        print("  [Freeze] n_layers=0 → full fine-tune (nothing frozen).")
        return frozen
    if arch == "resnet18":
        layer_groups = [
            ("stem",   [model.backbone.conv1, model.backbone.bn1]),
            ("layer1", [model.backbone.layer1]),
            ("layer2", [model.backbone.layer2]),
            ("layer3", [model.backbone.layer3]),
            ("layer4", [model.backbone.layer4]),
        ]
        for gi, (gname, modules) in enumerate(layer_groups):
            if gi >= n_layers: break
            for m in modules:
                for name, param in m.named_parameters():
                    param.requires_grad = False
                    frozen[f"backbone.{gname}.{name}"] = False
    elif arch == "efficientnet_b0":
        n_blocks = len(model.backbone.features)
        for i in range(min(n_layers, n_blocks)):
            for name, param in model.backbone.features[i].named_parameters():
                param.requires_grad = False
                frozen[f"backbone.features.{i}.{name}"] = False
    n_frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    n_total  = sum(1 for p in model.parameters())
    print(f"  [Freeze] Froze {n_layers} layer group(s) "
          f"({n_frozen}/{n_total} tensors frozen).")
    return frozen


def unfreeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = True


def build_finetune_optimizer(model: nn.Module, cfg: dict) -> optim.Optimizer:
    finetune_lr = cfg.get("curriculum_finetune_lr", 5e-5)
    finetune_wd = cfg.get("curriculum_finetune_wd", 3e-4)
    ita_params   = [p for n, p in model.named_parameters()
                    if "ita_proj" in n and p.requires_grad]
    other_params = [p for n, p in model.named_parameters()
                    if "ita_proj" not in n and p.requires_grad]
    return optim.AdamW([
        {"params": other_params, "lr": finetune_lr,       "weight_decay": finetune_wd},
        {"params": ita_params,   "lr": finetune_lr * 0.1, "weight_decay": finetune_wd},
    ])


# ============================================================
# 9. DATA LOADING UTILITIES
# ============================================================
def load_fold_df(cfg, fold):
    tmpl     = _fold_csv_tmpl_from_cfg(cfg)
    csv_path = os.path.join(cfg["fold_csv_dir"], tmpl.format(fold=fold))
    df = pd.read_csv(csv_path, low_memory=False)
    df["split"] = df["split"].str.strip().str.lower()

    curriculum_rows = df["split"].isin(["pretrain_train", "pretrain_valid"]).sum()
    if curriculum_rows > 0:
        print(f"  [WARN] Fold {fold}: {curriculum_rows:,} rows with "
              f"'pretrain_train'/'pretrain_valid' split labels found "
              f"but mode is not curriculum — these rows will be ignored.")

    train_df = df[df["split"] == "train"].copy()
    val_df   = df[df["split"] == "valid"].copy()
    train_df["label"] = train_df["fitzpatrick"].astype(int) - 1
    val_df["label"]   = val_df["fitzpatrick"].astype(int) - 1
    print(f"  Fold {fold} | train={len(train_df):,}, val={len(val_df):,}")
    print(f"  Train label dist: {dict(train_df['label'].value_counts().sort_index())}")
    print(f"  Val   label dist: {dict(val_df['label'].value_counts().sort_index())}")
    return train_df, val_df


def build_dataloaders(train_df, val_df, cfg, alpha=None):
    train_df = train_df.copy()
    val_df   = val_df.copy()
    train_df["label"] = train_df["fitzpatrick"].astype(int) - 1
    val_df["label"]   = val_df["fitzpatrick"].astype(int) - 1

    train_soft = _build_soft_labels(train_df, cfg, alpha)
    val_soft   = _build_soft_labels(val_df,   cfg)

    train_prov = _get_prov_weights(train_df, cfg)
    val_prov   = _get_prov_weights(val_df,   cfg)

    ita_col          = cfg.get("ita_col", "ITA_value_spectral_spatial")
    train_ita_raw    = extract_ita_array(train_df, ita_col)
    val_ita_raw      = extract_ita_array(val_df,   ita_col)
    normalizer       = ITANormalizer()
    train_ita_normed = normalizer.fit_transform(train_ita_raw)
    val_ita_normed   = normalizer.transform(val_ita_raw)

    train_ds = SkinDatasetITA(train_df, train_soft, train_ita_normed, train_prov,
                               is_train=True,  cfg=cfg)
    val_ds   = SkinDatasetITA(val_df,   val_soft,   val_ita_normed,   val_prov,
                               is_train=False, cfg=cfg)

    train_labels = train_df["label"].values

    if cfg.get("use_balanced_batch_sampler", False):
        sampler = BalancedBatchSampler(
            train_labels, cfg["batch_size"], cfg["num_classes"])
        train_loader = DataLoader(train_ds, batch_sampler=sampler,
                                  num_workers=cfg["num_workers"], pin_memory=True)
    elif cfg.get("use_weighted_sampler", True):
        cc  = np.maximum(np.bincount(train_labels, minlength=cfg["num_classes"]), 1)
        sw  = 1.0 / cc[train_labels]
        wrs = WeightedRandomSampler(torch.tensor(sw, dtype=torch.float32),
                                    num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                                  sampler=wrs, num_workers=cfg["num_workers"],
                                  pin_memory=True, drop_last=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                                  shuffle=True, num_workers=cfg["num_workers"],
                                  pin_memory=True, drop_last=True)

    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=cfg["num_workers"], pin_memory=True)
    cw = compute_class_weights(train_labels, cfg["num_classes"])
    return train_loader, val_loader, cw, normalizer


def build_test_dataloader(test_df, cfg, normalizer: ITANormalizer):
    """
    Build a DataLoader for the held-out test set.
    Accepts an explicit ITANormalizer (fitted on the full training pool)
    so ITA values are normalised consistently with training.
    """
    test_soft = _build_soft_labels(test_df, cfg)
    test_prov = _get_prov_weights(test_df, cfg)
    ita_col   = cfg.get("ita_col", "ITA_value_spectral_spatial")
    test_ita  = normalizer.transform(extract_ita_array(test_df, ita_col))
    test_ds   = SkinDatasetITA(test_df, test_soft, test_ita, test_prov,
                                is_train=False, cfg=cfg)
    return DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False,
                      num_workers=cfg["num_workers"], pin_memory=True)


# ============================================================
# 10. CURRICULUM DATA SPLITTING UTILITIES
# ============================================================
def load_curriculum_fold_df(cfg, fold):
    tmpl     = _fold_csv_tmpl_from_cfg(cfg)
    csv_path = os.path.join(cfg["fold_csv_dir"], tmpl.format(fold=fold))
    df       = pd.read_csv(csv_path, low_memory=False)
    df["split"] = df["split"].str.strip().str.lower()

    valid_splits = {"train", "valid", "pretrain_train", "pretrain_valid"}
    unknown = set(df["split"].unique()) - valid_splits
    if unknown:
        print(f"  [WARN] Fold {fold}: unrecognised split values "
              f"{unknown} — these rows will be ignored.")

    pretrain_tr = df[df["split"] == "pretrain_train"].copy().reset_index(drop=True)
    pretrain_va = df[df["split"] == "pretrain_valid"].copy().reset_index(drop=True)
    finetune_tr = df[df["split"] == "train"].copy().reset_index(drop=True)
    finetune_va = df[df["split"] == "valid"].copy().reset_index(drop=True)

    for part_df in (pretrain_tr, pretrain_va, finetune_tr, finetune_va):
        part_df["label"] = part_df["fitzpatrick"].astype(int) - 1

    _log_curriculum_split(fold, pretrain_tr, pretrain_va,
                          finetune_tr, finetune_va,
                          cfg.get("curriculum_dataset_col", "dataset"))
    return pretrain_tr, pretrain_va, finetune_tr, finetune_va


def _log_curriculum_split(fold, pretrain_tr, pretrain_va,
                           finetune_tr, finetune_va, dataset_col="dataset"):
    print(f"\n  ── Curriculum Split (Fold {fold}) ─────────────────────")
    for stage, tr, va in [
        ("Stage 1 (pretrain_train/valid)", pretrain_tr, pretrain_va),
        ("Stage 2 (train/valid)",          finetune_tr, finetune_va),
    ]:
        src_tr = (tr[dataset_col].value_counts().to_dict()
                  if dataset_col in tr.columns and len(tr) else {})
        src_va = (va[dataset_col].value_counts().to_dict()
                  if dataset_col in va.columns and len(va) else {})
        print(f"  {stage:34s} | train={len(tr):6,}  val={len(va):6,}")
        if src_tr: print(f"    train sources : {src_tr}")
        if src_va: print(f"    val   sources : {src_va}")
        lbl_tr = dict(tr["label"].value_counts().sort_index()) if len(tr) else {}
        lbl_va = dict(va["label"].value_counts().sort_index()) if len(va) else {}
        print(f"    train labels  : {lbl_tr}")
        print(f"    val   labels  : {lbl_va}")
    print(f"  {'─'*55}")


# ============================================================
# 11. TRAINING LOOP
# ============================================================
def _run_one_epoch(model, loader, criterion, optimizer, scheduler,
                   cfg, epoch, total_epochs, train_df, device,
                   desc="Train", alpha_start=None, alpha_end=None):
    model.train()
    running_loss = correct = total = 0

    a_start = alpha_start if alpha_start is not None else cfg.get("alpha_start", 0.20)
    a_end   = alpha_end   if alpha_end   is not None else cfg.get("alpha_end",   0.10)

    if cfg.get("use_alpha_annealing", True) and cfg.get("use_soft_labels", True):
        current_alpha = anneal_alpha(epoch, total_epochs, a_start, a_end)
        loader.dataset.soft_labels = _build_soft_labels(
            train_df, cfg, alpha=current_alpha)
    else:
        current_alpha = cfg.get("soft_label_alpha", 0.15)

    use_mixup   = cfg.get("use_mixup", False)
    mixup_prob  = cfg.get("mixup_prob", 0.5)
    mixup_alpha = cfg.get("mixup_alpha", 0.4)

    for batch_idx, (imgs, soft_lbls, hard_lbls,
                    ita, has_ita, pids, _, prov_w) in enumerate(
        tqdm(loader, desc=desc, leave=False)
    ):
        imgs      = imgs.to(device)
        soft_lbls = soft_lbls.to(device)
        hard_lbls = hard_lbls.to(device)
        ita       = ita.to(device)
        has_ita   = has_ita.to(device)
        prov_w    = prov_w.to(device)

        if use_mixup and random.random() < mixup_prob:
            imgs, soft_lbls, hard_lbls, prov_w = mixup_batch(
                imgs, soft_lbls, hard_lbls, prov_w, alpha=mixup_alpha)

        optimizer.zero_grad()
        logits = model(imgs, ita, has_ita)

        prov_arg = prov_w if cfg.get("use_provenance_weighting", False) else None
        loss = criterion(logits, soft_lbls, hard_lbls, prov_weights=prov_arg)

        if cfg["use_patient_consistency"]:
            loss = loss + patient_consistency_loss(
                logits, list(pids), cfg["patient_kl_weight"])
        if cfg.get("use_contrastive", False):
            loss = loss + supervised_contrastive_loss(
                logits, list(pids), hard_lbls,
                weight=cfg["contrastive_weight"])

        loss.backward()
        optimizer.step()

        if cfg.get("scheduler", "cosine") == "cosine":
            scheduler.step(epoch + batch_idx / max(len(loader), 1))

        running_loss += loss.item() * imgs.size(0)
        correct      += (logits.argmax(dim=1) == hard_lbls).sum().item()
        total        += imgs.size(0)

    if cfg.get("scheduler", "cosine") == "step":
        scheduler.step()

    return running_loss / max(total, 1), correct / max(total, 1), current_alpha


def train_one_fold(fold, train_df, val_df, cfg, dirs, device, trial=None):
    set_seed(cfg["seed"] + fold)
    train_loader, val_loader, class_weights, normalizer = build_dataloaders(
        train_df, val_df, cfg)

    norm_path = os.path.join(dirs["models"], f"ita_normalizer_fold{fold}.json")
    with open(norm_path, "w") as f:
        json.dump(normalizer.state_dict(), f, indent=2)

    model     = build_model(cfg, device)
    criterion = FocalKLLoss(
        class_weights=class_weights if cfg["use_class_weights"] else None,
        focal_gamma=cfg["focal_gamma"], use_focal=cfg["use_focal_loss"],
        use_class_weights=cfg["use_class_weights"],
    )
    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))

    best_val_acc  = 0.0
    best_val_loss = float("inf")
    best_state    = None
    best_epoch    = 0
    patience_ctr  = 0
    train_losses, val_losses, train_accs, val_accs, lr_history = [], [], [], [], []

    ckpt_path       = os.path.join(dirs["models"], f"checkpoint_fold{fold}.pth")
    best_model_path = os.path.join(dirs["models"], f"best_fold{fold}_ep0.pth")
    start_epoch     = 0
    epochs = cfg["debug_epochs"] if cfg["debug_mode"] else cfg["num_epochs"]

    if cfg.get("resume", False) and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_acc  = ckpt["best_val_acc"]
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        best_state    = ckpt["best_model_state"]
        best_epoch    = ckpt["best_epoch"]
        patience_ctr  = ckpt["patience_ctr"]
        train_losses  = ckpt.get("train_losses", [])
        val_losses    = ckpt.get("val_losses", [])
        train_accs    = ckpt.get("train_accs", [])
        val_accs      = ckpt.get("val_accs", [])
        lr_history    = ckpt.get("lr_history", [])
        print(f"  [RESUME] Fold {fold} from epoch {start_epoch} "
              f"(best val acc: {best_val_acc:.4f})")

    for epoch in range(start_epoch, epochs):
        lr_history.append(optimizer.param_groups[0]["lr"])
        ep_loss, ep_acc, current_alpha = _run_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            cfg, epoch, epochs, train_df, device,
            desc=f"Fold {fold} | Epoch {epoch+1}/{epochs}",
            alpha_start=cfg.get("alpha_start", 0.20),
            alpha_end  =cfg.get("alpha_end",   0.10),
        )
        train_losses.append(ep_loss)
        train_accs.append(ep_acc)

        val_m = evaluate(model, val_loader, criterion, device, cfg)
        val_losses.append(val_m["loss"])
        val_accs.append(val_m["accuracy"])

        print(f"  Fold {fold} | Ep {epoch+1:3d} | α={current_alpha:.3f} | "
              f"TrLoss {ep_loss:.4f} | TrAcc {ep_acc:.4f} | "
              f"VaLoss {val_m['loss']:.4f} | VaAcc {val_m['accuracy']:.4f}")

        if val_m["accuracy"] > best_val_acc + cfg["min_delta"]:
            if val_m["loss"] <= 1.05 * best_val_loss:
                print(f"  Validation accuracy improved to {val_m['accuracy']:.4f} "
                      f"with acceptable loss ({val_m['loss']:.4f}).")
                best_val_acc  = val_m["accuracy"]
                best_val_loss = val_m["loss"]
                best_epoch    = epoch + 1
                best_state    = copy.deepcopy(model.state_dict())
                patience_ctr  = 0
                for old in Path(dirs["models"]).glob(f"best_fold{fold}_ep*.pth"):
                    old.unlink()
                best_model_path = os.path.join(
                    dirs["models"], f"best_fold{fold}_ep{best_epoch}.pth")
                torch.save(best_state, best_model_path)
            else:
                print(f"  Accuracy improved to {val_m['accuracy']:.4f} but loss "
                      f"({val_m['loss']:.4f}) exceeds 1.05×best. Checkpoint skipped.")
                patience_ctr += 1
                if patience_ctr >= cfg["patience"]:
                    print(f"  Early stopping at epoch {epoch+1}.")
                    break
        else:
            patience_ctr += 1

        if patience_ctr >= cfg["patience"]:
            print(f"  Early stopping at epoch {epoch+1}.")
            break

        if trial is not None:
            trial.report(val_m["accuracy"], epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        torch.save({
            "epoch":            epoch,
            "model_state":      model.state_dict(),
            "optimizer_state":  optimizer.state_dict(),
            "scheduler_state":  scheduler.state_dict(),
            "best_val_acc":     best_val_acc,
            "best_val_loss":    best_val_loss,
            "best_model_state": best_state,
            "best_epoch":       best_epoch,
            "patience_ctr":     patience_ctr,
            "train_losses":     train_losses,
            "val_losses":       val_losses,
            "train_accs":       train_accs,
            "val_accs":         val_accs,
            "lr_history":       lr_history,
            "best_model_path":  best_model_path,
        }, ckpt_path)

    if best_state is not None:
        model.load_state_dict(best_state)

    final_metrics = evaluate(model, val_loader, criterion, device, cfg)
    final_metrics["best_epoch"] = best_epoch
    final_metrics["fold"]       = fold

    _save_curves(None, None, None, None, None, dirs["plots"],
                 fold=fold, train_losses=train_losses, val_losses=val_losses,
                 train_accs=train_accs, val_accs=val_accs, lr_history=lr_history)

    return model, final_metrics, normalizer, {
        "train_losses": train_losses, "val_losses": val_losses,
        "train_accs":   train_accs,   "val_accs":   val_accs,
        "lr_history":   lr_history,
    }


# ============================================================
# 12. CURRICULUM TRAINING LOOP — ONE FOLD
# ============================================================
def train_curriculum_one_fold(fold, cfg, stage_dirs, device):
    set_seed(cfg["seed"] + fold)

    pretrain_tr, pretrain_va, finetune_tr, finetune_va = \
        load_curriculum_fold_df(cfg, fold)

    if len(pretrain_tr) == 0:
        raise RuntimeError(
            f"Fold {fold}: 'pretrain_train' split is empty.")
    if len(finetune_tr) == 0:
        raise RuntimeError(
            f"Fold {fold}: 'train' split is empty.")

    val_for_stage2 = finetune_va if len(finetune_va) > 0 else pretrain_va.copy()
    if len(finetune_va) == 0:
        print(f"  [WARN] Fold {fold}: 'valid' split is empty — "
              f"using pretrain_valid rows for Stage 2 validation.")

    # ── STAGE 1 ───────────────────────────────────────────────
    print(f"\n  {'─'*55}\n  STAGE 1 — Pre-train  (Fold {fold})\n  {'─'*55}")

    pretrain_epochs = (
        cfg.get("debug_curriculum_pretrain_epochs", 2)
        if cfg.get("debug_mode", False)
        else cfg.get("num_epochs", 50)
    )

    s1_train_loader, s1_val_loader, s1_cw, s1_normalizer = build_dataloaders(
        pretrain_tr, pretrain_va, cfg)
    with open(os.path.join(stage_dirs["pretrain"]["models"],
                           f"ita_normalizer_s1_fold{fold}.json"), "w") as f:
        json.dump(s1_normalizer.state_dict(), f, indent=2)

    model = build_model(cfg, device)
    s1_criterion = FocalKLLoss(
        class_weights=s1_cw if cfg["use_class_weights"] else None,
        focal_gamma=cfg["focal_gamma"], use_focal=cfg["use_focal_loss"],
        use_class_weights=cfg["use_class_weights"])
    s1_optimizer = optim.AdamW(model.parameters(),
                               lr=cfg["learning_rate"],
                               weight_decay=cfg["weight_decay"])
    s1_scheduler = build_scheduler(s1_optimizer, cfg, len(s1_train_loader))

    s1_tl, s1_vl, s1_ta, s1_va_list, s1_lr = [], [], [], [], []
    s1_best_val_acc  = 0.0
    s1_best_val_loss = float("inf")
    s1_best_state    = None
    s1_best_epoch    = 0
    s1_patience_ctr  = 0
    s1_ckpt = os.path.join(stage_dirs["pretrain"]["models"],
                           f"checkpoint_fold{fold}.pth")

    for epoch in range(pretrain_epochs):
        s1_lr.append(s1_optimizer.param_groups[0]["lr"])
        ep_loss, ep_acc, cur_alpha = _run_one_epoch(
            model, s1_train_loader, s1_criterion, s1_optimizer, s1_scheduler,
            cfg, epoch, pretrain_epochs, pretrain_tr, device,
            desc=f"S1 Fold {fold} | Ep {epoch+1}/{pretrain_epochs} [pretrain]",
            alpha_start=cfg.get("alpha_start", 0.20),
            alpha_end  =cfg.get("alpha_end",   0.10))
        s1_tl.append(ep_loss); s1_ta.append(ep_acc)
        val_m = evaluate(model, s1_val_loader, s1_criterion, device, cfg)
        s1_vl.append(val_m["loss"]); s1_va_list.append(val_m["accuracy"])
        print(f"  S1 Fold {fold} | Ep {epoch+1:3d} | α={cur_alpha:.3f} | "
              f"TrLoss {ep_loss:.4f} | TrAcc {ep_acc:.4f} | "
              f"VaLoss {val_m['loss']:.4f} | VaAcc {val_m['accuracy']:.4f}")

        if val_m["accuracy"] > s1_best_val_acc + cfg["min_delta"]:
            if val_m["loss"] <= 1.05 * s1_best_val_loss:
                s1_best_val_acc  = val_m["accuracy"]
                s1_best_val_loss = val_m["loss"]
                s1_best_epoch    = epoch + 1
                s1_best_state    = copy.deepcopy(model.state_dict())
                s1_patience_ctr  = 0
                for old in Path(stage_dirs["pretrain"]["models"]).glob(
                    f"best_fold{fold}_ep*.pth"): old.unlink()
                torch.save(s1_best_state, os.path.join(
                    stage_dirs["pretrain"]["models"],
                    f"best_fold{fold}_ep{s1_best_epoch}.pth"))
            else:
                s1_patience_ctr += 1
        else:
            s1_patience_ctr += 1

        if s1_patience_ctr >= cfg["patience"]:
            print(f"  [S1] Early stopping at epoch {epoch+1}."); break

        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "best_model_state": s1_best_state,
                    "best_val_acc": s1_best_val_acc,
                    "best_val_loss": s1_best_val_loss,
                    "best_epoch": s1_best_epoch,
                    "patience_ctr": s1_patience_ctr}, s1_ckpt)

    if s1_best_state is not None:
        model.load_state_dict(s1_best_state)

    pretrain_metrics = evaluate(model, s1_val_loader, s1_criterion, device, cfg)
    pretrain_metrics["best_epoch"] = s1_best_epoch
    pretrain_metrics["fold"]       = fold
    pretrain_metrics["stage"]      = 1

    _save_curves(None, None, None, None, None, stage_dirs["pretrain"]["plots"],
                 fold=fold, train_losses=s1_tl, val_losses=s1_vl,
                 train_accs=s1_ta, val_accs=s1_va_list, lr_history=s1_lr)
    print(f"\n  Stage 1 Complete | Fold {fold} | Best Ep {s1_best_epoch} | "
          f"Val Acc {pretrain_metrics['accuracy']:.4f}")

    # ── STAGE 2 ───────────────────────────────────────────────
    print(f"\n  {'─'*55}\n  STAGE 2 — Fine-tune  (Fold {fold})\n  {'─'*55}")

    finetune_epochs = (
        cfg.get("debug_curriculum_finetune_epochs", 2)
        if cfg.get("debug_mode", False)
        else cfg.get("curriculum_finetune_epochs", 20)
    )
    finetune_alpha = cfg.get("curriculum_finetune_alpha", 0.08)

    s2_train_loader, s2_val_loader, s2_cw, s2_normalizer = build_dataloaders(
        finetune_tr, val_for_stage2, cfg, alpha=finetune_alpha)
    with open(os.path.join(stage_dirs["finetune"]["models"],
                           f"ita_normalizer_s2_fold{fold}.json"), "w") as f:
        json.dump(s2_normalizer.state_dict(), f, indent=2)

    n_freeze = cfg.get("curriculum_freeze_layers", 2)
    freeze_cnn_layers(model, n_freeze, cfg)

    s2_optimizer = build_finetune_optimizer(model, cfg)
    s2_scheduler = build_scheduler(
        s2_optimizer, cfg, len(s2_train_loader),
        scheduler_name=cfg.get("curriculum_finetune_scheduler", "cosine"))
    s2_criterion = FocalKLLoss(
        class_weights=s2_cw if cfg["use_class_weights"] else None,
        focal_gamma=cfg["focal_gamma"], use_focal=cfg["use_focal_loss"],
        use_class_weights=cfg["use_class_weights"])

    s2_tl, s2_vl, s2_ta, s2_va_list, s2_lr = [], [], [], [], []
    s2_best_val_acc  = 0.0
    s2_best_val_loss = float("inf")
    s2_best_state    = None
    s2_best_epoch    = 0
    s2_patience_ctr  = 0
    s2_patience      = cfg.get("curriculum_finetune_patience", 7)
    s2_ckpt = os.path.join(stage_dirs["finetune"]["models"],
                           f"checkpoint_fold{fold}.pth")

    for epoch in range(finetune_epochs):
        s2_lr.append(s2_optimizer.param_groups[0]["lr"])
        ep_loss, ep_acc, cur_alpha = _run_one_epoch(
            model, s2_train_loader, s2_criterion, s2_optimizer, s2_scheduler,
            cfg, epoch, finetune_epochs, finetune_tr, device,
            desc=f"S2 Fold {fold} | Ep {epoch+1}/{finetune_epochs} [finetune]",
            alpha_start=cfg.get("curriculum_finetune_alpha_start", 0.12),
            alpha_end  =cfg.get("curriculum_finetune_alpha_end",   0.05))
        s2_tl.append(ep_loss); s2_ta.append(ep_acc)
        val_m = evaluate(model, s2_val_loader, s2_criterion, device, cfg)
        s2_vl.append(val_m["loss"]); s2_va_list.append(val_m["accuracy"])
        print(f"  S2 Fold {fold} | Ep {epoch+1:3d} | α={cur_alpha:.3f} | "
              f"TrLoss {ep_loss:.4f} | TrAcc {ep_acc:.4f} | "
              f"VaLoss {val_m['loss']:.4f} | VaAcc {val_m['accuracy']:.4f}")

        if val_m["accuracy"] > s2_best_val_acc + cfg["min_delta"]:
            if val_m["loss"] <= 1.05 * s2_best_val_loss:
                s2_best_val_acc  = val_m["accuracy"]
                s2_best_val_loss = val_m["loss"]
                s2_best_epoch    = epoch + 1
                s2_best_state    = copy.deepcopy(model.state_dict())
                s2_patience_ctr  = 0
                for old in Path(stage_dirs["finetune"]["models"]).glob(
                    f"best_fold{fold}_ep*.pth"): old.unlink()
                torch.save(s2_best_state, os.path.join(
                    stage_dirs["finetune"]["models"],
                    f"best_fold{fold}_ep{s2_best_epoch}.pth"))
            else:
                s2_patience_ctr += 1
        else:
            s2_patience_ctr += 1

        if s2_patience_ctr >= s2_patience:
            print(f"  [S2] Early stopping at epoch {epoch+1}."); break

        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "best_model_state": s2_best_state,
                    "best_val_acc": s2_best_val_acc,
                    "best_val_loss": s2_best_val_loss,
                    "best_epoch": s2_best_epoch,
                    "patience_ctr": s2_patience_ctr}, s2_ckpt)

    if s2_best_state is not None:
        model.load_state_dict(s2_best_state)
    unfreeze_all(model)

    finetune_metrics = evaluate(model, s2_val_loader, s2_criterion, device, cfg)
    finetune_metrics["best_epoch"] = s2_best_epoch
    finetune_metrics["fold"]       = fold
    finetune_metrics["stage"]      = 2

    _save_curves(None, None, None, None, None, stage_dirs["finetune"]["plots"],
                 fold=fold, train_losses=s2_tl, val_losses=s2_vl,
                 train_accs=s2_ta, val_accs=s2_va_list, lr_history=s2_lr)
    print(f"\n  Stage 2 Complete | Fold {fold} | Best Ep {s2_best_epoch} | "
          f"Val Acc {finetune_metrics['accuracy']:.4f}")

    curves = {
        "pretrain": {
            "train_losses": s1_tl, "val_losses": s1_vl,
            "train_accs":   s1_ta, "val_accs":   s1_va_list,
            "lr_history":   s1_lr,
            "n_train":      len(pretrain_tr), "n_val": len(pretrain_va),
        },
        "finetune": {
            "train_losses": s2_tl, "val_losses": s2_vl,
            "train_accs":   s2_ta, "val_accs":   s2_va_list,
            "lr_history":   s2_lr,
            "n_train":      len(finetune_tr), "n_val": len(finetune_va),
        },
        "val_for_stage2": val_for_stage2,
    }
    return model, pretrain_metrics, finetune_metrics, curves, s1_normalizer, s2_normalizer


# ============================================================
# 13. EVALUATION
# ============================================================
def evaluate(model, dataloader, criterion, device, cfg):
    model.eval()
    total_loss = 0.0
    all_hard, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            # Support both 7-element (old) and 8-element (new, with prov_weight) batches
            if len(batch) == 8:
                imgs, soft_lbls, hard_lbls, ita, has_ita, pids, idxs, prov_w = batch
            else:
                imgs, soft_lbls, hard_lbls, ita, has_ita, pids, idxs = batch
                prov_w = torch.ones(imgs.size(0))
            imgs      = imgs.to(device)
            soft_lbls = soft_lbls.to(device)
            hard_lbls = hard_lbls.to(device)
            ita       = ita.to(device)
            has_ita   = has_ita.to(device)
            logits    = model(imgs, ita, has_ita)
            loss      = criterion(logits, soft_lbls, hard_lbls)
            total_loss += loss.item() * imgs.size(0)
            probs = F.softmax(logits, dim=1)
            all_hard.extend(hard_lbls.cpu().numpy())
            all_preds.extend(probs.argmax(dim=1).cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    from sklearn.metrics import accuracy_score
    all_hard  = np.array(all_hard)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    return {
        "loss":     total_loss / max(len(all_hard), 1),
        "accuracy": accuracy_score(all_hard, all_preds),
        "labels":   all_hard, "preds": all_preds, "probs": all_probs,
    }


# ============================================================
# 14. TTA  (CNN-specific: uses ITANormalizer)
# ============================================================
def build_tta_transforms(cfg):
    sz        = cfg["image_size"]
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std =[0.229, 0.224, 0.225])
    def _t(img): return normalize(TF.to_tensor(img))
    return [
        lambda img: _t(TF.resize(img, (sz, sz))),
        lambda img: _t(TF.hflip(TF.resize(img, (sz, sz)))),
        lambda img: _t(TF.adjust_brightness(TF.resize(img, (sz, sz)), 1.2)),
        lambda img: _t(TF.adjust_brightness(TF.resize(img, (sz, sz)), 0.8)),
        lambda img: _t(TF.adjust_contrast(TF.resize(img, (sz, sz)), 1.3)),
        lambda img: _t(TF.adjust_gamma(TF.resize(img, (sz, sz)), 0.85)),
        lambda img: _t(TF.adjust_gamma(TF.resize(img, (sz, sz)), 1.15)),
        lambda img: _t(TF.rotate(TF.resize(img, (sz, sz)),  5)),
        lambda img: _t(TF.rotate(TF.resize(img, (sz, sz)), -5)),
    ]


def tta_predict(model, df, cfg, device, normalizer: ITANormalizer,
                mask_col="mask_path_512"):
    model.eval()
    tta_transforms = build_tta_transforms(cfg)
    sz      = cfg["image_size"]
    ita_col = cfg.get("ita_col", "ITA_value_spectral_spatial")
    all_mean_probs, all_labels = [], []
    print(f"  Running TTA ({len(tta_transforms)} views) over {len(df)} samples...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="TTA", leave=False):
        img_path  = str(row["image_path_512"]).strip()
        mask_path = row.get(mask_col, None)
        image     = Image.open(img_path).convert("RGB")
        has_mask  = (mask_path is not None
                     and str(mask_path).strip() not in ("", "nan", "None")
                     and os.path.exists(str(mask_path).strip()))
        if has_mask:
            mask = TF.resize(Image.open(str(mask_path).strip()).convert("L"),
                              (sz, sz),
                              interpolation=transforms.InterpolationMode.NEAREST)
            mask_tensor = TF.to_tensor(mask)
        else:
            mask_tensor = torch.ones(1, sz, sz)

        ita_raw      = row[ita_col] if ita_col in df.columns else None
        has_real_ita = (ita_raw is not None
                        and not (isinstance(ita_raw, float)
                                 and np.isnan(float(ita_raw))))
        ita_normed_val = (
            float(normalizer.transform(
                np.array([float(ita_raw)], dtype=np.float32))[0])
            if has_real_ita else 0.0
        )
        ita_t     = torch.tensor([[ita_normed_val]], dtype=torch.float32).to(device)
        has_ita_t = torch.tensor([[1.0 if has_real_ita else 0.0]],
                                  dtype=torch.float32).to(device)
        view_probs = []
        for tfm in tta_transforms:
            img_t = (tfm(image) * mask_tensor).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = F.softmax(model(img_t, ita_t, has_ita_t), dim=1)
            view_probs.append(probs.cpu().numpy())
        all_mean_probs.append(np.stack(view_probs).mean(axis=0).squeeze())
        all_labels.append(int(row["label"]))
    mean_probs = np.array(all_mean_probs)
    return {"mean_probs": mean_probs, "preds": mean_probs.argmax(axis=1),
            "labels": np.array(all_labels)}


# ============================================================
# 15. MAIN CROSS-VALIDATION LOOP
# ============================================================
def run_cv(cfg, dirs, device):
    class_names      = CLASS_NAMES
    all_fold_metrics = []
    all_patient_dfs  = []
    timing_records   = []
    best_epochs      = []
    fold_normalizers = {}

    all_train_losses, all_val_losses = [], []
    all_train_accs,   all_val_accs   = [], []
    all_lr_histories                 = []

    quick = cfg.get("quick_mode", False)
    folds_to_run = ([cfg.get("quick_fold", 1)] if quick
                    else list(range(1, cfg["num_folds"] + 1)))
    if quick:
        print(f"\n  [QUICK] Running single fold {folds_to_run[0]} only. "
              f"TTA / final-retrain suppressed.")

    # ImageNet stats for CNN Grad-CAM denormalisation
    _mean_arr = np.array([0.485, 0.456, 0.406])
    _std_arr  = np.array([0.229, 0.224, 0.225])

    for fold in folds_to_run:
        print(f"\n{'='*60}")
        print(f"  FOLD {fold} / {cfg['num_folds']}"
              + ("  [QUICK]" if quick else ""))
        print(f"{'='*60}")

        t_fold_start = time.time()
        train_df, val_df = load_fold_df(cfg, fold)

        if cfg["debug_mode"]:
            n_per = cfg["debug_n_samples"] // cfg["num_classes"]
            train_df = pd.concat([
                train_df[train_df["fitzpatrick"] == fst].sample(
                    min(len(train_df[train_df["fitzpatrick"] == fst]), n_per),
                    random_state=cfg["seed"])
                for fst in range(1, cfg["num_classes"]+1)
                if len(train_df[train_df["fitzpatrick"] == fst]) > 0
            ], ignore_index=True)
            val_df = pd.concat([
                val_df[val_df["fitzpatrick"] == fst].sample(
                    min(len(val_df[val_df["fitzpatrick"] == fst]), n_per),
                    random_state=cfg["seed"])
                for fst in range(1, cfg["num_classes"]+1)
                if len(val_df[val_df["fitzpatrick"] == fst]) > 0
            ], ignore_index=True)
            print(f"  [DEBUG] train={len(train_df)}, val={len(val_df)}")

        train_df["label"] = train_df["fitzpatrick"].astype(int) - 1
        val_df["label"]   = val_df["fitzpatrick"].astype(int) - 1

        t_train_start = time.time()
        model, fold_metrics, normalizer, curves = train_one_fold(
            fold, train_df, val_df, cfg, dirs, device)
        train_time = time.time() - t_train_start

        fold_normalizers[fold] = normalizer
        all_train_losses.append(curves["train_losses"])
        all_val_losses.append(curves["val_losses"])
        all_train_accs.append(curves["train_accs"])
        all_val_accs.append(curves["val_accs"])
        all_lr_histories.append(curves["lr_history"])
        best_epochs.append(fold_metrics["best_epoch"])

        ext_m = compute_extended_metrics(
            fold_metrics["labels"], fold_metrics["preds"],
            fold_metrics["probs"], cfg["num_classes"], class_names)
        fold_metrics.update(ext_m)
        all_fold_metrics.append(fold_metrics)

        tag = f"fold{fold}"
        plot_confusion_matrix(fold_metrics["labels"], fold_metrics["preds"],
                              class_names, tag, dirs["plots"])
        plot_roc_curve(fold_metrics["labels"], fold_metrics["probs"],
                       class_names, tag, dirs["plots"])
        plot_pr_curve(fold_metrics["labels"], fold_metrics["probs"],
                      class_names, tag, dirs["plots"])
        plot_calibration_curve(fold_metrics["labels"], fold_metrics["probs"],
                               class_names, tag, dirs["plots"])

        inf_time  = 0.0
        mc_passes = (cfg.get("quick_mc_dropout_passes", 5) if quick
                     else cfg["mc_dropout_passes"])
        if cfg["use_mc_dropout"]:
            _, val_loader, _, _ = build_dataloaders(train_df, val_df, cfg)
            t_inf = time.time()
            unc = mc_dropout_predict(model, val_loader, device, mc_passes)
            inf_time = time.time() - t_inf
            unc_df = pd.DataFrame({
                "predictive_entropy": unc["predictive_entropy"],
                "mutual_information": unc["mutual_information"],
                "true_label":         fold_metrics["labels"],
                "pred_label":         fold_metrics["preds"],
            })
            mc_ece_r = mc_dropout_ece(unc["mean_probs"],
                                      fold_metrics["labels"], n_bins=15)
            unc_df["mc_ece_top_label"] = mc_ece_r["mc_ece_top_label"]
            unc_df["mc_ece_classwise"] = mc_ece_r["mc_ece_classwise"]
            print(f"  MC-Dropout ECE | "
                  f"top-label={mc_ece_r['mc_ece_top_label']:.4f}  "
                  f"classwise={mc_ece_r['mc_ece_classwise']:.4f}")
            unc_df.to_csv(
                os.path.join(dirs["uncertainty"], f"uncertainty_{tag}.csv"),
                index=False)
            plot_uncertainty_boxplot(unc_df, class_names, tag,
                                     dirs["plots"], mc_passes)

        val_soft_tmp = _build_soft_labels(val_df, cfg)
        val_prov_tmp = _get_prov_weights(val_df, cfg)
        val_ita_tmp  = normalizer.transform(
            extract_ita_array(val_df, cfg.get("ita_col",
                                              "ITA_value_spectral_spatial")))
        val_ds_tmp = SkinDatasetITA(val_df, val_soft_tmp, val_ita_tmp,
                                     val_prov_tmp, is_train=False, cfg=cfg)
        pat_df = compute_patient_metrics(
            fold_metrics["labels"], fold_metrics["preds"],
            fold_metrics["probs"], val_ds_tmp.patient_ids, cfg["num_classes"])
        all_patient_dfs.append(pat_df)
        plot_patient_scatter(pat_df, dirs["plots"], tag)

        _, val_loader, _, _ = build_dataloaders(train_df, val_df, cfg)
        generate_and_save_gradcam(
            model, val_loader, class_names, dirs["gradcam"],
            device, cfg, prefix=tag,
            pred_labels=fold_metrics["preds"],
            true_labels=fold_metrics["labels"],
            mean_arr=_mean_arr, std_arr=_std_arr)

        fold_time = time.time() - t_fold_start
        timing_records.append({
            "fold":              fold,
            "train_time_s":      round(train_time, 2),
            "inference_time_s":  round(inf_time, 2),
            "total_fold_time_s": round(fold_time, 2),
            "n_samples":         len(train_df) + len(val_df),
        })
        print(f"\n  Fold {fold} Summary | "
              f"Acc={fold_metrics['accuracy']:.4f} | "
              f"F1={fold_metrics['f1']:.4f} | "
              f"±1={fold_metrics['concordance_pm1']:.4f} | "
              f"AUROC={fold_metrics['auroc']:.4f} | "
              f"QWK={fold_metrics['qwk']:.4f}")

    _save_curves(all_train_losses, all_val_losses, all_train_accs,
                 all_val_accs, all_lr_histories, dirs["plots"])

    print(f"\n{'='*60}")
    print("  CROSS-VALIDATION SUMMARY"
          + ("  [QUICK — single fold]" if quick else ""))
    print(f"{'='*60}")
    for k in AGG_KEYS:
        vals = [m[k] for m in all_fold_metrics
                if not np.isnan(m.get(k, np.nan))]
        if vals:
            print(f"  {k:25s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    if not quick:
        print("\n  Bootstrap 95% CIs (1000 resamples):")
        boot_df = bootstrap_ci(all_fold_metrics, AGG_KEYS,
                                n_bootstrap=1000, ci_level=0.95,
                                seed=cfg["seed"])
        for _, row in boot_df.iterrows():
            print(f"  {row['metric']:25s}: {row['mean']:.4f}  "
                  f"95% CI [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]")
        boot_df.to_csv(
            os.path.join(dirs["metrics"], "cv_bootstrap_ci.csv"), index=False)

    plot_metric_bars(all_fold_metrics, class_names, dirs["plots"])
    if all_patient_dfs:
        combined_pat = pd.concat(all_patient_dfs, ignore_index=True)
        plot_patient_scatter(combined_pat, dirs["plots"], "all_folds")
    save_timing(timing_records, dirs["metrics"])

    serialisable = []
    for m in all_fold_metrics:
        row = {}
        for k in AGG_KEYS + ["specificity", "fold", "best_epoch"]:
            if k in m: row[k] = m[k]
        for cls, cv in m["per_class"].items():
            for metric in PC_KEYS:
                row[f"{cls}_{metric}"] = cv.get(metric, np.nan)
        serialisable.append(row)
    pd.DataFrame(serialisable).to_csv(
        os.path.join(dirs["metrics"], "per_fold_metrics.csv"), index=False)

    return all_fold_metrics, best_epochs, timing_records, fold_normalizers


# ============================================================
# 16. CURRICULUM CROSS-VALIDATION LOOP
# ============================================================
def run_curriculum_cv(cfg, dirs, device):
    class_names = CLASS_NAMES
    pretrain_fold_metrics = []; finetune_fold_metrics = []
    all_patient_dfs = []; timing_records = []; best_s2_epochs = []
    s2_fold_normalizers = {}
    all_pretrain_curves = []; all_finetune_curves = []
    s1_all_tl, s1_all_vl, s1_all_ta, s1_all_va = [], [], [], []
    s2_all_tl, s2_all_vl, s2_all_ta, s2_all_va = [], [], [], []

    pretrain_dirs = {k: str(Path(dirs["curriculum_pretrain"]) / k)
                     for k in ("models", "plots", "metrics", "gradcam", "uncertainty")}
    finetune_dirs = {k: str(Path(dirs["curriculum_finetune"]) / k)
                     for k in ("models", "plots", "metrics", "gradcam", "uncertainty")}
    stage_dirs = {"pretrain": pretrain_dirs, "finetune": finetune_dirs}

    quick = cfg.get("quick_mode", False)
    folds_to_run = ([cfg.get("quick_fold", 1)] if quick
                    else list(range(1, cfg["num_folds"] + 1)))
    if quick:
        print(f"\n  [QUICK] Curriculum: single fold {folds_to_run[0]} only.")

    _mean_arr = np.array([0.485, 0.456, 0.406])
    _std_arr  = np.array([0.229, 0.224, 0.225])

    for fold in folds_to_run:
        print(f"\n{'='*60}\n  CURRICULUM FOLD {fold} / {cfg['num_folds']}"
              + ("  [QUICK]" if quick else "") + f"\n{'='*60}")

        t_fold_start = time.time()
        t_train      = time.time()

        (model, s1_metrics, s2_metrics, curves,
         s1_normalizer, s2_normalizer) = train_curriculum_one_fold(
            fold, cfg, stage_dirs, device)
        train_time = time.time() - t_train

        val_for_stage2 = curves.pop("val_for_stage2")
        s2_fold_normalizers[fold] = s2_normalizer
        all_pretrain_curves.append(curves["pretrain"])
        all_finetune_curves.append(curves["finetune"])
        s1_all_tl.append(curves["pretrain"]["train_losses"])
        s1_all_vl.append(curves["pretrain"]["val_losses"])
        s1_all_ta.append(curves["pretrain"]["train_accs"])
        s1_all_va.append(curves["pretrain"]["val_accs"])
        s2_all_tl.append(curves["finetune"]["train_losses"])
        s2_all_vl.append(curves["finetune"]["val_losses"])
        s2_all_ta.append(curves["finetune"]["train_accs"])
        s2_all_va.append(curves["finetune"]["val_accs"])
        best_s2_epochs.append(s2_metrics["best_epoch"])

        for stage_m, stage_dirs_m, stage_tag, fold_list in [
            (s1_metrics, pretrain_dirs, f"s1_fold{fold}", pretrain_fold_metrics),
            (s2_metrics, finetune_dirs, f"s2_fold{fold}", finetune_fold_metrics),
        ]:
            ext_m = compute_extended_metrics(
                stage_m["labels"], stage_m["preds"],
                stage_m["probs"], cfg["num_classes"], class_names)
            stage_m.update(ext_m); fold_list.append(stage_m)
            plot_confusion_matrix(stage_m["labels"], stage_m["preds"],
                                  class_names, stage_tag, stage_dirs_m["plots"])
            plot_roc_curve(stage_m["labels"], stage_m["probs"],
                           class_names, stage_tag, stage_dirs_m["plots"])
            plot_pr_curve(stage_m["labels"], stage_m["probs"],
                          class_names, stage_tag, stage_dirs_m["plots"])
            plot_calibration_curve(stage_m["labels"], stage_m["probs"],
                                   class_names, stage_tag, stage_dirs_m["plots"])

        inf_time  = 0.0
        mc_passes = (cfg.get("quick_mc_dropout_passes", 5) if quick
                     else cfg["mc_dropout_passes"])
        if cfg["use_mc_dropout"]:
            unc_soft = _build_soft_labels(
                val_for_stage2, cfg,
                alpha=cfg.get("curriculum_finetune_alpha", 0.08))
            unc_prov = _get_prov_weights(val_for_stage2, cfg)
            unc_ita  = s2_normalizer.transform(
                extract_ita_array(val_for_stage2,
                                  cfg.get("ita_col", "ITA_value_spectral_spatial")))
            unc_ds   = SkinDatasetITA(val_for_stage2, unc_soft, unc_ita, unc_prov,
                                      is_train=False, cfg=cfg)
            unc_loader = DataLoader(unc_ds, batch_size=cfg["batch_size"],
                                    shuffle=False, num_workers=cfg["num_workers"],
                                    pin_memory=True)
            t_inf = time.time()
            unc = mc_dropout_predict(model, unc_loader, device, mc_passes)
            inf_time = time.time() - t_inf
            unc_df = pd.DataFrame({
                "predictive_entropy": unc["predictive_entropy"],
                "mutual_information": unc["mutual_information"],
                "true_label":         s2_metrics["labels"],
                "pred_label":         s2_metrics["preds"],
            })
            mc_ece_r = mc_dropout_ece(unc["mean_probs"],
                                      s2_metrics["labels"], n_bins=15)
            unc_df["mc_ece_top_label"] = mc_ece_r["mc_ece_top_label"]
            unc_df["mc_ece_classwise"] = mc_ece_r["mc_ece_classwise"]
            print(f"  MC-Dropout ECE | "
                  f"top-label={mc_ece_r['mc_ece_top_label']:.4f}  "
                  f"classwise={mc_ece_r['mc_ece_classwise']:.4f}")
            unc_df.to_csv(
                os.path.join(finetune_dirs["uncertainty"],
                             f"uncertainty_s2_fold{fold}.csv"), index=False)
            plot_uncertainty_boxplot(unc_df, class_names, f"s2_fold{fold}",
                                     finetune_dirs["plots"], mc_passes)

        pm_soft = _build_soft_labels(
            val_for_stage2, cfg,
            alpha=cfg.get("curriculum_finetune_alpha", 0.08))
        pm_prov = _get_prov_weights(val_for_stage2, cfg)
        pm_ita  = s2_normalizer.transform(
            extract_ita_array(val_for_stage2,
                              cfg.get("ita_col", "ITA_value_spectral_spatial")))
        val_ds_tmp = SkinDatasetITA(val_for_stage2, pm_soft, pm_ita, pm_prov,
                                     is_train=False, cfg=cfg)
        pat_df = compute_patient_metrics(
            s2_metrics["labels"], s2_metrics["preds"],
            s2_metrics["probs"], val_ds_tmp.patient_ids, cfg["num_classes"])
        all_patient_dfs.append(pat_df)
        plot_patient_scatter(pat_df, finetune_dirs["plots"], f"s2_fold{fold}")

        exp_soft = _build_soft_labels(
            val_for_stage2, cfg,
            alpha=cfg.get("curriculum_finetune_alpha", 0.08))
        exp_prov = _get_prov_weights(val_for_stage2, cfg)
        exp_ita  = s2_normalizer.transform(
            extract_ita_array(val_for_stage2,
                              cfg.get("ita_col", "ITA_value_spectral_spatial")))
        exp_ds   = SkinDatasetITA(val_for_stage2, exp_soft, exp_ita, exp_prov,
                                   is_train=False, cfg=cfg)
        exp_loader = DataLoader(exp_ds, batch_size=cfg["batch_size"],
                                shuffle=False, num_workers=cfg["num_workers"],
                                pin_memory=True)
        generate_and_save_gradcam(
            model, exp_loader, class_names,
            finetune_dirs["gradcam"], device, cfg,
            prefix=f"s2_fold{fold}",
            pred_labels=s2_metrics["preds"],
            true_labels=s2_metrics["labels"],
            mean_arr=_mean_arr, std_arr=_std_arr)

        n_samples = (
            curves["pretrain"]["n_train"] + curves["pretrain"]["n_val"]
            + curves["finetune"]["n_train"] + curves["finetune"]["n_val"]
        )
        fold_time = time.time() - t_fold_start
        timing_records.append({
            "fold":              fold,
            "train_time_s":      round(train_time, 2),
            "inference_time_s":  round(inf_time, 2),
            "total_fold_time_s": round(fold_time, 2),
            "n_samples":         n_samples,
        })
        print(f"\n  Curriculum Fold {fold} Summary\n"
              f"    Stage 1 (pretrain) | "
              f"Acc={s1_metrics['accuracy']:.4f} | F1={s1_metrics['f1']:.4f}\n"
              f"    Stage 2 (finetune) | "
              f"Acc={s2_metrics['accuracy']:.4f} | "
              f"F1={s2_metrics['f1']:.4f} | AUROC={s2_metrics['auroc']:.4f}")

    if s1_all_tl:
        _save_curves(s1_all_tl, s1_all_vl, s1_all_ta, s1_all_va,
                     [c["lr_history"] for c in all_pretrain_curves],
                     pretrain_dirs["plots"])
    if s2_all_tl:
        _save_curves(s2_all_tl, s2_all_vl, s2_all_ta, s2_all_va,
                     [c["lr_history"] for c in all_finetune_curves],
                     finetune_dirs["plots"])

    plot_curriculum_stage_comparison(pretrain_fold_metrics, finetune_fold_metrics,
                                      dirs["plots"])
    plot_curriculum_learning_curves(all_pretrain_curves, all_finetune_curves,
                                     dirs["plots"], len(folds_to_run))

    print(f"\n{'='*60}")
    print("  CURRICULUM CV SUMMARY — Stage 2 (Clinical Fine-tune)"
          + ("  [QUICK]" if quick else ""))
    print(f"{'='*60}")
    for k in AGG_KEYS:
        vals = [m[k] for m in finetune_fold_metrics
                if not np.isnan(m.get(k, np.nan))]
        if vals:
            print(f"  {k:25s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    if not quick:
        print("\n  Bootstrap 95% CIs (Stage 2, 1000 resamples):")
        boot_df = bootstrap_ci(finetune_fold_metrics, AGG_KEYS,
                                n_bootstrap=1000, ci_level=0.95,
                                seed=cfg["seed"])
        for _, row in boot_df.iterrows():
            print(f"  {row['metric']:25s}: {row['mean']:.4f}  "
                  f"95% CI [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]")
        boot_df.to_csv(
            os.path.join(finetune_dirs["metrics"],
                         "cv_bootstrap_ci_stage2.csv"), index=False)

    plot_metric_bars(finetune_fold_metrics, class_names, finetune_dirs["plots"])
    if all_patient_dfs:
        combined_pat = pd.concat(all_patient_dfs, ignore_index=True)
        plot_patient_scatter(combined_pat, dirs["plots"], "curriculum_all_folds")
    save_timing(timing_records, dirs["metrics"])

    for stage_label, fold_list, out_dir in [
        ("stage1", pretrain_fold_metrics, pretrain_dirs["metrics"]),
        ("stage2", finetune_fold_metrics, finetune_dirs["metrics"]),
    ]:
        serialisable = []
        for m in fold_list:
            row = {"stage": stage_label}
            for k in AGG_KEYS + ["specificity", "fold", "best_epoch"]:
                if k in m: row[k] = m[k]
            for cls, cv in m.get("per_class", {}).items():
                for metric in PC_KEYS:
                    row[f"{cls}_{metric}"] = cv.get(metric, np.nan)
            serialisable.append(row)
        pd.DataFrame(serialisable).to_csv(
            os.path.join(out_dir, f"per_fold_metrics_{stage_label}.csv"),
            index=False)

    return finetune_fold_metrics, best_s2_epochs, timing_records, s2_fold_normalizers


# ============================================================
# 17. HELD-OUT TEST EVALUATION
# ============================================================
def load_test_df(cfg):
    path = _test_csv_from_cfg(cfg)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Test CSV not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    df["label"] = df["fitzpatrick"].astype(int) - 1
    print(f"  Test set: {len(df)} samples")
    print(f"  Label distribution:\n"
          f"{df['label'].value_counts().sort_index().to_string()}\n")
    return df


def run_held_out_test(cfg, dirs, device, best_epochs,
                      all_fold_metrics, cv_timing, fold_normalizers):
    print(f"\n{'='*60}\n  HELD-OUT TEST EVALUATION\n{'='*60}")
    class_names = CLASS_NAMES
    test_df     = load_test_df(cfg)

    optimal_epochs = (max(1, int(np.mean(best_epochs)))
                      if best_epochs else cfg["num_epochs"])
    print(f"  Optimal epochs from CV: {best_epochs} → {optimal_epochs}\n")

    print("  Assembling full train+val dataset from all fold CSVs...")
    all_tvdfs = []
    tmpl = _fold_csv_tmpl_from_cfg(cfg)
    for fold in range(1, cfg["num_folds"] + 1):
        fdf = pd.read_csv(
            os.path.join(cfg["fold_csv_dir"], tmpl.format(fold=fold)))
        fdf["split"] = fdf["split"].str.strip().str.lower()
        all_tvdfs.append(fdf[fdf["split"].isin(["train", "valid"])])
    full_df = (pd.concat(all_tvdfs, ignore_index=True)
               .drop_duplicates(subset=["image_path_512"])
               .reset_index(drop=True))
    full_df["label"] = full_df["fitzpatrick"].astype(int) - 1
    print(f"  Full train+val (deduplicated): {len(full_df)} samples\n")

    ita_col          = cfg.get("ita_col", "ITA_value_spectral_spatial")
    final_normalizer = ITANormalizer()
    full_ita_normed  = final_normalizer.fit_transform(
        extract_ita_array(full_df, ita_col))
    with open(os.path.join(dirs["models"], "ita_normalizer_final.json"), "w") as f:
        json.dump(final_normalizer.state_dict(), f, indent=2)

    full_soft  = _build_soft_labels(full_df, cfg)
    full_prov  = _get_prov_weights(full_df, cfg)
    full_ds    = SkinDatasetITA(full_df, full_soft, full_ita_normed, full_prov,
                                is_train=True, cfg=cfg)
    full_labels = full_df["label"].values

    if cfg["use_weighted_sampler"]:
        cc = np.maximum(np.bincount(full_labels, minlength=cfg["num_classes"]), 1)
        sw = 1.0 / cc[full_labels]
        sampler = WeightedRandomSampler(torch.tensor(sw, dtype=torch.float32),
                                        num_samples=len(full_ds), replacement=True)
        full_loader = DataLoader(full_ds, batch_size=cfg["batch_size"],
                                 sampler=sampler, num_workers=cfg["num_workers"],
                                 pin_memory=True, drop_last=True)
    else:
        full_loader = DataLoader(full_ds, batch_size=cfg["batch_size"],
                                 shuffle=True, num_workers=cfg["num_workers"],
                                 pin_memory=True, drop_last=True)

    final_model   = build_model(cfg, device)
    class_weights = compute_class_weights(full_labels, cfg["num_classes"])
    criterion = FocalKLLoss(
        class_weights=class_weights if cfg["use_class_weights"] else None,
        focal_gamma=cfg["focal_gamma"], use_focal=cfg["use_focal_loss"],
        use_class_weights=cfg["use_class_weights"],
    )
    optimizer = optim.AdamW(final_model.parameters(),
                            lr=cfg["learning_rate"],
                            weight_decay=cfg["weight_decay"])
    scheduler = build_scheduler(optimizer, cfg, len(full_loader))

    final_train_losses, final_train_accs = [], []
    t_train = time.time()
    for epoch in range(optimal_epochs):
        ep_loss, ep_acc, _ = _run_one_epoch(
            final_model, full_loader, criterion, optimizer, scheduler,
            cfg, epoch, optimal_epochs, full_df, device,
            desc=f"Final Model Epoch {epoch+1}/{optimal_epochs}",
            alpha_start=cfg.get("alpha_start", 0.20),
            alpha_end  =cfg.get("alpha_end",   0.10))
        final_train_losses.append(ep_loss)
        final_train_accs.append(ep_acc)
        print(f"  Final | Ep {epoch+1:3d}/{optimal_epochs} | "
              f"Loss {ep_loss:.4f} | Acc {ep_acc:.4f}")
    train_time = time.time() - t_train

    final_model_path = os.path.join(
        dirs["models"], f"final_model_ep{optimal_epochs}.pth")
    torch.save(final_model.state_dict(), final_model_path)
    print(f"\n  Final model saved → {final_model_path}")
    _save_final_model_curves(final_train_losses, final_train_accs, dirs["plots"])

    test_loader  = build_test_dataloader(test_df, cfg, final_normalizer)
    t_inf        = time.time()
    test_results = evaluate(final_model, test_loader, criterion, device, cfg)

    if (cfg.get("use_tta", True)
            and not cfg.get("debug_mode", False)
            and not cfg.get("quick_mode", False)):
        print("\n  Running TTA on test set...")
        tta_results = tta_predict(final_model, test_df, cfg, device,
                                   final_normalizer)
        tta_metrics = compute_extended_metrics(
            tta_results["labels"], tta_results["preds"],
            tta_results["mean_probs"], cfg["num_classes"], class_names)
        tta_df = test_df.copy().reset_index(drop=True)
        tta_df["tta_prediction"] = tta_results["preds"]
        for i in range(cfg["num_classes"]):
            tta_df[f"tta_prob_{class_names[i]}"] = tta_results["mean_probs"][:, i]
        tta_df.to_csv(
            os.path.join(dirs["metrics"], "test_predictions_tta.csv"), index=False)
        pd.DataFrame(
            [{"metric": k, "value": tta_metrics.get(k, np.nan),
              "level": "aggregate", "method": "TTA"} for k in AGG_KEYS]
            + [{"metric": f"{cls}_{m}", "value": v.get(m, np.nan),
                "level": "per_class", "method": "TTA"}
               for cls, v in tta_metrics["per_class"].items() for m in PC_KEYS]
        ).to_csv(os.path.join(dirs["metrics"], "test_metrics_tta.csv"), index=False)
        plot_confusion_matrix(tta_results["labels"], tta_results["preds"],
                              class_names, "test_tta", dirs["plots"])
        plot_roc_curve(tta_results["labels"], tta_results["mean_probs"],
                       class_names, "test_tta", dirs["plots"])
        print(f"  TTA Accuracy: {tta_metrics['accuracy']:.4f}  "
              f"(standard: {test_results['accuracy']:.4f})")

    inf_time     = time.time() - t_inf
    test_metrics = compute_extended_metrics(
        test_results["labels"], test_results["preds"],
        test_results["probs"], cfg["num_classes"], class_names)

    print(f"\n  {'─'*40}\n  HELD-OUT TEST RESULTS\n  {'─'*40}")
    for k in AGG_KEYS:
        val = test_metrics.get(k, np.nan)
        print(f"  {k:25s}: {val:.4f}" if not np.isnan(val)
              else f"  {k:25s}: N/A")

    pred_df = test_df.copy().reset_index(drop=True)
    pred_df["prediction"] = test_results["preds"]
    pred_df["true_label"] = test_results["labels"]
    for i in range(cfg["num_classes"]):
        pred_df[f"prob_{class_names[i]}"] = test_results["probs"][:, i]
    pred_df.to_csv(
        os.path.join(dirs["metrics"], "test_predictions.csv"), index=False)

    pd.DataFrame(
        [{"metric": k, "value": test_metrics.get(k, np.nan),
          "level": "aggregate"} for k in AGG_KEYS]
        + [{"metric": f"{cls}_{m}", "value": v.get(m, np.nan),
            "level": "per_class"}
           for cls, v in test_metrics["per_class"].items() for m in PC_KEYS]
    ).to_csv(os.path.join(dirs["metrics"], "test_metrics.csv"), index=False)

    plot_confusion_matrix(test_results["labels"], test_results["preds"],
                          class_names, "test", dirs["plots"])
    plot_roc_curve(test_results["labels"], test_results["probs"],
                   class_names, "test", dirs["plots"])
    plot_pr_curve(test_results["labels"], test_results["probs"],
                  class_names, "test", dirs["plots"])
    plot_calibration_curve(test_results["labels"], test_results["probs"],
                           class_names, "test", dirs["plots"])

    if cfg["use_mc_dropout"]:
        unc = mc_dropout_predict(final_model, test_loader, device,
                                 cfg["mc_dropout_passes"])
        unc_df = pd.DataFrame({
            "predictive_entropy": unc["predictive_entropy"],
            "mutual_information": unc["mutual_information"],
            "true_label":         test_results["labels"],
            "pred_label":         test_results["preds"],
        })
        unc_df.to_csv(
            os.path.join(dirs["uncertainty"], "uncertainty_test.csv"), index=False)
        plot_uncertainty_boxplot(unc_df, class_names, "test",
                                 dirs["plots"], cfg["mc_dropout_passes"])

    test_soft_tmp = _build_soft_labels(test_df, cfg)
    test_prov_tmp = _get_prov_weights(test_df, cfg)
    test_ita_tmp  = final_normalizer.transform(
        extract_ita_array(test_df,
                          cfg.get("ita_col", "ITA_value_spectral_spatial")))
    test_ds_tmp = SkinDatasetITA(test_df, test_soft_tmp, test_ita_tmp,
                                  test_prov_tmp, is_train=False, cfg=cfg)
    pat_df = compute_patient_metrics(
        test_results["labels"], test_results["preds"],
        test_results["probs"], test_ds_tmp.patient_ids, cfg["num_classes"])
    plot_patient_scatter(pat_df, dirs["plots"], "test")

    _mean_arr = np.array([0.485, 0.456, 0.406])
    _std_arr  = np.array([0.229, 0.224, 0.225])
    generate_and_save_gradcam(
        final_model, test_loader, class_names, dirs["gradcam"],
        device, cfg, prefix="test",
        pred_labels=test_results["preds"],
        true_labels=test_results["labels"],
        mean_arr=_mean_arr, std_arr=_std_arr)

    all_timing = cv_timing + [{
        "phase":             "held_out_test",
        "fold":              "—",
        "train_time_s":      round(train_time, 2),
        "inference_time_s":  round(inf_time, 2),
        "total_fold_time_s": round(train_time + inf_time, 2),
        "n_samples":         len(test_df),
    }]
    save_timing(all_timing, dirs["metrics"])
    print(f"\n  All test outputs saved → {dirs['base']}")
    return test_metrics


# ============================================================
# 18. CURRICULUM HELD-OUT TEST
# ============================================================
def run_curriculum_held_out_test(cfg, dirs, device, best_s2_epochs,
                                  finetune_fold_metrics, cv_timing):
    print(f"\n{'='*60}\n  CURRICULUM HELD-OUT TEST EVALUATION\n{'='*60}")
    class_names = CLASS_NAMES
    test_df     = load_test_df(cfg)

    optimal_s1_epochs = (
        cfg.get("debug_curriculum_pretrain_epochs", 2)
        if cfg.get("debug_mode", False)
        else cfg.get("num_epochs", 50)
    )
    optimal_s2_epochs = (
        cfg.get("debug_curriculum_finetune_epochs", 2)
        if cfg.get("debug_mode", False)
        else max(1, int(np.mean(best_s2_epochs))) if best_s2_epochs
        else cfg.get("curriculum_finetune_epochs", 20)
    )
    print(f"  Optimal Stage 1 epochs : {optimal_s1_epochs}")
    print(f"  Optimal Stage 2 epochs : {best_s2_epochs} → {optimal_s2_epochs}\n")

    print("  Assembling full train+val dataset from all fold CSVs...")
    all_pretrain, all_finetune = [], []
    tmpl = _fold_csv_tmpl_from_cfg(cfg)
    for fold in range(1, cfg["num_folds"] + 1):
        fdf = pd.read_csv(
            os.path.join(cfg["fold_csv_dir"], tmpl.format(fold=fold)),
            low_memory=False)
        fdf["split"] = fdf["split"].str.strip().str.lower()
        all_pretrain.append(
            fdf[fdf["split"].isin(["pretrain_train", "pretrain_valid"])])
        all_finetune.append(fdf[fdf["split"].isin(["train", "valid"])])

    pretrain_full = (pd.concat(all_pretrain, ignore_index=True)
                     .drop_duplicates(subset=["image_path_512"])
                     .reset_index(drop=True))
    finetune_full = (pd.concat(all_finetune, ignore_index=True)
                     .drop_duplicates(subset=["image_path_512"])
                     .reset_index(drop=True))
    pretrain_full["label"] = pretrain_full["fitzpatrick"].astype(int) - 1
    finetune_full["label"] = finetune_full["fitzpatrick"].astype(int) - 1
    print(f"  Stage 1 pool: {len(pretrain_full):,} samples")
    print(f"  Stage 2 pool: {len(finetune_full):,} samples\n")

    ita_col     = cfg.get("ita_col", "ITA_value_spectral_spatial")
    final_model = build_model(cfg, device)

    # ── Stage 1 ──────────────────────────────────────────────
    print(f"  {'─'*40}\n  Final Stage 1 — Pre-train "
          f"({optimal_s1_epochs} epochs)\n  {'─'*40}")
    s1_normalizer = ITANormalizer()
    s1_ita_normed = s1_normalizer.fit_transform(
        extract_ita_array(pretrain_full, ita_col))
    with open(os.path.join(dirs["curriculum_pretrain"], "models",
                           "ita_normalizer_final_s1.json"), "w") as f:
        json.dump(s1_normalizer.state_dict(), f, indent=2)

    s1_soft   = _build_soft_labels(pretrain_full, cfg)
    s1_prov   = _get_prov_weights(pretrain_full, cfg)
    s1_ds     = SkinDatasetITA(pretrain_full, s1_soft, s1_ita_normed, s1_prov,
                                is_train=True, cfg=cfg)
    s1_labels = pretrain_full["label"].values
    if cfg["use_weighted_sampler"]:
        cc = np.maximum(np.bincount(s1_labels, minlength=cfg["num_classes"]), 1)
        sw = 1.0 / cc[s1_labels]
        s1_loader = DataLoader(
            s1_ds, batch_size=cfg["batch_size"],
            sampler=WeightedRandomSampler(
                torch.tensor(sw, dtype=torch.float32),
                num_samples=len(s1_ds), replacement=True),
            num_workers=cfg["num_workers"], pin_memory=True, drop_last=True)
    else:
        s1_loader = DataLoader(s1_ds, batch_size=cfg["batch_size"], shuffle=True,
                               num_workers=cfg["num_workers"], pin_memory=True,
                               drop_last=True)

    s1_cw = compute_class_weights(s1_labels, cfg["num_classes"])
    s1_criterion = FocalKLLoss(
        class_weights=s1_cw if cfg["use_class_weights"] else None,
        focal_gamma=cfg["focal_gamma"], use_focal=cfg["use_focal_loss"],
        use_class_weights=cfg["use_class_weights"])
    s1_optimizer = optim.AdamW(final_model.parameters(),
                               lr=cfg["learning_rate"],
                               weight_decay=cfg["weight_decay"])
    s1_scheduler = build_scheduler(s1_optimizer, cfg, len(s1_loader))

    t_s1 = time.time()
    for epoch in range(optimal_s1_epochs):
        ep_loss, ep_acc, _ = _run_one_epoch(
            final_model, s1_loader, s1_criterion, s1_optimizer, s1_scheduler,
            cfg, epoch, optimal_s1_epochs, pretrain_full, device,
            desc=f"Final S1 Ep {epoch+1}/{optimal_s1_epochs}",
            alpha_start=cfg.get("alpha_start", 0.20),
            alpha_end  =cfg.get("alpha_end",   0.10))
        print(f"  Final S1 | Ep {epoch+1:3d} | "
              f"Loss {ep_loss:.4f} | Acc {ep_acc:.4f}")
    s1_time = time.time() - t_s1
    torch.save(final_model.state_dict(), os.path.join(
        dirs["curriculum_pretrain"], "models",
        f"final_pretrain_ep{optimal_s1_epochs}.pth"))

    # ── Stage 2 ──────────────────────────────────────────────
    print(f"\n  {'─'*40}\n  Final Stage 2 — Fine-tune "
          f"({optimal_s2_epochs} epochs)\n  {'─'*40}")
    s2_normalizer = ITANormalizer()
    s2_ita_normed = s2_normalizer.fit_transform(
        extract_ita_array(finetune_full, ita_col))
    with open(os.path.join(dirs["curriculum_finetune"], "models",
                           "ita_normalizer_final_s2.json"), "w") as f:
        json.dump(s2_normalizer.state_dict(), f, indent=2)

    n_freeze = cfg.get("curriculum_freeze_layers", 2)
    freeze_cnn_layers(final_model, n_freeze, cfg)

    finetune_alpha = cfg.get("curriculum_finetune_alpha", 0.08)
    s2_soft   = _build_soft_labels(finetune_full, cfg, alpha=finetune_alpha)
    s2_prov   = _get_prov_weights(finetune_full, cfg)
    s2_ds     = SkinDatasetITA(finetune_full, s2_soft, s2_ita_normed, s2_prov,
                                is_train=True, cfg=cfg)
    s2_labels = finetune_full["label"].values
    if cfg["use_weighted_sampler"]:
        cc = np.maximum(np.bincount(s2_labels, minlength=cfg["num_classes"]), 1)
        sw = 1.0 / cc[s2_labels]
        s2_loader = DataLoader(
            s2_ds, batch_size=cfg["batch_size"],
            sampler=WeightedRandomSampler(
                torch.tensor(sw, dtype=torch.float32),
                num_samples=len(s2_ds), replacement=True),
            num_workers=cfg["num_workers"], pin_memory=True, drop_last=True)
    else:
        s2_loader = DataLoader(s2_ds, batch_size=cfg["batch_size"], shuffle=True,
                               num_workers=cfg["num_workers"], pin_memory=True,
                               drop_last=True)

    s2_cw = compute_class_weights(s2_labels, cfg["num_classes"])
    s2_criterion = FocalKLLoss(
        class_weights=s2_cw if cfg["use_class_weights"] else None,
        focal_gamma=cfg["focal_gamma"], use_focal=cfg["use_focal_loss"],
        use_class_weights=cfg["use_class_weights"])
    s2_optimizer = build_finetune_optimizer(final_model, cfg)
    s2_scheduler = build_scheduler(
        s2_optimizer, cfg, len(s2_loader),
        scheduler_name=cfg.get("curriculum_finetune_scheduler", "cosine"))

    t_s2 = time.time()
    for epoch in range(optimal_s2_epochs):
        ep_loss, ep_acc, _ = _run_one_epoch(
            final_model, s2_loader, s2_criterion, s2_optimizer, s2_scheduler,
            cfg, epoch, optimal_s2_epochs, finetune_full, device,
            desc=f"Final S2 Ep {epoch+1}/{optimal_s2_epochs}",
            alpha_start=cfg.get("curriculum_finetune_alpha_start", 0.12),
            alpha_end  =cfg.get("curriculum_finetune_alpha_end",   0.05))
        print(f"  Final S2 | Ep {epoch+1:3d} | "
              f"Loss {ep_loss:.4f} | Acc {ep_acc:.4f}")
    s2_time = time.time() - t_s2

    unfreeze_all(final_model)
    torch.save(final_model.state_dict(), os.path.join(
        dirs["curriculum_finetune"], "models",
        f"final_finetune_ep{optimal_s2_epochs}.pth"))

    test_loader  = build_test_dataloader(test_df, cfg, s2_normalizer)
    t_inf        = time.time()
    test_results = evaluate(final_model, test_loader, s2_criterion, device, cfg)

    if (cfg.get("use_tta", True)
            and not cfg.get("debug_mode", False)
            and not cfg.get("quick_mode", False)):
        print("\n  Running TTA on test set (curriculum final model)...")
        tta_results = tta_predict(final_model, test_df, cfg, device, s2_normalizer)
        tta_metrics = compute_extended_metrics(
            tta_results["labels"], tta_results["preds"],
            tta_results["mean_probs"], cfg["num_classes"], class_names)
        tta_df = test_df.copy().reset_index(drop=True)
        tta_df["tta_prediction"] = tta_results["preds"]
        for i in range(cfg["num_classes"]):
            tta_df[f"tta_prob_{class_names[i]}"] = tta_results["mean_probs"][:, i]
        tta_df.to_csv(
            os.path.join(dirs["curriculum_finetune"], "metrics",
                         "test_predictions_tta.csv"), index=False)
        plot_confusion_matrix(
            tta_results["labels"], tta_results["preds"],
            class_names, "curriculum_test_tta",
            os.path.join(dirs["curriculum_finetune"], "plots"))
        print(f"  TTA Accuracy: {tta_metrics['accuracy']:.4f}  "
              f"(standard: {test_results['accuracy']:.4f})")

    inf_time     = time.time() - t_inf
    test_metrics = compute_extended_metrics(
        test_results["labels"], test_results["preds"],
        test_results["probs"], cfg["num_classes"], class_names)

    print(f"\n  {'─'*40}\n  CURRICULUM HELD-OUT TEST RESULTS\n  {'─'*40}")
    for k in AGG_KEYS:
        val = test_metrics.get(k, np.nan)
        print(f"  {k:25s}: {val:.4f}" if not np.isnan(val)
              else f"  {k:25s}: N/A")

    ft_plots   = os.path.join(dirs["curriculum_finetune"], "plots")
    ft_metrics = os.path.join(dirs["curriculum_finetune"], "metrics")

    pred_df = test_df.copy().reset_index(drop=True)
    pred_df["prediction"] = test_results["preds"]
    pred_df["true_label"] = test_results["labels"]
    for i in range(cfg["num_classes"]):
        pred_df[f"prob_{class_names[i]}"] = test_results["probs"][:, i]
    pred_df.to_csv(os.path.join(ft_metrics, "test_predictions.csv"), index=False)

    pd.DataFrame(
        [{"metric": k, "value": test_metrics.get(k, np.nan),
          "level": "aggregate"} for k in AGG_KEYS]
        + [{"metric": f"{cls}_{m}", "value": v.get(m, np.nan),
            "level": "per_class"}
           for cls, v in test_metrics["per_class"].items() for m in PC_KEYS]
    ).to_csv(os.path.join(ft_metrics, "test_metrics.csv"), index=False)

    plot_confusion_matrix(test_results["labels"], test_results["preds"],
                          class_names, "curriculum_test", ft_plots)
    plot_roc_curve(test_results["labels"], test_results["probs"],
                   class_names, "curriculum_test", ft_plots)
    plot_pr_curve(test_results["labels"], test_results["probs"],
                  class_names, "curriculum_test", ft_plots)
    plot_calibration_curve(test_results["labels"], test_results["probs"],
                           class_names, "curriculum_test", ft_plots)

    if cfg["use_mc_dropout"]:
        unc = mc_dropout_predict(final_model, test_loader, device,
                                 cfg["mc_dropout_passes"])
        unc_df = pd.DataFrame({
            "predictive_entropy": unc["predictive_entropy"],
            "mutual_information": unc["mutual_information"],
            "true_label":         test_results["labels"],
            "pred_label":         test_results["preds"],
        })
        unc_df.to_csv(
            os.path.join(dirs["curriculum_finetune"], "uncertainty",
                         "uncertainty_test.csv"), index=False)
        plot_uncertainty_boxplot(unc_df, class_names, "curriculum_test",
                                 ft_plots, cfg["mc_dropout_passes"])

    _mean_arr = np.array([0.485, 0.456, 0.406])
    _std_arr  = np.array([0.229, 0.224, 0.225])
    generate_and_save_gradcam(
        final_model, test_loader, class_names,
        os.path.join(dirs["curriculum_finetune"], "gradcam"),
        device, cfg, prefix="curriculum_test",
        pred_labels=test_results["preds"],
        true_labels=test_results["labels"],
        mean_arr=_mean_arr, std_arr=_std_arr)

    all_timing = cv_timing + [
        {"phase": "curriculum_final_stage1", "fold": "all",
         "train_time_s": round(s1_time, 2), "inference_time_s": 0,
         "total_fold_time_s": round(s1_time, 2),
         "n_samples": len(pretrain_full)},
        {"phase": "curriculum_final_stage2", "fold": "all",
         "train_time_s": round(s2_time, 2),
         "inference_time_s": round(inf_time, 2),
         "total_fold_time_s": round(s2_time + inf_time, 2),
         "n_samples": len(finetune_full)},
    ]
    save_timing(all_timing, dirs["metrics"])
    print(f"\n  All curriculum test outputs saved → {dirs['base']}")
    return test_metrics


# ============================================================
# 19. OPTUNA INTEGRATION
# ============================================================
def run_optuna(cfg, device):
    if not OPTUNA_AVAILABLE:
        print("[ERROR] optuna not installed."); return

    dirs = make_dirs(cfg, mode="optuna")

    def objective(trial):
        tc = copy.deepcopy(cfg)
        tc["learning_rate"]     = trial.suggest_float("lr", 1e-5, 1e-4, log=True)
        tc["weight_decay"]      = trial.suggest_float("wd", 1e-5, 1e-3, log=True)
        tc["soft_label_alpha"]  = trial.suggest_float("alpha", 0.05, 0.25)
        tc["focal_gamma"]       = trial.suggest_float("focal_gamma", 0.5, 3.0)
        tc["ita_proj_dim"]      = trial.suggest_categorical(
            "ita_proj_dim", [4, 8, 16, 32])
        tc["batch_size"]        = trial.suggest_categorical(
            "batch_size", [16, 32, 64])
        tc["patient_kl_weight"] = trial.suggest_float("pat_kl_w", 0.01, 0.3)

        fold = cfg["optuna_fold"]
        train_df, val_df = load_fold_df(tc, fold)

        n_per_class = 300
        train_df = pd.concat([
            train_df[train_df["label"] == lbl].sample(
                min(len(train_df[train_df["label"] == lbl]), n_per_class),
                random_state=tc["seed"])
            for lbl in range(tc["num_classes"])
            if len(train_df[train_df["label"] == lbl]) > 0
        ], ignore_index=True)
        val_df = pd.concat([
            val_df[val_df["label"] == lbl].sample(
                min(len(val_df[val_df["label"] == lbl]), n_per_class),
                random_state=tc["seed"])
            for lbl in range(tc["num_classes"])
            if len(val_df[val_df["label"] == lbl]) > 0
        ], ignore_index=True)

        print(f"\n  [Trial {trial.number}] Subsampled: "
              f"train={len(train_df)}, val={len(val_df)}")
        print(f"  [Trial {trial.number}] Hyperparams:")
        print(f"    lr={tc['learning_rate']:.2e}, wd={tc['weight_decay']:.2e}, "
              f"alpha={tc['soft_label_alpha']:.3f}, "
              f"focal_gamma={tc['focal_gamma']:.2f}")
        print(f"    ita_proj_dim={tc['ita_proj_dim']}, bs={tc['batch_size']}, "
              f"pat_kl_w={tc['patient_kl_weight']:.3f}")

        try:
            _, metrics, _, _ = train_one_fold(
                fold, train_df, val_df, tc,
                {"models": dirs["models"], "plots": dirs["plots"]},
                device, trial=trial)
            val_acc = metrics["accuracy"]
            print(f"  [Trial {trial.number}] Final accuracy: {val_acc:.4f}")
            return val_acc
        except optuna.exceptions.TrialPruned:
            print(f"  [Trial {trial.number}] Pruned.")
            raise
        except Exception as e:
            print(f"  [Trial {trial.number}] Failed: {str(e)[:100]}")
            return 0.0

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=cfg["seed"]),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=2, interval_steps=1),
        study_name=build_experiment_name(cfg))

    print(f"\n[Optuna] Starting optimization...")
    print(f"[Optuna] Study name : {study.study_name}")
    print(f"[Optuna] n_trials   : {cfg['optuna_n_trials']}")
    print(f"[Optuna] Sampler    : TPE | Pruner: Median (aggressive)")

    study.optimize(objective, n_trials=cfg["optuna_n_trials"],
                   show_progress_bar=True)

    print(f"\n{'='*60}\n[Optuna] OPTIMIZATION COMPLETE\n{'='*60}")
    print(f"[Optuna] Best trial    : {study.best_trial.number}")
    print(f"[Optuna] Best accuracy : {study.best_value:.4f}")
    print(f"[Optuna] Best hyperparameters:")
    for k, v in study.best_params.items():
        print(f"         {k:20s}: {v}")

    with open(os.path.join(dirs["base"], "best_params.json"), "w") as f:
        json.dump(study.best_params, f, indent=2)
    study.trials_dataframe().to_csv(
        os.path.join(dirs["base"], "optuna_trials.csv"), index=False)

    try:
        import optuna.visualization as ov
        ov.plot_optimization_history(study).write_html(
            os.path.join(dirs["plots"], "optuna_history.html"))
        ov.plot_param_importances(study).write_html(
            os.path.join(dirs["plots"], "optuna_importance.html"))
        ov.plot_slice(study).write_html(
            os.path.join(dirs["plots"], "optuna_slice.html"))
    except Exception as e:
        print(f"[Optuna] plotly unavailable — skipping HTML plots: {e}")

    return study.best_params


# ============================================================
# 20. CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Fitzpatrick FST — Multimodal CNN + ITA Late Fusion")
    p.add_argument("--mode",
                   choices=["train", "debug", "quick", "optuna",
                             "curriculum", "debug_curriculum",
                             "quick_curriculum"],
                   default="train")
    p.add_argument("--arch", choices=["resnet18", "efficientnet_b0"],
                   default="resnet18")
    p.add_argument("--tag",     default="baseline")
    p.add_argument("--folds",   type=int,   default=5)
    p.add_argument("--epochs",  type=int,   default=50)
    p.add_argument("--lr",      type=float, default=5e-5)
    p.add_argument("--wd",      type=float, default=3e-4)
    p.add_argument("--bs",      type=int,   default=32)
    p.add_argument("--seed",    type=int,   default=42)
    p.add_argument("--workers", type=int,   default=4)
    p.add_argument("--patience",        type=int,   default=8)
    p.add_argument("--min_delta",       type=float, default=2e-4)
    p.add_argument("--focal_gamma",     type=float, default=2.0)
    p.add_argument("--mc_dropout_rate", type=float, default=0.4)
    p.add_argument("--data_split",
                   choices=list(DATA_SPLIT_VARIANTS.keys()),
                   default="standard", dest="data_split")
    p.add_argument("--quick_fold",      type=int, default=1)
    p.add_argument("--quick_mc_passes", type=int, default=5)
    p.add_argument("--no_ita_fusion",  action="store_true")
    p.add_argument("--ita_proj_dim",   type=int, default=None)
    p.add_argument("--no_soft_labels",   action="store_true")
    p.add_argument("--no_focal",         action="store_true")
    p.add_argument("--no_class_weights", action="store_true")
    p.add_argument("--no_wsampler",      action="store_true")
    p.add_argument("--no_min_aug",       action="store_true")
    p.add_argument("--no_pat_cons",      action="store_true")
    p.add_argument("--no_mc_dropout",    action="store_true")
    p.add_argument("--use_contrastive",  action="store_true")
    p.add_argument("--use_mixup",   action="store_true")
    p.add_argument("--mixup_alpha", type=float, default=0.4)
    p.add_argument("--mixup_prob",  type=float, default=0.5)
    p.add_argument("--use_exposure_augment", action="store_true")
    p.add_argument("--exposure_aug_prob", type=float, default=0.3)
    p.add_argument("--use_provenance_weighting", action="store_true")
    p.add_argument("--prov_col", type=str, default="dataset")
    p.add_argument("--prov_w_tla_i4i",        type=float, default=None)
    p.add_argument("--prov_w_eh_web",          type=float, default=None)
    p.add_argument("--prov_w_pad_ufes_20",     type=float, default=None)
    p.add_argument("--prov_w_passion",         type=float, default=None)
    p.add_argument("--prov_w_scin",            type=float, default=None)
    p.add_argument("--prov_w_fitzpatrick17k",  type=float, default=None)
    p.add_argument("--use_adaptive_smoothing", action="store_true")
    p.add_argument("--alpha_tla_i4i",        type=float, default=None)
    p.add_argument("--alpha_eh_web",          type=float, default=None)
    p.add_argument("--alpha_pad_ufes_20",     type=float, default=None)
    p.add_argument("--alpha_passion",         type=float, default=None)
    p.add_argument("--alpha_scin",            type=float, default=None)
    p.add_argument("--alpha_fitzpatrick17k",  type=float, default=None)
    p.add_argument("--curriculum_finetune_epochs", type=int,   default=None)
    p.add_argument("--curriculum_freeze_layers",   type=int,   default=None)
    p.add_argument("--curriculum_finetune_lr",     type=float, default=None)
    p.add_argument("--curriculum_finetune_alpha",  type=float, default=None)
    p.add_argument("--curriculum_dataset_col",     type=str,   default=None)
    p.add_argument("--no_curriculum_test",         action="store_true")
    p.add_argument("--optuna_trials", type=int, default=30)
    p.add_argument("--optuna_fold",   type=int, default=1)
    p.add_argument("--config",        default=None)
    p.add_argument("--test_csv",      default=None)
    p.add_argument("--resume",        action="store_true")
    p.add_argument("--save_all_test_gradcam", action="store_true")
    return p.parse_args()


def apply_args_to_cfg(args, cfg):
    cfg = copy.deepcopy(cfg)

    cfg["arch"]            = args.arch
    cfg["experiment_tag"]  = args.tag
    cfg["num_folds"]       = args.folds
    cfg["num_epochs"]      = args.epochs
    cfg["learning_rate"]   = args.lr
    cfg["weight_decay"]    = args.wd
    cfg["batch_size"]      = args.bs
    cfg["seed"]            = args.seed
    cfg["num_workers"]     = args.workers
    cfg["patience"]        = args.patience
    cfg["min_delta"]       = args.min_delta
    cfg["focal_gamma"]     = args.focal_gamma
    cfg["mc_dropout_rate"] = args.mc_dropout_rate
    cfg["data_split_variant"] = args.data_split
    cfg["optuna_n_trials"] = args.optuna_trials
    cfg["optuna_fold"]     = args.optuna_fold

    if args.mode in ("quick", "quick_curriculum"):
        cfg["quick_mode"]              = True
        cfg["quick_fold"]              = args.quick_fold
        cfg["quick_mc_dropout_passes"] = args.quick_mc_passes
    else:
        cfg["quick_mode"] = False

    if args.no_ita_fusion:    cfg["use_ita_fusion"]          = False
    if args.ita_proj_dim:     cfg["ita_proj_dim"]             = args.ita_proj_dim
    if args.no_soft_labels:   cfg["use_soft_labels"]          = False
    if args.no_focal:         cfg["use_focal_loss"]           = False
    if args.no_class_weights: cfg["use_class_weights"]        = False
    if args.no_wsampler:      cfg["use_weighted_sampler"]     = False
    if args.no_min_aug:       cfg["use_minority_augment"]     = False
    if args.no_pat_cons:      cfg["use_patient_consistency"]  = False
    if args.no_mc_dropout:    cfg["use_mc_dropout"]           = False
    if args.use_contrastive:  cfg["use_contrastive"]          = True

    if args.use_mixup:
        cfg["use_mixup"]   = True
        cfg["mixup_alpha"] = args.mixup_alpha
        cfg["mixup_prob"]  = args.mixup_prob

    if args.use_exposure_augment:
        cfg["use_exposure_augment"] = True
        cfg["exposure_aug_prob"]    = args.exposure_aug_prob

    if args.use_provenance_weighting:
        cfg["use_provenance_weighting"] = True
        cfg["provenance_col"]           = args.prov_col
        _prov_map = cfg["provenance_loss_weights"].copy()
        if args.prov_w_tla_i4i       is not None: _prov_map["tla_i4i"]        = args.prov_w_tla_i4i
        if args.prov_w_eh_web         is not None: _prov_map["eh_web"]          = args.prov_w_eh_web
        if args.prov_w_pad_ufes_20    is not None: _prov_map["pad-ufes-20"]     = args.prov_w_pad_ufes_20
        if args.prov_w_passion        is not None: _prov_map["passion"]          = args.prov_w_passion
        if args.prov_w_scin           is not None: _prov_map["scin"]             = args.prov_w_scin
        if args.prov_w_fitzpatrick17k is not None: _prov_map["fitzpatrick17k"]   = args.prov_w_fitzpatrick17k
        cfg["provenance_loss_weights"] = _prov_map

    if args.use_adaptive_smoothing:
        cfg["use_adaptive_smoothing"] = True
        _alpha_map = cfg["adaptive_alpha_map"].copy()
        if args.alpha_tla_i4i        is not None: _alpha_map["tla_i4i"]        = args.alpha_tla_i4i
        if args.alpha_eh_web          is not None: _alpha_map["eh_web"]          = args.alpha_eh_web
        if args.alpha_pad_ufes_20     is not None: _alpha_map["pad-ufes-20"]     = args.alpha_pad_ufes_20
        if args.alpha_passion         is not None: _alpha_map["passion"]          = args.alpha_passion
        if args.alpha_scin            is not None: _alpha_map["scin"]             = args.alpha_scin
        if args.alpha_fitzpatrick17k  is not None: _alpha_map["fitzpatrick17k"]   = args.alpha_fitzpatrick17k
        cfg["adaptive_alpha_map"] = _alpha_map

    if args.curriculum_finetune_epochs is not None:
        cfg["curriculum_finetune_epochs"] = args.curriculum_finetune_epochs
    if args.curriculum_freeze_layers is not None:
        cfg["curriculum_freeze_layers"] = args.curriculum_freeze_layers
    if args.curriculum_finetune_lr is not None:
        cfg["curriculum_finetune_lr"] = args.curriculum_finetune_lr
    if args.curriculum_finetune_alpha is not None:
        cfg["curriculum_finetune_alpha"] = args.curriculum_finetune_alpha
    if args.curriculum_dataset_col is not None:
        cfg["curriculum_dataset_col"] = args.curriculum_dataset_col
    if args.no_curriculum_test:
        cfg["curriculum_run_test"] = False

    if args.mode in ("debug", "debug_curriculum"):
        cfg["debug_mode"]  = True
        cfg["num_workers"] = 0

    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            cfg.update(json.load(f))
        print(f"  Config overrides loaded from {args.config}")

    if args.test_csv:
        cfg["test_csv_path"] = args.test_csv
    if args.resume:
        cfg["resume"] = True
    if args.save_all_test_gradcam:
        cfg["save_all_test_gradcam"] = True

    return cfg


def main(override_cfg=None):
    in_notebook = "ipykernel" in sys.modules and len(sys.argv) <= 1

    if in_notebook:
        cfg = copy.deepcopy(CFG)
        if override_cfg: cfg.update(override_cfg)
        if cfg.get("debug_mode", False): cfg["num_workers"] = 0
        if cfg.get("curriculum_run", False):
            mode = "debug_curriculum" if cfg.get("debug_mode") else "curriculum"
        elif cfg.get("quick_mode", False):
            mode = "quick"
        else:
            mode = "debug" if cfg.get("debug_mode") else "train"
    else:
        args = parse_args()
        cfg  = apply_args_to_cfg(args, CFG)
        mode = args.mode

    set_seed(cfg["seed"])

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        print("[WARN] CUDA_VISIBLE_DEVICES not set — defaulting to GPU 0.")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        print(f"[INFO] CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n[INFO] Device        : {device}")
    print(f"[INFO] Mode          : {mode}")
    print(f"[INFO] Arch          : {cfg['arch']}  (image_size={cfg['image_size']})")
    print(f"[INFO] Data split    : {cfg.get('data_split_variant','standard')}  "
          f"(tmpl: {_fold_csv_tmpl_from_cfg(cfg)})")
    print(f"[INFO] ITA fusion    : {cfg['use_ita_fusion']}")
    print(f"[INFO] Soft labels   : {cfg['use_soft_labels']}  "
          f"(triangular α={cfg['soft_label_alpha']})")
    print(f"[INFO] Provenance wt : {cfg.get('use_provenance_weighting', False)}")
    print(f"[INFO] Adaptive α    : {cfg.get('use_adaptive_smoothing', False)}")
    print(f"[INFO] Mixup         : {cfg.get('use_mixup', False)}")
    print(f"[INFO] Exposure aug  : {cfg.get('use_exposure_augment', False)}"
          + (f"  (prob={cfg.get('exposure_aug_prob', 0.3)})"
             if cfg.get('use_exposure_augment', False) else ""))
    print(f"[INFO] LR / WD       : {cfg['learning_rate']} / {cfg['weight_decay']}")
    print(f"[INFO] Patience      : {cfg['patience']}  min_delta={cfg['min_delta']}")
    print(f"[INFO] MC dropout    : rate={cfg.get('mc_dropout_rate',0.4)}")
    if cfg.get("quick_mode", False):
        print(f"[INFO] Quick mode    : fold={cfg['quick_fold']}  "
              f"mc_passes={cfg.get('quick_mc_dropout_passes',5)}  "
              f"TTA=OFF  final-retrain=OFF")
    print(f"[INFO] Exp name      : {build_experiment_name(cfg)}\n")

    if mode == "optuna":
        run_optuna(cfg, device)
        return None

    dirs = make_dirs(cfg, mode=mode)
    print(f"[INFO] Output root   : {dirs['base']}\n")

    t_total = time.time()

    if mode in ("curriculum", "debug_curriculum", "quick_curriculum"):
        print(f"[INFO] Curriculum  : Stage 1 = {cfg['curriculum_pretrain_datasets']}")
        print(f"[INFO] Curriculum  : Stage 2 = {cfg['curriculum_finetune_datasets']}")
        print(f"[INFO] Freeze layers: {cfg['curriculum_freeze_layers']}")
        print(f"[INFO] Finetune LR : {cfg['curriculum_finetune_lr']}\n")

        finetune_fold_metrics, best_s2_epochs, cv_timing, s2_normalizers = \
            run_curriculum_cv(cfg, dirs, device)

        run_test = (
            cfg.get("curriculum_run_test", True)
            and not cfg.get("debug_mode", False)
            and not cfg.get("quick_mode", False)
        )
        if run_test:
            run_curriculum_held_out_test(
                cfg, dirs, device,
                best_s2_epochs, finetune_fold_metrics, cv_timing)
        elif cfg.get("quick_mode", False):
            print("\n  [QUICK] Skipping final held-out retrain.")
    else:
        all_fold_metrics, best_epochs, cv_timing, fold_normalizers = run_cv(
            cfg, dirs, device)

        run_test = (
            not cfg.get("debug_mode", False)
            and not cfg.get("quick_mode", False)
        )
        if run_test:
            run_held_out_test(cfg, dirs, device,
                              best_epochs, all_fold_metrics,
                              cv_timing, fold_normalizers)
        elif cfg.get("quick_mode", False):
            print("\n  [QUICK] Skipping final held-out retrain.")

    print(f"\n[DONE] Total wall time: {(time.time()-t_total)/60:.1f} min\n"
          f"       Outputs → {dirs['base']}")
    return dirs


if __name__ == "__main__":
    main()
