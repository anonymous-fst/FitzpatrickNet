# FitzpatrickNet: Multimodal Classification of Fitzpatrick Skin Types from Clinical Images

This repository contains code for the paper *"FitzpatrickNet: Multimodal Classification of Fitzpatrick Skin Types from Clinical Images"*

Author information is withheld for anonymous peer review.  

The repository will be made public upon acceptance.

---

## Overview

This repository provides a complete, reproducible pipeline for automatic
**Fitzpatrick Skin Type (FST) classification** (six ordinal classes, FST I–VI)
from clinical dermatological images. The core contribution is a **multimodal
late-fusion architecture** that combines deep image features with a scalar
**Individual Typology Angle (ITA)** measurement — a colourimetric proxy for
skin pigmentation computed directly from the image.

The pipeline covers every stage from raw metadata to final evaluation:

| Stage | Script(s) |
|---|---|
| ITA computation | `ita_spatial_isolation_spectral_spatial.py` |
| Data splitting | `create_data_splits.py` |
| Model training & CV | `train_eval_vit_ita.py`, `train_eval_cnn_ita.py` |
| Fold-level ensemble | `ensemble_fold_averaging.py` |
| Seed-level ensemble | `ensemble_logit_averaging.py` |
| External validation | `external_validation.py` |
| ITA-only baseline | `ita_only_classification.py` |
| Uncertainty analysis | `uncertainty_significance_test.py` |

---

## Architectures

### Vision Transformer models (`train_eval_vit_ita.py`)

| Architecture | Timm ID | Input size |
|---|---|---|
| ViT-Small | `vit_small_patch16_224.augreg_in21k_ft_in1k` | 224 × 224 |
| EVA-02-Small | `eva02_small_patch14_336.mim_in22k_ft_in1k` | 336 × 336 |

Three ITA fusion strategies are supported via `--ita_fusion_mode`:

- **`late_concat`** *(default)*: image CLS token concatenated with a 2-layer ITA MLP projection
- **`cross_attention`**: image features attend to ITA embedding via multi-head cross-attention
- **`adaptive_gating`**: per-dimension gating network learns the contribution of ITA features

### CNN models (`train_eval_cnn_ita.py`)

| Architecture | Backbone | Feature dim |
|---|---|---|
| ResNet-18 | `torchvision.models.resnet18` | 512 |
| EfficientNet-B0 | `torchvision.models.efficientnet_b0` | 1280 |

Both CNN models use an `ITAProjector` (2-layer MLP with a learnable missing-value token)
and a data-driven `ITANormalizer` fitted on training ITA values per fold.

### Shared training features (both architectures)

- **Focal KL-divergence loss** with optional class-frequency weighting
- **Triangular soft labels** with cosine alpha annealing
- **Adaptive label smoothing** (per-dataset alpha map)
- **Provenance-based loss weighting** (per-dataset scalar)
- **Patient consistency regularisation** (KL penalty across same-patient images)
- **Supervised contrastive loss** (optional)
- **Mixup augmentation** (configurable probability and alpha)
- **Synthetic exposure augmentation** (over/underexposure injection)
- **Weighted random sampler** and **balanced batch sampler**
- **MC-Dropout uncertainty** (predictive entropy + mutual information)
- **Test-time augmentation** (9 photometric/geometric views)
- **Grad-CAM explainability** (patch-token path for ViT; spatial-map path for CNN)
- **Attention rollout** (ViT only)
- **5-fold cross-validation** with greedy size-balanced group assignment
- **Curriculum learning** (two-stage: noisy pre-train → clinical fine-tune)
- **Optuna hyperparameter search** (TPE sampler + median pruner)
- **Bootstrap 95 % confidence intervals** across folds

---

## Repository Structure

```
.
├── fst/                              # Shared internal package
│   ├── __init__.py
│   ├── config.py                     # Data-split variant registry
│   ├── reproducibility.py            # set_seed
│   ├── soft_labels.py                # Triangular / adaptive soft-label generation
│   ├── losses.py                     # FocalKLLoss, patient consistency, contrastive
│   ├── sampling.py                   # BalancedBatchSampler, mixup, provenance weights
│   ├── metrics.py                    # Extended metric suite, bootstrap CIs
│   ├── uncertainty.py                # MC-Dropout prediction, ECE, box-plot
│   ├── plotting.py                   # All visualisation utilities
│   └── explainability.py             # MinimalGradCAM, Grad-CAM save helpers
│
├── train_eval_vit_ita.py             # ViT / EVA-02 training & evaluation
├── train_eval_cnn_ita.py             # ResNet-18 / EfficientNet-B0 training & evaluation
│
├── create_data_splits.py             # Patient-safe FST-balanced data splitting
├── ita_spatial_isolation_spectral_spatial.py  # ITA computation (spectral-spatial)
│
├── ensemble_fold_averaging.py        # Fold-level ensemble (averages across folds)
├── ensemble_logit_averaging.py       # Seed-level ensemble (averages across seeds)
├── external_validation.py            # Batch external validation on held-out dataset
├── ita_only_classification.py        # ITA-threshold FST classification baseline
├── uncertainty_significance_test.py  # Mann-Whitney U test on predictive entropy
│
├── pyproject.toml                    # PEP 517/518 package configuration
├── setup.py                          # Minimal shim for editable installs
└── requirements.txt                  # Pinned dependencies for HPC reproducibility
```

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (strongly recommended; 16 GB VRAM for EVA-02 at batch size 32)

### 1. Clone and create a virtual environment

```bash
git clone <repository-url>
cd <repository-name>
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
```

### 2. Install the package and dependencies

```bash
pip install -e .          # installs fst package in editable mode
pip install -r requirements.txt
```

### 3. Verify GPU availability

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0)}')"
```

### Optional: Optuna hyperparameter search

Uncomment the `optuna` line in `requirements.txt` and reinstall, or:

```bash
pip install optuna==3.5.0
```

---

## Datasets

### Training datasets (internal)

| Dataset | Role | 
|---|---|
| TLA_i4i | Primary (fully included) | 
| PASSION | Top-up (FST-balanced sampling) | 
| PAD-UFES-20 | Top-up (FST-balanced sampling) | 
| Fitzpatrick17k | Top-up (FST-balanced sampling) | 
| SCIN | Top-up (FST-balanced sampling) | 

### External validation dataset

| Dataset | Role |
|---|---|
| EH_Web | Held-out external validation only (excluded from training) |

### Metadata CSV format

Each metadata CSV must contain at minimum:

| Column | Description |
|---|---|
| `image_path_512` | Absolute path to the 512 × 512 image |
| `mask_path_512` | Absolute path to the binary skin mask (optional) |
| `fitzpatrick` | Ground-truth FST label (integer 1–6) |
| `dataset` | Source dataset name |
| `ITA_value_spectral_spatial` | Pre-computed ITA value (float; NaN if unavailable) |
| `patient_id` / `subject_id` / `case_id` / `refno` | Any one patient identifier (for leak-free splitting) |

---

## Usage

### Step 1 — Compute ITA values

```bash
python ita_spatial_isolation_spectral_spatial.py \
    --image_dir   /path/to/images \
    --metadata_csv metadata/dataset_metadata.csv
```

The script appends an `ITA_value_spectral_spatial` column to the CSV in-place.

### Step 2 — Create data splits

```bash
python create_data_splits.py \
    --input              metadata/all_dataset_metadata_extended.csv \
    --output-metadata-dir data_split/ \
    --n-splits           5 \
    --n-trials           200 \
    --seed               42
```

This produces:
- `data_split/train_val_dataset_metadata_extended_balanced_fold_{1..5}.csv`
- `data_split/test_dataset_metadata_extended_balanced.csv`

### Step 3 — Train a model

**ViT-Small (standard training):**
```bash
python train_eval_vit_ita.py \
    --mode        train \
    --arch        vit_small \
    --data_split  balanced \
    --tag         exp1 \
    --epochs      30 \
    --seed        42
```

**EVA-02 with cross-attention ITA fusion:**
```bash
python train_eval_vit_ita.py \
    --mode            train \
    --arch            eva02_small \
    --ita_fusion_mode cross_attention \
    --data_split      balanced \
    --tag             exp1_crossattn \
    --seed            42
```

**ResNet-18 (CNN):**
```bash
python train_eval_cnn_ita.py \
    --mode       train \
    --arch       resnet18 \
    --data_split balanced \
    --tag        exp1 \
    --seed       42
```

**Curriculum learning (two-stage):**
```bash
python train_eval_vit_ita.py \
    --mode                     curriculum \
    --arch                     vit_small \
    --curriculum_freeze_blocks 8 \
    --curriculum_finetune_lr   5e-5 \
    --tag                      exp1_curriculum \
    --seed                     42
```

**Quick single-fold sanity check:**
```bash
python train_eval_vit_ita.py --mode quick --arch vit_small --tag debug --seed 42
```

**Hyperparameter search (Optuna):**
```bash
python train_eval_vit_ita.py \
    --mode         optuna \
    --arch         vit_small \
    --optuna_trials 50 \
    --tag          hpo
```

### Step 4 — Fold-level ensemble evaluation

```bash
python ensemble_fold_averaging.py \
    --results_dirs results/multimodal_vit_ita results/multimodal_cnn_ita \
    --test_csv     data_split/test_dataset_metadata_extended_balanced.csv \
    --seeds        42 43 44 45 46
```

### Step 5 — Seed-level ensemble (from saved prediction CSVs)

```bash
python ensemble_logit_averaging.py \
    --model_dirs \
        results/multimodal_vit_ita/vit_small_exp1_seed42 \
        results/multimodal_vit_ita/vit_small_exp1_seed43 \
        results/multimodal_vit_ita/vit_small_exp1_seed44 \
        results/multimodal_vit_ita/vit_small_exp1_seed45 \
        results/multimodal_vit_ita/vit_small_exp1_seed46 \
    --output_dir results/seed_ensemble
```

### Step 6 — External validation

```bash
python external_validation.py \
    --results_dirs    results/multimodal_vit_ita results/multimodal_cnn_ita \
    --metadata_csv    metadata/all_dataset_metadata_extended.csv \
    --base_output_dir results/external_validation \
    --target_dataset  EH_Web \
    --seeds           42 43 44 45 46
```

### Step 7 — ITA-only baseline

```bash
python ita_only_classification.py \
    --internal_csv     data_split/test_dataset_metadata_extended_balanced.csv \
    --external_csv     metadata/all_dataset_metadata_extended.csv \
    --external_dataset EH_Web \
    --output_dir       results/ita/
```

### Step 8 — Uncertainty significance test

```bash
python uncertainty_significance_test.py \
    --data_path results/multimodal_vit_ita/vit_small_exp1_seed42/metrics/uncertainty_ensemble_all_folds.csv
```

---

## Key CLI Flags

### Common to both training scripts

| Flag | Default | Description |
|---|---|---|
| `--mode` | `train` | `train`, `debug`, `quick`, `optuna`, `curriculum`, `debug_curriculum`, `quick_curriculum` |
| `--arch` | `vit_small` / `resnet18` | Architecture identifier |
| `--data_split` | `standard` | `standard`, `balanced`, `balanced_full_pretrain`, `filtered` |
| `--tag` | `baseline` | Free-text experiment label appended to output directory name |
| `--seed` | `42` | Random seed (also offsets per-fold seeds) |
| `--epochs` | `30` / `50` | Maximum training epochs |
| `--lr` | `5e-5` | AdamW learning rate |
| `--wd` | `3e-4` | AdamW weight decay |
| `--bs` | `32` | Batch size |
| `--patience` | `8` | Early-stopping patience (epochs) |
| `--no_ita_fusion` | — | Disable ITA branch (image-only ablation) |
| `--use_mixup` | — | Enable Mixup augmentation |
| `--use_exposure_augment` | — | Enable synthetic exposure augmentation |
| `--use_provenance_weighting` | — | Enable per-dataset loss scaling |
| `--use_adaptive_smoothing` | — | Enable per-dataset label smoothing alpha |
| `--no_soft_labels` | — | Hard labels only |
| `--no_focal` | — | Disable focal weighting in loss |
| `--no_mc_dropout` | — | Disable MC-Dropout uncertainty |
| `--resume` | — | Resume from last checkpoint |

### ViT-specific

| Flag | Default | Description |
|---|---|---|
| `--ita_fusion_mode` | `late_concat` | `late_concat`, `cross_attention`, `adaptive_gating` |
| `--ita_proj_dim` | `64` | ITA MLP projection dimension |
| `--curriculum_freeze_blocks` | `8` | Transformer blocks frozen during Stage 2 |
| `--save_all_test_gradcam` | — | Save Grad-CAM for every test sample |

### CNN-specific

| Flag | Default | Description |
|---|---|---|
| `--ita_proj_dim` | `8` | ITA projector output dimension |
| `--curriculum_freeze_layers` | `2` | Layer groups frozen during Stage 2 |

---

## Output Structure

Each training run writes to `results/<arch>_<flags>_<tag>/`:

```
results/<experiment_name>/
├── config.json                        # Full configuration snapshot
├── models/
│   ├── best_fold{1..5}_ep{N}.pth      # Best checkpoint per fold
│   ├── ita_normalizer_fold{1..5}.json # CNN: per-fold ITA normaliser (mean/std)
│   └── final_model_ep{N}.pth          # Final model retrained on full train+val
├── plots/
│   ├── loss_all_folds.png
│   ├── accuracy_all_folds.png
│   ├── cm_fold{N}.png / .csv / _normalised.csv
│   ├── roc_fold{N}.png
│   ├── pr_fold{N}.png
│   ├── calibration_fold{N}.png
│   ├── bar_aggregate.png
│   ├── bar_perclass_{metric}.png
│   ├── patient_scatter_{tag}.png
│   └── uncertainty_entropy_boxplot_{tag}.png
├── metrics/
│   ├── per_fold_metrics.csv
│   ├── cv_bootstrap_ci.csv
│   ├── test_predictions.csv
│   ├── test_metrics.csv
│   └── timing.csv
├── gradcam/                           # Grad-CAM panels, heatmaps, overlays
├── attention_rollout/                 # ViT only
└── uncertainty/
    └── uncertainty_{tag}.csv
```

---

## Evaluation Metrics

| Metric | Symbol | Notes |
|---|---|---|
| Accuracy | Acc | Exact FST match |
| Off-by-one accuracy | OBO | \|pred − true\| ≤ 1 |
| Macro F1 | F1 | Unweighted across 6 classes |
| Macro AUROC | AUROC | One-vs-rest |
| Quadratic Weighted Kappa | QWK | Penalises ordinal distance |
| Spearman ρ | ρ | Rank correlation |
| Macro Specificity | Spec | |
| Adjacent Precision | AdjPrec | FST II–V only |
| ECE (top-label) | ECE | Expected calibration error |
| ECE (classwise) | ECE-cw | Per-class OvR calibration |

Bootstrap 95 % confidence intervals are reported for all aggregate metrics
across the 5 cross-validation folds.

---

## Training Configuration

### Optimiser and scheduler

- **Optimiser**: AdamW
- **Scheduler**: Cosine annealing with warm restarts (`T₀ = 5` epochs, `T_mult = 2`)
- **Early stopping**: patience = 8 epochs on validation accuracy; loss gate (≤ 1.05 × best) prevents accuracy-only overfitting

### Soft label scheme

Triangular soft labels assign mass α to the two adjacent FST classes and 1 - 2α to the true class. Alpha is linearly annealed from `alpha_start` (0.20) to `alpha_end` (0.10) over training. Boundary classes receive the neighbour mass on the true class instead.

### Data augmentation (training)

- Random resized crop (scale 0.8–1.0, ratio 0.9–1.1)
- Random horizontal flip
- Random rotation ±15°
- Minority-class shear augmentation (FST I and FST VI)
- ColorJitter (brightness 0.4, contrast 0.4, saturation 0.2, hue 0.05)
- Gamma jitter (0.7–1.4)
- Optional synthetic exposure augmentation (over/underexposure)

---

## Data Availability

The primary clinical dataset (TLA_i4i) contains identifiable patient
information and cannot be made publicly available due to data governance
restrictions. The publicly available datasets used in this study are:

- **PAD-UFES-20**: [https://data.mendeley.com/datasets/zr7vgbcyr2/1](https://data.mendeley.com/datasets/zr7vgbcyr2/1)
- **PASSION**: [https://passionderm.github.io/](https://passionderm.github.io/)
- **fitzpatrick17k**: [https://github.com/mattgroh/fitzpatrick17k](https://github.com/mattgroh/fitzpatrick17k)
- **SCIN**: [https://github.com/google-research-datasets/scin](https://github.com/google-research-datasets/scin)

---

## References

1. Chardon A, Cretois I, Hourseau C. Skin colour typology and suntanning pathways. Int J Cosmet Sci. 1991;13(4):191–208.
2. Dosovitskiy A, et al. An image is worth 16×16 words: Transformers for image recognition at scale. ICLR 2021.
3. Fang Y, et al. EVA-02: A visual representation for neon genesis. arXiv 2023.
4. He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. CVPR 2016.
5. Tan M, Le QV. EfficientNet: Rethinking model scaling for convolutional neural networks. ICML 2019.
6. Lin TY, et al. Focal loss for dense object detection. ICCV 2017.
7. Zhang H, Cissé M, Dauphin YN, Lopez-Paz D. Mixup: Beyond empirical risk minimisation. ICLR 2018.
8. Gal Y, Ghahramani Z. Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. ICML 2016.
9. Abnar S, Zuidema W. Quantifying attention flow in transformers. ACL 2020.
10. Selvaraju RR, et al. Grad-CAM: Visual explanations from deep networks via gradient-based localization. ICCV 2017.
11. Akiba T, Sano S, Yanase T, Ohta T, Koyama M. Optuna: A next-generation hyperparameter optimization framework. KDD 2019.

---

## Acknowledgements

This work uses publicly available datasets and open-source libraries including
PyTorch, timm, scikit-learn, and OpenCV. We thank the authors of all
referenced works for their contributions.
