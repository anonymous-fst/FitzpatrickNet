#!/usr/bin/env python3
"""
external_validation.py
======================================
Generalised external validation pipeline for all multimodal ViT and CNN
models on a held-out external dataset.

Scans RESULTS_DIRS, discovers experiment groups across seeds, performs
fold inference, ensemble logit averaging per seed, cross-seed aggregation,
and saves comprehensive summaries into a centralised output directory.

Usage
-----
    python external_validation.py \
        --results_dirs   results/multimodal_vit_ita results/multimodal_cnn_ita \
        --metadata_csv   metadata/all_dataset_metadata_extended.csv \
        --base_output_dir results/external_validation \
        --target_dataset  EH_Web \
        --seeds           42 43 44 45 46
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import train_eval_vit_ita as vit_te
import train_eval_cnn_ita as cnn_te

# ── Architecture routing ──────────────────────────────────────────────────────

_VIT_ARCHS = {"vit_small", "eva02_small"}
_CNN_ARCHS = {"resnet18", "efficientnet_b0"}


def is_cnn_experiment(cfg: dict) -> bool:
    return cfg.get("arch", "").lower() in _CNN_ARCHS


def get_te_module(cfg: dict):
    arch = cfg.get("arch", "").lower()
    if arch in _VIT_ARCHS:
        return vit_te
    if arch in _CNN_ARCHS:
        return cnn_te
    raise ValueError(f"Unknown architecture '{arch}' in config.")


# ── CNN ITA normaliser loader ─────────────────────────────────────────────────

def load_cnn_ita_normalizer(model_dir: str, fold: int):
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
        return norm, fallbacks[0]

    return cnn_te.ITANormalizer(), None


# ── Directory / config utilities ──────────────────────────────────────────────

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


def extract_base_and_seed(dir_name: str):
    """
    Extract base experiment name and seed number from a directory name.
    Expects a suffix of the form '_seed<N>' or '_seed<N>_'.
    """
    match = re.search(r'_seed(\d+)(?:_|$)', dir_name)
    if match:
        seed      = int(match.group(1))
        base_name = dir_name[:match.start()]
        return base_name, seed
    return None, None


def find_all_experiment_groups(results_dirs: list, seeds: list) -> dict:
    """
    Scan results_dirs and group directories by base name across all required seeds.
    Returns {base_name: {seed: full_path}} for complete groups only.
    """
    groups: dict = defaultdict(dict)

    for r_dir in results_dirs:
        if not os.path.isdir(r_dir):
            continue
        for d in os.listdir(r_dir):
            if not os.path.isdir(os.path.join(r_dir, d)):
                continue
            base_name, seed = extract_base_and_seed(d)
            if base_name and seed in seeds:
                groups[base_name][seed] = os.path.join(r_dir, d)

    return {
        base: seed_dict
        for base, seed_dict in groups.items()
        if all(s in seed_dict for s in seeds)
    }


def load_and_filter_metadata(metadata_csv: str, target_dataset: str) -> pd.DataFrame:
    print(f"  Loading metadata from {metadata_csv}...")
    df = pd.read_csv(metadata_csv, low_memory=False)
    print(f"  Total samples: {len(df)}")

    if "dataset" not in df.columns:
        raise ValueError("'dataset' column not found in metadata CSV")

    df_filtered = df[
        df["dataset"].str.strip().str.lower() == target_dataset.lower()
    ].copy()
    print(f"  Samples in '{target_dataset}': {len(df_filtered)}")

    if len(df_filtered) == 0:
        raise ValueError(f"No samples found for dataset '{target_dataset}'")

    if "fitzpatrick" not in df_filtered.columns:
        raise ValueError("'fitzpatrick' column not found in metadata CSV")

    df_filtered["label"] = df_filtered["fitzpatrick"].astype(int) - 1
    return df_filtered


# ── Prediction generation ─────────────────────────────────────────────────────

def _generate_fold_predictions_vit(model_dir, fold, test_df, cfg, device):
    model_path = get_best_fold_model(model_dir, fold)
    if not model_path:
        return None, None, None

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


def _generate_fold_predictions_cnn(model_dir, fold, test_df, cfg, device):
    model_path = get_best_fold_model(model_dir, fold)
    if not model_path:
        return None, None, None

    model = cnn_te.build_model(cfg, device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    normalizer, _ = load_cnn_ita_normalizer(model_dir, fold)
    test_loader   = cnn_te.build_test_dataloader(test_df, cfg, normalizer)

    fold_probs, fold_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            imgs, _, hard_lbls, ita, has_ita, _, _, _ = batch
            logits = model(imgs.to(device), ita.to(device), has_ita.to(device))
            fold_probs.append(F.softmax(logits, dim=1).cpu().numpy())
            fold_labels.extend(hard_lbls.cpu().numpy())

    fold_probs  = np.concatenate(fold_probs, axis=0)
    fold_labels = np.array(fold_labels)
    return fold_probs.argmax(axis=1), fold_probs, fold_labels


def generate_fold_predictions(model_dir, fold, test_df, cfg, device):
    if is_cnn_experiment(cfg):
        return _generate_fold_predictions_cnn(model_dir, fold, test_df, cfg, device)
    return _generate_fold_predictions_vit(model_dir, fold, test_df, cfg, device)


def generate_predictions_for_seed(seed_model_dir, test_df, cfg, device):
    all_fold_probs: list  = []
    all_labels            = None
    fold_preds_dict: dict = {}

    for fold in range(1, 6):
        preds, probs, labels = generate_fold_predictions(
            seed_model_dir, fold, test_df, cfg, device
        )
        if probs is None:
            return None, None, None
        all_fold_probs.append(probs)
        all_labels = labels
        fold_preds_dict[fold] = {
            "predictions":   preds,
            "probabilities": probs,
            "labels":        labels,
        }
    return all_fold_probs, all_labels, fold_preds_dict


def create_ensemble_predictions(all_fold_probs):
    if not all_fold_probs:
        return None, None
    avg_probs = np.mean(all_fold_probs, axis=0)
    return avg_probs.argmax(axis=1), avg_probs


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_extended_metrics(labels, preds, probs, num_classes):
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, roc_auc_score, cohen_kappa_score,
    )
    from scipy.stats import spearmanr

    metrics = {
        "accuracy":  accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall":    recall_score(labels, preds, average="macro", zero_division=0),
        "f1":        f1_score(labels, preds, average="macro", zero_division=0),
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
        metrics["auroc"] = roc_auc_score(
            labels, probs, multi_class="ovr", average="macro"
        )
    except Exception:
        metrics["auroc"] = np.nan

    metrics["off_by_one_accuracy"] = float((np.abs(labels - preds) <= 1).mean())

    try:
        rho, _ = spearmanr(labels, preds)
        metrics["spearman_rho"] = float(rho)
    except Exception:
        metrics["spearman_rho"] = np.nan

    try:
        metrics["qwk"] = float(cohen_kappa_score(labels, preds, weights="quadratic"))
    except Exception:
        metrics["qwk"] = np.nan

    return metrics


# ── Per-experiment processing ─────────────────────────────────────────────────

def process_experiment_group(
    base_name, seed_dirs_map, test_df, device, base_output_dir, seeds
):
    print(f"\n{'='*70}")
    print(f"  PROCESSING EXPERIMENT: {base_name}")
    print(f"{'='*70}")

    exp_output_dir = os.path.join(base_output_dir, base_name)
    seed_results   = {}

    for seed in seeds:
        seed_model_dir = seed_dirs_map[seed]
        print(f"  -> Seed {seed}...")

        models_dir = os.path.join(seed_model_dir, "models")
        if not os.path.isdir(models_dir):
            print(f"     [SKIP] models/ directory missing.")
            return None

        try:
            cfg = load_config(seed_model_dir)
        except Exception as e:
            print(f"     [ERROR] Config load failed: {e}")
            return None

        all_fold_probs, all_labels, fold_preds_dict = generate_predictions_for_seed(
            seed_model_dir, test_df, cfg, device
        )

        if all_fold_probs is None:
            print(f"     [SKIP] Missing model checkpoints for one or more folds.")
            return None

        ensemble_preds, ensemble_probs = create_ensemble_predictions(all_fold_probs)
        class_names = get_te_module(cfg).CLASS_NAMES

        fold_metrics_list = []
        for fold, fold_data in sorted(fold_preds_dict.items()):
            fm = compute_extended_metrics(
                fold_data["labels"], fold_data["predictions"],
                fold_data["probabilities"], cfg["num_classes"]
            )
            fm["fold"] = fold
            fold_metrics_list.append(fm)

        ensemble_metrics = compute_extended_metrics(
            all_labels, ensemble_preds, ensemble_probs, cfg["num_classes"]
        )

        seed_out  = os.path.join(exp_output_dir, f"seed{seed}")
        metrics_d = os.path.join(seed_out, "metrics")
        os.makedirs(metrics_d, exist_ok=True)

        pred_df = test_df.copy().reset_index(drop=True)
        pred_df["prediction"] = ensemble_preds
        pred_df["true_label"] = all_labels
        for i in range(cfg["num_classes"]):
            pred_df[f"prob_{class_names[i]}"] = ensemble_probs[:, i]
        pred_df.to_csv(
            os.path.join(metrics_d, "test_predictions_all_folds_ensemble.csv"),
            index=False
        )

        mean_fold_acc = float(np.mean([m["accuracy"] for m in fold_metrics_list]))
        improvement   = float(ensemble_metrics["accuracy"] - mean_fold_acc)
        summary = {
            "seed":                 seed,
            "individual_fold_metrics": fold_metrics_list,
            "ensemble_metrics":     ensemble_metrics,
            "mean_fold_accuracy":   mean_fold_acc,
            "improvement_accuracy": improvement,
            "improvement_pct":      (
                improvement / mean_fold_acc * 100 if mean_fold_acc > 0 else 0.0
            ),
        }

        with open(os.path.join(metrics_d, "ensemble_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        seed_results[seed] = {
            "ensemble_metrics": ensemble_metrics,
            "summary":          summary,
            "labels":           all_labels,
        }
        print(f"     [OK] Seed {seed} complete.")

    agg_df = aggregate_and_save_group_results(base_name, exp_output_dir, seed_results)
    return agg_df


def aggregate_and_save_group_results(base_name, exp_output_dir, seed_results):
    print(f"\n  --- Aggregating across seeds for {base_name} ---")

    aggregated  = defaultdict(list)
    improvements = defaultdict(list)

    for seed, res in seed_results.items():
        for k, v in res["ensemble_metrics"].items():
            if not np.isnan(v):
                aggregated[k].append(v)
        for k, v in res["summary"].items():
            if k in ["mean_fold_accuracy", "improvement_accuracy", "improvement_pct"]:
                if not np.isnan(float(v)):
                    improvements[k].append(float(v))

    metrics_dir = os.path.join(exp_output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    agg_records = [
        {
            "metric":  metric,
            "mean":    float(np.mean(values)),
            "std":     float(np.std(values)),
            "n_seeds": len(values),
        }
        for metric, values in aggregated.items()
        if values
    ]
    agg_df = pd.DataFrame(agg_records)
    agg_df.to_csv(os.path.join(metrics_dir, "aggregated_ensemble_metrics.csv"),
                  index=False)

    master_summary = {
        "experiment":         base_name,
        "n_seeds":            len(seed_results),
        "aggregated_metrics": {
            r["metric"]: {"mean": r["mean"], "std": r["std"]}
            for r in agg_records
        },
    }
    with open(os.path.join(metrics_dir, "external_validation_summary.json"), "w") as f:
        json.dump(master_summary, f, indent=2)

    print(f"  [OK] Saved group aggregation to {metrics_dir}")
    return agg_df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch external validation for all multimodal ViT/CNN models."
    )
    parser.add_argument(
        "--results_dirs",
        nargs="+",
        required=True,
        help="One or more directories containing seed-specific experiment folders."
    )
    parser.add_argument(
        "--metadata_csv",
        required=True,
        help="Path to the full metadata CSV (must contain 'dataset' and 'fitzpatrick')."
    )
    parser.add_argument(
        "--base_output_dir",
        required=True,
        help="Root directory where per-experiment validation results will be written."
    )
    parser.add_argument(
        "--target_dataset",
        default="EH_Web",
        help="Value of the 'dataset' column to use as the external validation set."
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45, 46],
        help="Seed values to look for (default: 42 43 44 45 46)."
    )
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  BATCH EXTERNAL VALIDATION — {args.target_dataset} Dataset")
    print(f"{'='*70}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    test_df = load_and_filter_metadata(args.metadata_csv, args.target_dataset)

    complete_groups = find_all_experiment_groups(args.results_dirs, args.seeds)
    if not complete_groups:
        print("  [ERROR] No complete experiment groups found across RESULTS_DIRS.")
        return

    print(f"  Found {len(complete_groups)} complete experiment group(s) to validate:\n")
    for b_name in sorted(complete_groups):
        print(f"    • {b_name}")

    global_summaries = []
    for base_name in sorted(complete_groups):
        seed_dirs_map = complete_groups[base_name]
        agg_df = process_experiment_group(
            base_name, seed_dirs_map, test_df, device,
            args.base_output_dir, args.seeds
        )

        if agg_df is not None:
            acc_row  = agg_df[agg_df["metric"] == "accuracy"]
            obo_row  = agg_df[agg_df["metric"] == "off_by_one_accuracy"]
            mean_acc = acc_row["mean"].values[0]  if not acc_row.empty  else np.nan
            std_acc  = acc_row["std"].values[0]   if not acc_row.empty  else np.nan
            mean_obo = obo_row["mean"].values[0]  if not obo_row.empty  else np.nan
            std_obo  = obo_row["std"].values[0]   if not obo_row.empty  else np.nan

            global_summaries.append({
                "experiment":              base_name,
                "mean_accuracy":           mean_acc,
                "std_accuracy":            std_acc,
                "mean_off_by_one_accuracy":mean_obo,
                "std_off_by_one_accuracy": std_obo,
            })

    os.makedirs(args.base_output_dir, exist_ok=True)
    master_csv = os.path.join(
        args.base_output_dir, "all_models_external_validation_summary.csv"
    )
    pd.DataFrame(global_summaries).to_csv(master_csv, index=False)

    print(f"\n{'='*70}")
    print(f"  BATCH EXTERNAL VALIDATION COMPLETE")
    print(f"  Master summary saved to: {master_csv}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
