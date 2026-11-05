#!/usr/bin/env python3
"""
ita_only_classification.py
==========================
Evaluate ITA-threshold-based Fitzpatrick Skin Type (FST) classification
against ground-truth labels on an internal test split and an external
held-out dataset.

Usage
-----
    python ita_only_classification.py \
        --internal_csv  /path/to/test_split.csv \
        --external_csv  /path/to/all_metadata.csv \
        --external_dataset EH_Web \
        --output_dir    results/ita/

The external CSV must contain a 'dataset' column; only rows whose
'dataset' value matches --external_dataset are evaluated.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

# ── ITA → FST mapping ─────────────────────────────────────────────────────────

# Historical Fitzpatrick thresholds (Chardon et al.)
HISTORICAL_THRESHOLDS = [55, 41, 28, 10, -30]

PREDICTION_COLS = [
    "ITA_value_spectral_spatial"
]


def ita_to_fitzpatrick(ita_value):
    """
    Map a scalar ITA value to Fitzpatrick Skin Type (1–6).

    Thresholds
    ----------
    FST I   : ITA > 55
    FST II  : 41 < ITA ≤ 55
    FST III : 28 < ITA ≤ 41
    FST IV  : 10 < ITA ≤ 28
    FST V   : -30 < ITA ≤ 10
    FST VI  : ITA ≤ -30
    """
    if pd.isna(ita_value):
        return np.nan

    if ita_value > HISTORICAL_THRESHOLDS[0]:
        return 1
    elif ita_value > HISTORICAL_THRESHOLDS[1]:
        return 2
    elif ita_value > HISTORICAL_THRESHOLDS[2]:
        return 3
    elif ita_value > HISTORICAL_THRESHOLDS[3]:
        return 4
    elif ita_value > HISTORICAL_THRESHOLDS[4]:
        return 5
    else:
        return 6


def compute_metrics(df, col):
    """
    Compute exact accuracy and off-by-one accuracy for a given ITA column.

    Returns
    -------
    (exact_acc, obo_acc, n_valid, n_total)
    """
    if col not in df.columns:
        return np.nan, np.nan, 0, len(df)

    predictions  = df[col].apply(ita_to_fitzpatrick).values
    ground_truth = df["fitzpatrick"].values

    valid_mask  = ~np.isnan(predictions)
    valid_count = valid_mask.sum()

    if valid_count == 0:
        return np.nan, np.nan, 0, len(df)

    exact_acc = (
        (predictions[valid_mask] == ground_truth[valid_mask]).sum() / valid_count
    )
    obo_acc = (
        (np.abs(predictions[valid_mask] - ground_truth[valid_mask]) <= 1).sum()
        / valid_count
    )

    return exact_acc, obo_acc, valid_count, len(df)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ITA-threshold FST classification evaluation."
    )
    parser.add_argument(
        "--internal_csv",
        required=True,
        help="Path to the internal test-split CSV (must contain 'fitzpatrick' column)."
    )
    parser.add_argument(
        "--external_csv",
        required=True,
        help="Path to the full metadata CSV containing multiple datasets."
    )
    parser.add_argument(
        "--external_dataset",
        default="EH_Web",
        help="Value of the 'dataset' column to use as the external validation set."
    )
    parser.add_argument(
        "--output_dir",
        default="results/ita",
        help="Directory where accuracy_results.csv will be written."
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load internal dataset ─────────────────────────────────────────────────
    print(f"[*] Loading internal dataset from {args.internal_csv}...")
    df_int = pd.read_csv(args.internal_csv, low_memory=False)
    print(f"    Loaded {len(df_int)} internal rows")

    # ── Load external dataset ─────────────────────────────────────────────────
    print(f"[*] Loading external dataset from {args.external_csv}...")
    df_ext_full = pd.read_csv(args.external_csv, low_memory=False)
    if "dataset" not in df_ext_full.columns:
        raise ValueError("Column 'dataset' not found in external CSV.")
    df_ext = df_ext_full[
        df_ext_full["dataset"].str.strip().str.lower()
        == args.external_dataset.lower()
    ].copy()
    print(f"    Loaded {len(df_ext)} external rows ({args.external_dataset})")

    # ── Compute metrics ───────────────────────────────────────────────────────
    results = []

    print("\n[*] Processing prediction columns...")
    for col in PREDICTION_COLS:
        int_acc, int_obo, int_valid, int_tot = compute_metrics(df_int, col)
        ext_acc, ext_obo, ext_valid, ext_tot = compute_metrics(df_ext, col)

        results.append({
            "Prediction_Column":    col,
            "Internal_Accuracy":    int_acc,
            "Internal_OBO_Accuracy":int_obo,
            "Internal_Valid_Samples":int_valid,
            "External_Accuracy":    ext_acc,
            "External_OBO_Accuracy":ext_obo,
            "External_Valid_Samples":ext_valid,
        })

        print(f"  - {col}:")
        print(
            f"      Internal: Acc={int_acc:.4f}, OBO={int_obo:.4f} "
            f"(Valid: {int_valid}/{int_tot})"
        )
        print(
            f"      External: Acc={ext_acc:.4f}, OBO={ext_obo:.4f} "
            f"(Valid: {ext_valid}/{ext_tot})"
        )

    # ── Save and summarise ────────────────────────────────────────────────────
    results_df  = pd.DataFrame(results)
    output_path = output_dir / "accuracy_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n[+] Results successfully saved to {output_path}")

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    best_int_col = results_df.loc[
        results_df['Internal_Accuracy'].idxmax(), 'Prediction_Column'
    ]
    best_ext_col = results_df.loc[
        results_df['External_Accuracy'].idxmax(), 'Prediction_Column'
    ]

    print(
        f"Best internal exact accuracy   : "
        f"{results_df['Internal_Accuracy'].max():.4f} ({best_int_col})"
    )
    print(
        f"Best internal off-by-one       : "
        f"{results_df['Internal_OBO_Accuracy'].max():.4f}"
    )
    print(
        f"Mean internal accuracy         : "
        f"{results_df['Internal_Accuracy'].mean():.4f}"
    )
    print("-" * 80)
    print(
        f"Best external exact accuracy   : "
        f"{results_df['External_Accuracy'].max():.4f} ({best_ext_col})"
    )
    print(
        f"Best external off-by-one       : "
        f"{results_df['External_OBO_Accuracy'].max():.4f}"
    )
    print(
        f"Mean external accuracy         : "
        f"{results_df['External_Accuracy'].mean():.4f}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
