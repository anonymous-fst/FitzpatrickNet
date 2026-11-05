#!/usr/bin/env python3
"""
uncertainty_significance_test.py
==================================
Mann-Whitney U test comparing predictive entropy between correctly and
incorrectly classified samples, stratified by Fitzpatrick Skin Type class.

Usage
-----
    python uncertainty_significance_test.py \
        --data_path  results/seed42/metrics/uncertainty_ensemble_all_folds.csv

The CSV must contain columns: true_label, pred_label, predictive_entropy.
If --data_path is not found, the script will attempt to pool per-fold
uncertainty CSVs from --fold_dir instead.
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


def main():
    parser = argparse.ArgumentParser(
        description="Mann-Whitney U test on predictive entropy (correct vs. incorrect)."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help=(
            "Path to an ensemble uncertainty CSV with columns: "
            "true_label, pred_label, predictive_entropy."
        ),
    )
    parser.add_argument(
        "--fold_dir",
        type=str,
        default=None,
        help=(
            "Directory containing per-fold uncertainty CSVs "
            "(uncertainty_fold1.csv … uncertainty_fold5.csv). "
            "Used as a fallback when --data_path is not found."
        ),
    )
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    df = None

    if args.data_path and os.path.exists(args.data_path):
        print(f"Loading ensemble uncertainty data: {args.data_path}")
        df = pd.read_csv(args.data_path)

    elif args.fold_dir:
        print(f"Ensemble CSV not found. Pooling fold CSVs from: {args.fold_dir}")
        fold_files = [
            os.path.join(args.fold_dir, f"uncertainty_fold{i}.csv")
            for i in range(1, 6)
        ]
        df_list = []
        for f in fold_files:
            if os.path.exists(f):
                df_list.append(pd.read_csv(f))
            else:
                raise FileNotFoundError(f"Missing file: {f}")
        df = pd.concat(df_list, ignore_index=True)

    else:
        raise ValueError(
            "Provide either --data_path (ensemble CSV) or --fold_dir "
            "(directory with per-fold CSVs)."
        )

    required_cols = {"true_label", "pred_label", "predictive_entropy"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # ── Run statistical tests per FST class ───────────────────────────────────
    class_names = [f"FST {i+1}" for i in range(6)]
    results     = []

    for i, name in enumerate(class_names):
        class_df    = df[df["true_label"] == i]
        correct_e   = class_df[class_df["pred_label"] == i]["predictive_entropy"].values
        incorrect_e = class_df[class_df["pred_label"] != i]["predictive_entropy"].values

        n_correct   = len(correct_e)
        n_incorrect = len(incorrect_e)
        med_c       = np.median(correct_e)   if n_correct   > 0 else np.nan
        med_i       = np.median(incorrect_e) if n_incorrect > 0 else np.nan

        if n_correct > 0 and n_incorrect > 0:
            stat, p_val = mannwhitneyu(
                correct_e, incorrect_e, alternative='two-sided'
            )
        else:
            stat, p_val = np.nan, np.nan

        results.append({
            "Class":            name,
            "N_Correct":        n_correct,
            "N_Incorrect":      n_incorrect,
            "Median_Correct":   med_c,
            "Median_Incorrect": med_i,
            "U_Statistic":      stat,
            "P_Value":          p_val,
            "Significant":      (p_val < 0.05) if not np.isnan(p_val) else False,
        })

    # ── Display results ───────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)

    print("\n" + "=" * 85)
    print(
        " MANN-WHITNEY U TEST: "
        "Predictive Entropy (Correct vs. Incorrect Predictions)"
    )
    print("=" * 85)

    formatters = {
        "Median_Correct":   "{:.4f}".format,
        "Median_Incorrect": "{:.4f}".format,
        "P_Value":          lambda x: f"{x:.2e}" if pd.notnull(x) else "NaN",
        "U_Statistic":      lambda x: f"{x:.1f}" if pd.notnull(x) else "NaN",
    }

    print(results_df.to_string(index=False, formatters=formatters))
    print("\n* P-Value < 0.05 is considered statistically significant.")
    print("* 'NaN' means a test could not be performed (e.g., zero incorrect predictions).")


if __name__ == "__main__":
    main()
