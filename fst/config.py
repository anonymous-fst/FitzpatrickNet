"""
fst.config
==========
Data-split variant registry and the two helper functions that derive
fold-CSV template strings and test-CSV paths from a cfg dict.

Nothing in this module imports from any other fst submodule, so it is
safe to import first and avoids circular dependencies.
"""

import os

# ── Data-split variant registry ───────────────────────────────────────────────
# Each entry maps a variant name to a (fold_template, test_filename) pair.
# The fold template must contain exactly one ``{fold}`` placeholder.
DATA_SPLIT_VARIANTS: dict[str, tuple[str, str]] = {
    "standard": (
        "train_val_dataset_metadata_extended_fold_{fold}.csv",
        "test_dataset_metadata_extended.csv",
    ),
    "balanced": (
        "train_val_dataset_metadata_extended_balanced_fold_{fold}.csv",
        "test_dataset_metadata_extended_balanced.csv",
    ),
    "balanced_full_pretrain": (
        "train_val_dataset_metadata_extended_balanced_full_pretrain_retained_fold_{fold}.csv",
        "test_dataset_metadata_extended_balanced_full_pretrain_retained.csv",
    ),
    "filtered": (
        "filtered_fold_{fold}.csv",
        "filtered_test_dataset_metadata_extended.csv",
    ),
}


def fold_csv_template(cfg: dict) -> str:
    """Return the fold-CSV filename template for the variant named in *cfg*."""
    variant = cfg.get("data_split_variant", "standard")
    if variant not in DATA_SPLIT_VARIANTS:
        raise ValueError(
            f"Unknown data_split_variant '{variant}'. "
            f"Valid options: {list(DATA_SPLIT_VARIANTS.keys())}"
        )
    return DATA_SPLIT_VARIANTS[variant][0]


def test_csv_path(cfg: dict) -> str:
    """Return the absolute test-CSV path for the variant named in *cfg*."""
    variant = cfg.get("data_split_variant", "standard")
    if variant not in DATA_SPLIT_VARIANTS:
        raise ValueError(
            f"Unknown data_split_variant '{variant}'. "
            f"Valid options: {list(DATA_SPLIT_VARIANTS.keys())}"
        )
    filename = DATA_SPLIT_VARIANTS[variant][1]
    return os.path.join(cfg["fold_csv_dir"], filename)
