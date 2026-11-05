"""
create_data_splits.py
=======================
Create reproducible, patient-leak-free test / train / validation splits
with FST-balanced dataset top-up.

Dataset inclusion policy
------------------------
  EXCLUDED  : EH_Web
  FULL      : TLA_i4i  (unconditionally included in full)
  TOP-UP    : PASSION, PAD-UFES-20, fitzpatrick17k, SCIN  (FST-balanced sampling)
"""

import os
import argparse
import random
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm


# ── Constants ─────────────────────────────────────────────────────────────────

PATIENT_COLS = ['patient_id', 'subject_id', 'case_id', 'refno']

EXCLUDED_DATASETS  = {'EH_Web'}
FULL_DATASETS      = {'TLA_i4i'}
TOPUP_DATASETS     = ['PASSION', 'PAD-UFES-20', 'fitzpatrick17k', 'SCIN']


# ── Utilities ─────────────────────────────────────────────────────────────────

def first_nonnull(row, cols):
    for c in cols:
        if c in row and pd.notna(row[c]) and str(row[c]).strip() != '':
            return str(row[c]).strip()
    return None


def make_patient_group(df):
    """
    Build a group label per row.
    Same patient within the same dataset → same group label.
    Rows with no patient ID → guaranteed-unique singleton group.
    Different datasets → always different groups even if ID values match.
    """
    groups = []
    for pos, (idx, r) in enumerate(df.iterrows()):
        pid = first_nonnull(r, PATIENT_COLS)
        if pid is None:
            groups.append(f'__unique__{idx}__{pos}')
        else:
            groups.append(f"{r['dataset']}__{pid}")
    return pd.Series(groups, index=df.index)


# ── FST top-up sampling ───────────────────────────────────────────────────────

def fst_topup_sample(base_df, topup_datasets_dfs, seed=42):
    rng = random.Random(seed)

    tla_rows = base_df[base_df['dataset'] == 'TLA_i4i'].copy()

    if tla_rows.empty or 'fitzpatrick' not in tla_rows.columns:
        print(
            "  [FST top-up] TLA_i4i not found or has no 'fitzpatrick' column. "
            "Skipping top-up."
        )
        return base_df

    fst_counts_tla = (
        tla_rows['fitzpatrick']
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
    )

    all_fst_classes = sorted([1, 2, 3, 4, 5, 6])
    for fst in all_fst_classes:
        if fst not in fst_counts_tla.index:
            fst_counts_tla[fst] = 0
    fst_counts_tla = fst_counts_tla.sort_index()

    max_fst_class = int(fst_counts_tla.idxmax())
    max_fst_count = int(fst_counts_tla.max())

    print(f"\n  [FST top-up] TLA_i4i FST distribution:")
    for fst in all_fst_classes:
        marker = " ← most populous" if fst == max_fst_class else ""
        print(f"    FST{fst}: {fst_counts_tla.get(fst, 0):,}{marker}")
    print(f"\n  [FST top-up] Target count per FST class: {max_fst_count:,}")

    current_fst_counts = fst_counts_tla.reindex(all_fst_classes, fill_value=0).copy()

    print(f"\n  [FST top-up] Initial FST counts (TLA_i4i only):")
    for fst in all_fst_classes:
        deficit = max(0, max_fst_count - current_fst_counts.get(fst, 0))
        print(
            f"    FST{fst}: {current_fst_counts.get(fst, 0):,}  "
            f"(deficit: {deficit:,})"
        )

    selected_topup_frames = []

    for ds_name, ds_df in topup_datasets_dfs.items():
        if ds_df.empty:
            print(f"\n  [FST top-up] '{ds_name}' is empty — skipping.")
            continue

        if 'fitzpatrick' not in ds_df.columns:
            print(
                f"\n  [FST top-up] '{ds_name}' has no 'fitzpatrick' column — "
                f"skipping."
            )
            continue

        ds_df = ds_df.copy()
        ds_df['fitzpatrick'] = pd.to_numeric(ds_df['fitzpatrick'], errors='coerce')

        selected_from_ds = []
        total_selected   = 0

        print(f"\n  [FST top-up] Sampling from '{ds_name}':")

        for fst in all_fst_classes:
            if fst == max_fst_class:
                deficit = max(0, max_fst_count - current_fst_counts.get(fst, 0))
                if deficit == 0:
                    print(
                        f"    FST{fst}: most populous class — "
                        f"no top-up needed (already at target)"
                    )
                    continue

            deficit = max(0, max_fst_count - current_fst_counts.get(fst, 0))
            if deficit == 0:
                print(f"    FST{fst}: already at or above target — skipping")
                continue

            pool = ds_df[ds_df['fitzpatrick'] == fst]
            if pool.empty:
                print(
                    f"    FST{fst}: deficit={deficit:,}  "
                    f"available=0  selected=0"
                )
                continue

            n_to_sample = min(deficit, len(pool))
            sampled = pool.sample(n=n_to_sample, random_state=seed, replace=False)
            selected_from_ds.append(sampled)
            current_fst_counts[fst] = current_fst_counts.get(fst, 0) + n_to_sample
            total_selected += n_to_sample

            print(
                f"    FST{fst}: deficit={deficit:,}  "
                f"available={len(pool):,}  selected={n_to_sample:,}  "
                f"(new total: {current_fst_counts[fst]:,})"
            )

        if selected_from_ds:
            ds_selected_df = pd.concat(selected_from_ds, ignore_index=True)
            selected_topup_frames.append(ds_selected_df)
            print(f"  → '{ds_name}' contributed {total_selected:,} rows")
        else:
            print(f"  → '{ds_name}' contributed 0 rows")

    print(f"\n  [FST top-up] Final FST counts after top-up:")
    for fst in all_fst_classes:
        status = (
            "✓ at target"
            if current_fst_counts.get(fst, 0) >= max_fst_count
            else f"✗ still {max_fst_count - current_fst_counts.get(fst, 0):,} short"
        )
        print(f"    FST{fst}: {current_fst_counts.get(fst, 0):,}  {status}")

    if selected_topup_frames:
        result = pd.concat([base_df] + selected_topup_frames, ignore_index=True)
    else:
        result = base_df.reset_index(drop=True)

    print(
        f"\n  [FST top-up] Total rows after top-up: {len(result):,} "
        f"(base: {len(base_df):,} + top-up: {len(result) - len(base_df):,})"
    )
    return result


# ── Leakage audit ─────────────────────────────────────────────────────────────

def audit_patient_leakage(df, groups, verbose=True):
    """
    Verify no real patient identity is split across multiple group labels.
    Returns n_conflicts (0 = clean).
    """
    conflicts = []

    for ds in df['dataset'].dropna().unique():
        ds_mask = df['dataset'] == ds
        ds_df   = df[ds_mask]
        ds_grps = groups[ds_mask]

        for col in PATIENT_COLS:
            if col not in ds_df.columns:
                continue
            for val, sub in ds_df.groupby(
                ds_df[col].fillna('__NA__').astype(str)
            ):
                if val == '__NA__':
                    continue
                unique_group_labels = ds_grps.loc[sub.index].unique()
                if len(unique_group_labels) > 1:
                    conflicts.append({
                        'dataset':      ds,
                        'id_column':    col,
                        'id_value':     val,
                        'n_rows':       len(sub),
                        'group_labels': list(unique_group_labels),
                    })

    n_conflicts = len(conflicts)
    if n_conflicts == 0:
        print("  [Leakage audit] PASSED — no patient identity conflicts.")
    else:
        print(f"  [Leakage audit] WARNING — {n_conflicts} conflict(s):")
        for c in conflicts:
            print(
                f"    dataset={c['dataset']}  col={c['id_column']}  "
                f"value={c['id_value']}  n_rows={c['n_rows']}"
            )
        print("  ACTION: Harmonise patient ID columns before splitting.")
    return n_conflicts


# ── Stratification mode detection ─────────────────────────────────────────────

def make_stratify_key(df, mode):
    """
    Build a per-row stratification key string.

    mode='composite'  : dataset + fitzpatrick
    mode='fitzpatrick': fitzpatrick only
    mode='none'       : constant (pure size balance)
    """
    if mode == 'composite':
        if 'fitzpatrick' in df.columns and 'dataset' in df.columns:
            return (
                df['dataset'].fillna('unknown').astype(str)
                + '__FST'
                + df['fitzpatrick'].fillna(-1).astype(int).astype(str)
            )
        elif 'dataset' in df.columns:
            return df['dataset'].fillna('unknown').astype(str)
        else:
            return pd.Series(['all'] * len(df), index=df.index)

    elif mode == 'fitzpatrick':
        if 'fitzpatrick' in df.columns:
            return 'FST' + df['fitzpatrick'].fillna(-1).astype(int).astype(str)
        else:
            return pd.Series(['all'] * len(df), index=df.index)

    else:  # 'none'
        return pd.Series(['all'] * len(df), index=df.index)


def detect_stratify_mode(df, n_splits, min_per_stratum=2):
    """
    Select the most informative stratification mode that is feasible.
    A mode is feasible when every stratum has >= min_per_stratum * n_splits samples.
    """
    threshold = min_per_stratum * n_splits
    for mode in ('composite', 'fitzpatrick', 'none'):
        key    = make_stratify_key(df, mode)
        counts = key.value_counts()
        if (counts >= threshold).all():
            print(f"  Stratification mode: '{mode}'")
            if mode != 'composite':
                print(
                    f"  [INFO] Composite infeasible "
                    f"(some strata < {threshold} samples). "
                    f"Falling back to '{mode}'."
                )
            return mode
    return 'none'


# ── Distribution scoring ──────────────────────────────────────────────────────

def distribution_score(full_df, candidate_df, mode):
    """
    Sum of absolute differences in normalised value-count proportions
    for the active stratification key. Lower = more representative.
    Returns inf for empty candidates.
    """
    if candidate_df.empty:
        return float('inf')

    key_full = make_stratify_key(full_df, mode)
    key_cand = make_stratify_key(candidate_df, mode)
    full_cnt = key_full.value_counts(normalize=True)
    cand_cnt = key_cand.value_counts(normalize=True)

    return sum(abs(v - cand_cnt.get(k, 0.0)) for k, v in full_cnt.items())


# ── Debug-aware stratified subsampling ───────────────────────────────────────

def stratified_group_subsample(df, target_n, seed=42):
    """
    Subsample ~target_n rows while:
      1. Preserving dataset × FST proportions as closely as possible.
      2. Never splitting a patient group.
      3. Ensuring every dataset present in the full data is represented.
    """
    groups_all = make_patient_group(df)

    group_meta = {}
    for g, idxs in groups_all.groupby(groups_all).groups.items():
        sub  = df.loc[idxs]
        size = len(idxs)
        ds   = sub['dataset'].mode().iloc[0] if 'dataset' in sub.columns else 'unknown'
        if 'fitzpatrick' in sub.columns:
            fst_mode = sub['fitzpatrick'].dropna()
            fst = int(fst_mode.mode().iloc[0]) if not fst_mode.empty else -1
        else:
            fst = -1
        stratum = f"{ds}__FST{fst}"
        group_meta[g] = {'size': size, 'stratum': stratum, 'idxs': list(idxs)}

    stratum_groups = defaultdict(list)
    for g, meta in group_meta.items():
        stratum_groups[meta['stratum']].append(g)

    total = len(df)
    stratum_targets = {}
    for stratum, gs in stratum_groups.items():
        stratum_size = sum(group_meta[g]['size'] for g in gs)
        proportion   = stratum_size / total
        stratum_targets[stratum] = max(1, round(proportion * target_n))

    rng = random.Random(seed)
    selected_idx    = []
    selected_groups = []

    for stratum, gs in stratum_groups.items():
        target_samples = stratum_targets[stratum]
        gs_sorted      = sorted(gs, key=lambda g: group_meta[g]['size'])
        accumulated    = 0
        for g in gs_sorted:
            if accumulated >= target_samples and accumulated > 0:
                break
            selected_groups.append(g)
            selected_idx.extend(group_meta[g]['idxs'])
            accumulated += group_meta[g]['size']

    df_sub = df.loc[selected_idx].reset_index(drop=True)
    print(
        f"  Stratified subsample: {len(df_sub)} rows "
        f"across {len(selected_groups)} groups "
        f"({len(stratum_groups)} strata)"
    )
    return df_sub


# ── Test-split selection ──────────────────────────────────────────────────────

def choose_best_test_split(df, groups, mode, test_size=0.2,
                            n_trials=200, seed=42):
    """
    Try n_trials random group-aware splits; return the boolean mask
    for the test set whose distribution best mirrors the full data.
    """
    unique_groups = groups.unique()
    best_score    = float('inf')
    best_mask     = None

    for i in range(n_trials):
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=seed + i
        )
        for _, test_g in splitter.split(unique_groups, groups=unique_groups):
            test_groups = unique_groups[test_g]
            mask        = groups.isin(test_groups)
            score       = distribution_score(df, df[mask], mode)
            if score < best_score:
                best_score = score
                best_mask  = mask

    return best_mask


# ── Greedy fold assignment ────────────────────────────────────────────────────

def greedy_group_fold_assignment(df, groups, n_splits=5,
                                  mode='composite', seed=42,
                                  rebalance_dist_tolerance=0.10,
                                  max_rebalance_iters=200):
    """
    Assign every patient-group to exactly one validation fold.
    Size equity is the primary objective; distributional quality is a
    secondary tiebreaker.
    """
    random.seed(seed)

    group_to_idx = defaultdict(list)
    for idx, g in groups.items():
        group_to_idx[g].append(idx)

    unique_groups = list(groups.unique())

    if len(unique_groups) < n_splits:
        raise ValueError(
            f"Only {len(unique_groups)} distinct groups but {n_splits} "
            f"folds requested.  Reduce --n-splits or increase data size."
        )

    groups_sorted = sorted(
        unique_groups,
        key=lambda g: len(group_to_idx[g]),
        reverse=True
    )

    folds = {i: [] for i in range(n_splits)}
    for fold_idx, group in enumerate(groups_sorted[:n_splits]):
        folds[fold_idx].append(group)
    remaining_groups = groups_sorted[n_splits:]

    def fold_n_samples(fold_id):
        return sum(len(group_to_idx[g]) for g in folds[fold_id])

    def fold_val_df(fold_id):
        idxs = [i for g in folds[fold_id] for i in group_to_idx[g]]
        return df.loc[idxs] if idxs else df.iloc[:0]

    def dist_score_after_add(fold_id, g):
        idxs = (
            [i for gg in folds[fold_id] for i in group_to_idx[gg]]
            + group_to_idx[g]
        )
        return distribution_score(df, df.loc[idxs], mode)

    def dist_score_after_remove(fold_id, g):
        remove_set = set(group_to_idx[g])
        idxs = [
            i for gg in folds[fold_id]
            for i in group_to_idx[gg]
            if i not in remove_set
        ]
        return distribution_score(df, df.loc[idxs], mode) if idxs else float('inf')

    for g in tqdm(
        remaining_groups,
        desc=f'Greedy assignment (size-first, {mode} tiebreak)',
        unit='group'
    ):
        g_size     = len(group_to_idx[g])
        fold_sizes = {f: fold_n_samples(f) for f in range(n_splits)}
        min_size   = min(fold_sizes.values())
        candidates = [
            f for f, s in fold_sizes.items()
            if s <= min_size + g_size
        ]
        if not candidates:
            candidates = list(range(n_splits))
        best_fold = min(candidates, key=lambda f: dist_score_after_add(f, g))
        folds[best_fold].append(g)

    print(
        f"\n  Rebalancing (max {max_rebalance_iters} iters, "
        f"dist_tolerance={rebalance_dist_tolerance:.2f})..."
    )

    for iteration in range(max_rebalance_iters):
        fold_sizes    = {f: fold_n_samples(f) for f in range(n_splits)}
        largest_fold  = max(fold_sizes, key=fold_sizes.get)
        smallest_fold = min(fold_sizes, key=fold_sizes.get)
        size_gap      = fold_sizes[largest_fold] - fold_sizes[smallest_fold]

        if size_gap <= 1:
            print(f"  Converged at iteration {iteration} (size gap = {size_gap}).")
            break

        candidates = sorted(
            folds[largest_fold],
            key=lambda g: len(group_to_idx[g])
        )

        moved = False
        for g in candidates:
            if len(folds[largest_fold]) <= 1:
                break

            g_size    = len(group_to_idx[g])
            new_large = fold_sizes[largest_fold] - g_size
            new_small = fold_sizes[smallest_fold] + g_size
            new_gap   = abs(new_large - new_small)

            if new_gap >= size_gap:
                continue

            dist_before = distribution_score(df, fold_val_df(smallest_fold), mode)
            dist_after  = dist_score_after_add(smallest_fold, g)

            if dist_after <= dist_before + rebalance_dist_tolerance:
                folds[largest_fold].remove(g)
                folds[smallest_fold].append(g)
                moved = True
                print(
                    f"  [iter {iteration+1:3d}] "
                    f"Moved group (n={g_size:,}) "
                    f"fold {largest_fold} → fold {smallest_fold} | "
                    f"gap {size_gap:,} → {new_gap:,}"
                )
                break

        if not moved:
            rebalance_dist_tolerance *= 1.5
            if rebalance_dist_tolerance > 10.0:
                print(
                    f"  Rebalancing stopped at iteration {iteration+1} "
                    f"(tolerance ceiling reached, gap={size_gap:,})."
                )
                break
            print(
                f"  [iter {iteration+1:3d}] No move found — "
                f"relaxing tolerance to {rebalance_dist_tolerance:.3f}"
            )

    fold_sizes = {f: fold_n_samples(f) for f in range(n_splits)}
    print(f"\n  Final val sizes per fold:")
    for f in range(n_splits):
        print(f"    Fold {f+1}: {fold_sizes[f]:,} samples, {len(folds[f])} groups")
    final_gap = max(fold_sizes.values()) - min(fold_sizes.values())
    print(f"  Final size gap: {final_gap:,} samples")

    idx_to_fold = {}
    for f, g_list in folds.items():
        for g in g_list:
            for idx in group_to_idx[g]:
                idx_to_fold[idx] = f

    return idx_to_fold


# ── CSV output ────────────────────────────────────────────────────────────────

def save_fold_csvs(df, idx_to_fold, out_dir,
                   prefix='train_val_dataset_metadata_extended_balanced_fold_'):
    os.makedirs(out_dir, exist_ok=True)
    n_splits = max(idx_to_fold.values()) + 1
    for f in range(n_splits):
        val_mask         = df.index.to_series().map(lambda i: idx_to_fold.get(i) == f)
        df_fold          = df.copy()
        df_fold['split'] = ['valid' if v else 'train' for v in val_mask]
        out_path         = os.path.join(out_dir, f'{prefix}{f+1}.csv')
        df_fold.to_csv(out_path, index=False)
        print(f"  Saved fold {f+1} → {out_path}")


# ── Summary statistics ────────────────────────────────────────────────────────

def print_split_statistics(df, idx_to_fold, test_df, mode):
    n_splits       = max(idx_to_fold.values()) + 1
    total_trainval = len(df)

    print("\n" + "=" * 72)
    print("  SPLIT STATISTICS")
    print(f"  Stratification mode : {mode}")
    print("=" * 72)

    print(
        f"\n  {'Fold':<6} {'n_train':>8} {'n_val':>8} "
        f"{'pct_val':>9}  FST val distribution"
    )
    print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*9}  {'-'*32}")

    val_sizes = []
    for f in range(n_splits):
        val_idx   = [i for i, fold in idx_to_fold.items() if fold == f]
        train_idx = [i for i, fold in idx_to_fold.items() if fold != f]
        n_val     = len(val_idx)
        n_train   = len(train_idx)
        val_sizes.append(n_val)
        pct_val   = 100.0 * n_val / total_trainval if total_trainval > 0 else 0.0

        if 'fitzpatrick' in df.columns and n_val > 0:
            fst_counts = (
                df.loc[val_idx, 'fitzpatrick']
                .value_counts().sort_index()
            )
            fst_str = '  '.join(
                f"FST{int(k)}:{v}"
                for k, v in fst_counts.items()
                if pd.notna(k)
            )
        else:
            fst_str = 'N/A'

        print(
            f"  {f+1:<6} {n_train:>8,} {n_val:>8,} "
            f"{pct_val:>8.1f}%  {fst_str}"
        )

    ideal   = total_trainval / n_splits
    max_dev = max(abs(s - ideal) for s in val_sizes)
    print(
        f"\n  Val size range  : "
        f"min={min(val_sizes):,}  max={max(val_sizes):,}  "
        f"spread={max(val_sizes)-min(val_sizes):,}  "
        f"ideal={ideal:.0f}  max_dev={max_dev:.0f}"
    )

    datasets = sorted(df['dataset'].dropna().unique())
    col_w    = max(10, max(len(d) for d in datasets))

    print(f"\n  Dataset breakdown per validation fold:")
    header_ds = f"  {'Fold':<6}"
    for ds in datasets:
        header_ds += f"  {ds[:col_w]:>{col_w}}"
    print(header_ds)
    print(f"  {'-'*6}" + f"  {'-'*col_w}" * len(datasets))

    for f in range(n_splits):
        val_idx  = [i for i, fold in idx_to_fold.items() if fold == f]
        val_df_f = df.loc[val_idx]
        row_str  = f"  {f+1:<6}"
        for ds in datasets:
            cnt      = (val_df_f['dataset'] == ds).sum()
            row_str += f"  {cnt:>{col_w},}"
        print(row_str)

    if 'fitzpatrick' in df.columns:
        fst_vals = sorted(df['fitzpatrick'].dropna().unique())
        print(f"\n  FST breakdown per validation fold:")
        header_fst = f"  {'Fold':<6}"
        for fv in fst_vals:
            header_fst += f"  {'FST'+str(int(fv)):>6}"
        print(header_fst)
        print(f"  {'-'*6}" + "  " + "  ".join(["-" * 6] * len(fst_vals)))
        for f in range(n_splits):
            val_idx  = [i for i, fold in idx_to_fold.items() if fold == f]
            val_df_f = df.loc[val_idx]
            row_str  = f"  {f+1:<6}"
            for fv in fst_vals:
                cnt      = (val_df_f['fitzpatrick'] == fv).sum()
                row_str += f"  {cnt:>6,}"
            print(row_str)

    grand_total = len(test_df) + total_trainval
    print(f"\n  {'─'*72}")
    print(f"  TEST SPLIT")
    print(f"  {'─'*72}")
    print(f"  n_test          : {len(test_df):,}")
    print(f"  pct_of_total    : {100.*len(test_df)/grand_total:.1f}%")

    if 'fitzpatrick' in test_df.columns:
        fst_test = test_df['fitzpatrick'].value_counts().sort_index()
        fst_str  = '  '.join(
            f"FST{int(k)}:{v}"
            for k, v in fst_test.items()
            if pd.notna(k)
        )
        print(f"  FST distribution: {fst_str}")

    if 'dataset' in test_df.columns:
        print(f"  Dataset breakdown:")
        for ds, cnt in test_df['dataset'].value_counts().sort_index().items():
            print(f"    {ds:<22}: {cnt:>6,}  ({100.*cnt/len(test_df):.1f}%)")

    print("=" * 72 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    raw_df = pd.read_csv(args.input, dtype=str)

    if 'fitzpatrick' in raw_df.columns:
        raw_df['fitzpatrick'] = pd.to_numeric(raw_df['fitzpatrick'], errors='coerce')

    print("\n── Dataset inclusion policy ──────────────────────────────────────")

    all_datasets = raw_df['dataset'].dropna().unique()
    for ds in sorted(all_datasets):
        n = (raw_df['dataset'] == ds).sum()
        if ds in EXCLUDED_DATASETS:
            tag = "EXCLUDED"
        elif ds in FULL_DATASETS:
            tag = "FULL"
        elif ds in TOPUP_DATASETS:
            tag = "TOP-UP (FST-balanced sampling)"
        else:
            tag = "UNKNOWN (will be excluded — add to policy constants)"
        print(f"  {ds:<22}: {n:>7,} rows  [{tag}]")

    base_df = raw_df[raw_df['dataset'].isin(FULL_DATASETS)].copy()
    print(f"\n  Base (full-inclusion) rows : {len(base_df):,}")

    topup_dfs = {}
    for ds_name in TOPUP_DATASETS:
        subset = raw_df[raw_df['dataset'] == ds_name].copy()
        topup_dfs[ds_name] = subset
        print(f"  Top-up pool '{ds_name}'      : {len(subset):,} rows")

    print("\n── FST-balanced top-up ───────────────────────────────────────────")
    df = fst_topup_sample(base_df, topup_dfs, seed=args.seed)

    if getattr(args, 'debug_n', 0) and int(args.debug_n) > 0:
        target_n = int(args.debug_n)
        print(f"\nDebug mode: stratified subsample of ~{target_n} rows")
        df = stratified_group_subsample(df, target_n, seed=args.seed)

    groups          = make_patient_group(df)
    n_unique_groups = groups.nunique()
    print(f"\nTotal samples        : {len(df):,}")
    print(f"Distinct groups      : {n_unique_groups:,}")

    print("\nRunning patient leakage audit...")
    n_conflicts = audit_patient_leakage(df, groups)
    if n_conflicts > 0:
        raise RuntimeError(
            f"Aborting: {n_conflicts} patient identity conflict(s). "
            f"Fix metadata before splitting."
        )

    min_groups_needed = args.n_splits + 1
    if n_unique_groups < min_groups_needed:
        raise ValueError(
            f"Only {n_unique_groups} distinct groups but at least "
            f"{min_groups_needed} needed. "
            f"Reduce --n-splits or increase --debug-n."
        )

    print("\nDetecting feasible stratification mode...")
    mode = detect_stratify_mode(df, args.n_splits)

    print('\nSelecting best test split...')
    test_mask = choose_best_test_split(
        df, groups, mode,
        test_size=0.2,
        n_trials=args.n_trials,
        seed=args.seed
    )
    df['split'] = 'train'
    df.loc[test_mask, 'split'] = 'test'

    os.makedirs(args.output_metadata_dir, exist_ok=True)
    test_out = os.path.join(
        args.output_metadata_dir,
        'test_dataset_metadata_extended_balanced.csv'
    )
    test_df = df[df['split'] == 'test'].copy()
    test_df.to_csv(test_out, index=False)
    print(f"Saved test split → {test_out}  ({len(test_df):,} samples)")

    train_df     = df[df['split'] == 'train'].copy()
    train_groups = make_patient_group(train_df)

    print(f"\nTrain+val samples    : {len(train_df):,}")
    print(f"Train+val groups     : {train_groups.nunique():,}")

    ideal_fold_size    = len(train_df) / args.n_splits
    group_to_idx_check = defaultdict(list)
    for idx, g in train_groups.items():
        group_to_idx_check[g].append(idx)
    large_groups = {
        g: len(idxs)
        for g, idxs in group_to_idx_check.items()
        if len(idxs) > ideal_fold_size * 0.5
    }
    if large_groups:
        print(
            f"\n  [WARN] {len(large_groups)} group(s) contain more than "
            f"50% of the ideal fold size ({ideal_fold_size:.0f}). "
            f"Perfect size balance is not achievable without splitting "
            f"patient groups (which would cause leakage)."
        )
        for g, sz in sorted(large_groups.items(), key=lambda x: -x[1]):
            print(f"    '{g}': {sz:,} samples")

    print(
        f"\nAssigning groups to {args.n_splits} folds "
        f"(mode='{mode}', size-first greedy + rebalance)..."
    )
    idx_to_fold = greedy_group_fold_assignment(
        train_df, train_groups,
        n_splits=args.n_splits,
        mode=mode,
        seed=args.seed,
        rebalance_dist_tolerance=0.10,
        max_rebalance_iters=200,
    )

    print(f"\nSaving fold CSVs → {args.output_metadata_dir}")
    save_fold_csvs(train_df, idx_to_fold, args.output_metadata_dir)

    df['fold'] = ''
    for idx, f in idx_to_fold.items():
        df.loc[idx, 'fold'] = int(f) + 1

    print_split_statistics(train_df, idx_to_fold, test_df, mode)


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Create reproducible test/train/val splits and folds '
                    'with FST-balanced dataset top-up'
    )
    p.add_argument(
        '--input',
        default='metadata/all_dataset_metadata_extended.csv',
        help='Path to the combined metadata CSV.'
    )
    p.add_argument(
        '--output-metadata-dir',
        default='metadata',
        help='Directory where split CSVs will be written.'
    )
    p.add_argument('--n-splits',  type=int, default=5)
    p.add_argument('--n-trials',  type=int, default=200)
    p.add_argument('--seed',      type=int, default=42)
    p.add_argument('--debug-n',   type=int, default=0,
                   help='If > 0, subsample to this many rows for fast debugging.')
    args = p.parse_args()
    main(args)
