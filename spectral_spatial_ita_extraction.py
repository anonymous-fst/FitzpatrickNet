"""
spectral_spatial_ita_extraction.py
==========================================
Compute per-image Individual Typology Angle (ITA) values using a
spectral-spatial clustering pipeline.

Usage
-----
CLI mode (recommended):
    python spectral_spatial_ita_extraction.py \
        --image_dir /path/to/images \
        --metadata_csv /path/to/metadata.csv

The metadata CSV must contain a 'filename' column with image file names
relative to --image_dir.  Results are written back to the same CSV as a
new column 'ITA_value_spectral_spatial'.
"""

import os

os.environ["OMP_NUM_THREADS"]     = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"]     = "1"

import cv2
import math
import argparse
import multiprocessing

import numpy as np
import pandas as pd
from skimage import color
from sklearn.cluster import KMeans, spectral_clustering
from sklearn.feature_extraction.image import img_to_graph
from kneed import KneeLocator
from tqdm import tqdm
from joblib import Parallel, delayed

tqdm.pandas(desc="Spectral Spatial ITA")

# ── Configuration ─────────────────────────────────────────────────────────────

N_JOBS = multiprocessing.cpu_count() - 1

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Compute ITA values for skin images using spectral-spatial clustering."
)
parser.add_argument(
    "--image_dir",
    type=str,
    default=None,
    help="Path to the directory containing input images."
)
parser.add_argument(
    "--metadata_csv",
    type=str,
    default=None,
    help="Path to the metadata CSV file. Must contain a 'filename' column."
)
args = parser.parse_args()

USE_CLI_MODE = args.image_dir is not None and args.metadata_csv is not None

if (args.image_dir is None) != (args.metadata_csv is None):
    parser.error(
        "Both --image_dir and --metadata_csv must be provided together, or neither."
    )


# ── Core functions ────────────────────────────────────────────────────────────

def get_optimal_k(skin_pixels_sampled):
    """Kneedle-based optimal-k detection for KMeans clustering."""
    inertias = []
    K_range  = range(2, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42).fit(
            skin_pixels_sampled
        )
        inertias.append(kmeans.inertia_)
    kneedle = KneeLocator(
        list(K_range), inertias, curve='convex', direction='decreasing'
    )
    return kneedle.knee if kneedle.knee is not None else 3


def calculate_ita_spectral_spatial(image_path, mask_path):
    """
    Compute the ITA value for a single image using spectral-spatial clustering.

    Parameters
    ----------
    image_path : str
        Full path to the input image.
    mask_path : str or None
        Full path to a binary foreground mask, or None / NaN to use the
        full image.

    Returns
    -------
    (float, str or None)
        ITA value and an error code string (None on success).
    """
    if not isinstance(image_path, str) or not os.path.exists(image_path):
        return np.nan, "IMAGE_NOT_FOUND"

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return np.nan, "CV2_READ_FAILED"

    H, W, _ = img_bgr.shape

    # ── Masking ───────────────────────────────────────────────────────────────
    if (mask_path is None
            or pd.isna(mask_path)
            or not isinstance(mask_path, str)
            or not os.path.exists(mask_path)):
        bg_mask = np.ones((H, W), dtype=np.uint8) * 255
    else:
        bg_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if bg_mask is None:
            bg_mask = np.ones((H, W), dtype=np.uint8) * 255
        else:
            _, bg_mask = cv2.threshold(bg_mask, 127, 255, cv2.THRESH_BINARY)

    if np.sum(bg_mask > 0) == 0:
        return np.nan, "MASK_ALL_ZERO"

    try:
        # ── CLAHE preprocessing ───────────────────────────────────────────────
        img_lab_cv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b_ch = cv2.split(img_lab_cv)
        clahe      = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_lab_clahe = cv2.merge((clahe.apply(l), a, b_ch))

        # ── Otsu thresholding on HSV V-channel ────────────────────────────────
        img_hsv   = cv2.cvtColor(
            cv2.cvtColor(img_lab_clahe, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2HSV
        )
        v_channel = img_hsv[:, :, 2]
        valid_v   = v_channel[bg_mask > 0]
        if len(valid_v) == 0:
            return np.nan, "NO_VALID_V_PIXELS"

        hist, _   = np.histogram(valid_v, bins=256, range=(0, 256))
        total     = valid_v.shape[0]
        sum_total = np.dot(np.arange(256), hist)
        sum_b = 0.0; w_b = 0; var_max = 0.0; threshold = 0
        for i in range(256):
            w_b += hist[i]
            if w_b == 0:
                continue
            w_f = total - w_b
            if w_f == 0:
                break
            sum_b += i * hist[i]
            var_between = (
                w_b * w_f
                * ((sum_b / w_b) - ((sum_total - sum_b) / w_f)) ** 2
            )
            if var_between > var_max:
                var_max = var_between
                threshold = i

        skin_mask_otsu = np.zeros((H, W), dtype=np.uint8)
        skin_mask_otsu[v_channel >= threshold] = 255
        skin_mask_otsu[bg_mask == 0] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        healthy_skin_mask = cv2.bitwise_and(
            cv2.erode(skin_mask_otsu, kernel, iterations=2), bg_mask
        )
        if np.sum(healthy_skin_mask > 0) < 10:
            healthy_skin_mask = bg_mask

        img_rgb      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_lab_real = color.rgb2lab(img_rgb)
        skin_pixels  = img_lab_real[healthy_skin_mask > 0]

        if len(skin_pixels) == 0:
            return np.nan, "NO_SKIN_PIXELS_AFTER_MASK"

        sampled_indices     = np.random.choice(
            len(skin_pixels), size=min(5000, len(skin_pixels)), replace=False
        )
        skin_pixels_sampled = skin_pixels[sampled_indices]
        optimal_k           = get_optimal_k(skin_pixels_sampled)

        # ── Spatial graph spectral clustering ─────────────────────────────────
        dim        = 64
        lab_small  = cv2.resize(img_lab_real, (dim, dim))
        mask_small = cv2.resize(
            healthy_skin_mask, (dim, dim),
            interpolation=cv2.INTER_NEAREST
        )

        try:
            graph = img_to_graph(lab_small).tocsr()
        except Exception as e:
            return np.nan, f"IMG_TO_GRAPH_FAILED::{type(e).__name__}"

        valid_nodes = np.where(mask_small.ravel() > 0)[0]
        n_valid     = len(valid_nodes)

        if n_valid < optimal_k:
            if n_valid < 2:
                return np.nan, f"TOO_FEW_VALID_NODES_UNRECOVERABLE::n_valid={n_valid}"
            optimal_k = n_valid

        try:
            subgraph = graph[valid_nodes][:, valid_nodes]
        except Exception as e:
            return np.nan, f"SUBGRAPH_SLICE_FAILED::{type(e).__name__}"

        std = subgraph.data.std()
        if std == 0:
            std = 1e-6
        subgraph.data = np.exp(-(subgraph.data ** 2) / (2 * std ** 2))

        if np.any(np.isnan(subgraph.data)) or np.any(np.isinf(subgraph.data)):
            return np.nan, "SUBGRAPH_CONTAINS_NAN_OR_INF_AFTER_RBF"

        safe_k = min(optimal_k, n_valid - 1)

        try:
            labels_spatial = spectral_clustering(
                subgraph,
                n_clusters=safe_k,
                eigen_solver='arpack',
                random_state=42
            )
        except Exception as e:
            return np.nan, f"SPECTRAL_CLUSTERING_FAILED::{type(e).__name__}::{e}"

        largest_c        = np.argmax(np.bincount(labels_spatial))
        valid_lab_pixels = lab_small.reshape(-1, 3)[valid_nodes]
        center           = np.mean(
            valid_lab_pixels[labels_spatial == largest_c], axis=0
        )

        b_mean = center[2] if center[2] != 0 else 1e-6
        return math.atan((center[0] - 50) / b_mean) * (180 / math.pi), None

    except Exception as e:
        return np.nan, f"UNHANDLED::{type(e).__name__}::{e}"


# ── CLI mode ──────────────────────────────────────────────────────────────────

if USE_CLI_MODE:
    print("--- CLI MODE ---")
    print(f"  Image directory : {args.image_dir}")
    print(f"  Metadata CSV    : {args.metadata_csv}")

    if not os.path.exists(args.metadata_csv):
        print(f"  -> ERROR: Metadata CSV not found: {args.metadata_csv}")
        exit(1)

    if not os.path.isdir(args.image_dir):
        print(f"  -> ERROR: Image directory not found: {args.image_dir}")
        exit(1)

    try:
        df = pd.read_csv(args.metadata_csv)

        if 'filename' not in df.columns:
            print("  -> ERROR: Metadata CSV must contain a 'filename' column.")
            exit(1)

        image_paths = df['filename'].apply(
            lambda fn: os.path.join(args.image_dir, fn)
            if isinstance(fn, str) else None
        ).tolist()

        rows_with_paths = list(zip(df.index, image_paths))

        results = Parallel(n_jobs=N_JOBS)(
            delayed(calculate_ita_spectral_spatial)(image_path, None)
            for _, image_path in tqdm(rows_with_paths, desc="Spectral Spatial")
        )

        result_series = pd.Series(np.nan, index=df.index, dtype="float64")
        for (idx, _), result in zip(rows_with_paths, results):
            result_series.loc[idx] = result[0]

        df['ITA_value_spectral_spatial'] = result_series
        df.to_csv(args.metadata_csv, index=False)
        print(f"  -> Saved: {args.metadata_csv}")

    except Exception as e:
        print(f"  -> Error: {e}")
