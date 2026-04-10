"""
Landcover Classification Utilities

This module provides utilities for discrete landcover classification workflows,
including tile export with background filtering and radiometric normalization
for multi-temporal image comparability.

Key Features:
- Enhanced tile filtering with configurable feature ratio thresholds
- Separate statistics tracking for different skip reasons
- LIRRN (Location-Independent Relative Radiometric Normalization)
- Maintains full compatibility with base geoai workflow
- Optimized for discrete landcover classification tasks

Date: November 2025
"""

import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/headless use
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio import features
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.warp import calculate_default_transform, reproject
from skimage.filters import threshold_multiotsu
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

__all__ = [
    "export_landcover_tiles",
    "generate_radiometric_histogram_report",
    "normalize_radiometric_lirrn",
    "normalize_radiometric_planet",
    "normalize_radiometric",
    "normalize_multitemporal",
]


def export_landcover_tiles(
    in_raster: str,
    out_folder: str,
    in_class_data: Optional[Union[str, gpd.GeoDataFrame]] = None,
    tile_size: int = 256,
    stride: int = 128,
    class_value_field: str = "class",
    buffer_radius: float = 0,
    max_tiles: Optional[int] = None,
    quiet: bool = False,
    all_touched: bool = True,
    create_overview: bool = False,
    skip_empty_tiles: bool = False,
    min_feature_ratio: Union[bool, float] = False,
    metadata_format: str = "PASCAL_VOC",
) -> Dict[str, Any]:
    """
    Export GeoTIFF tiles optimized for landcover classification training.

    This function extends the base export_geotiff_tiles with enhanced filtering
    capabilities specifically designed for discrete landcover classification.
    It can filter out tiles dominated by background pixels to improve training
    data quality and reduce dataset size.

    Args:
        in_raster: Path to input raster (image to tile)
        out_folder: Output directory for tiles
        in_class_data: Path to vector mask or GeoDataFrame (optional for image-only export)
        tile_size: Size of output tiles in pixels (default: 256)
        stride: Stride for sliding window (default: 128)
        class_value_field: Field name containing class values (default: "class")
        buffer_radius: Buffer radius around features in pixels (default: 0)
        max_tiles: Maximum number of tiles to export (default: None)
        quiet: Suppress progress output (default: False)
        all_touched: Include pixels touched by geometry (default: True)
        create_overview: Create overview image showing tile locations (default: False)
        skip_empty_tiles: Skip tiles with no features (default: False)
        min_feature_ratio: Minimum ratio of non-background pixels required to keep tile
            - False: Disable ratio filtering (default)
            - 0.0-1.0: Minimum ratio threshold (e.g., 0.1 = 10% features required)
        metadata_format: Annotation format ("PASCAL_VOC" or "YOLO")

    Returns:
        Dictionary containing:
            - tiles_exported: Number of tiles successfully exported
            - tiles_skipped_empty: Number of completely empty tiles skipped
            - tiles_skipped_ratio: Number of tiles filtered by min_feature_ratio
            - output_dirs: Dictionary with paths to images and labels directories

    Examples:
        # Original behavior (no filtering)
        export_landcover_tiles(
            "input.tif",
            "output",
            "mask.shp",
            skip_empty_tiles=True
        )

        # Light filtering (keep tiles with ≥5% features)
        export_landcover_tiles(
            "input.tif",
            "output",
            "mask.shp",
            skip_empty_tiles=True,
            min_feature_ratio=0.05
        )

        # Moderate filtering (keep tiles with ≥15% features)
        export_landcover_tiles(
            "input.tif",
            "output",
            "mask.shp",
            skip_empty_tiles=True,
            min_feature_ratio=0.15
        )

    Note:
        This function is designed for discrete landcover classification where
        class 0 typically represents background/no data. The min_feature_ratio
        parameter counts non-zero pixels as "features".
    """

    # Validate min_feature_ratio parameter
    if min_feature_ratio is not False:
        if not isinstance(min_feature_ratio, (int, float)):
            warnings.warn(
                f"min_feature_ratio must be a number between 0.0 and 1.0, got {type(min_feature_ratio)}. "
                "Disabling ratio filtering."
            )
            min_feature_ratio = False
        elif not (0.0 <= min_feature_ratio <= 1.0):
            warnings.warn(
                f"min_feature_ratio must be between 0.0 and 1.0, got {min_feature_ratio}. "
                "Disabling ratio filtering."
            )
            min_feature_ratio = False

    # Create output directories
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    images_dir = out_folder / "images"
    labels_dir = out_folder / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    if metadata_format == "PASCAL_VOC":
        ann_dir = out_folder / "annotations"
        ann_dir.mkdir(exist_ok=True)

    # Initialize statistics
    stats = {
        "tiles_exported": 0,
        "tiles_skipped_empty": 0,
        "tiles_skipped_ratio": 0,
        "output_dirs": {"images": str(images_dir), "labels": str(labels_dir)},
    }

    # Open raster
    with rasterio.open(in_raster) as src:
        height, width = src.shape

        # Detect if in_class_data is raster or vector
        is_class_data_raster = False
        class_src = None
        gdf = None
        mask_array = None

        if in_class_data is not None:
            if isinstance(in_class_data, str):
                file_ext = Path(in_class_data).suffix.lower()
                if file_ext in [
                    ".tif",
                    ".tiff",
                    ".img",
                    ".jp2",
                    ".png",
                    ".bmp",
                    ".gif",
                ]:
                    try:
                        # Try to open as raster
                        class_src = rasterio.open(in_class_data)
                        is_class_data_raster = True

                        # Verify CRS match
                        if class_src.crs != src.crs:
                            if not quiet:
                                print(
                                    f"Warning: CRS mismatch between image ({src.crs}) and mask ({class_src.crs})"
                                )
                    except Exception as e:
                        is_class_data_raster = False
                        if not quiet:
                            print(f"Could not open as raster, trying vector: {e}")

                # If not raster or raster open failed, try vector
                if not is_class_data_raster:
                    gdf = gpd.read_file(in_class_data)

                    # Reproject if needed
                    if gdf.crs != src.crs:
                        if not quiet:
                            print(f"Reprojecting mask from {gdf.crs} to {src.crs}")
                        gdf = gdf.to_crs(src.crs)

                    # Apply buffer if requested
                    if buffer_radius > 0:
                        gdf.geometry = gdf.geometry.buffer(buffer_radius)

                    # For vector data, rasterize entire mask up front for efficiency
                    shapes = [
                        (geom, value)
                        for geom, value in zip(gdf.geometry, gdf[class_value_field])
                    ]
                    mask_array = features.rasterize(
                        shapes,
                        out_shape=(height, width),
                        transform=src.transform,
                        all_touched=all_touched,
                        fill=0,
                        dtype=np.uint8,
                    )
            else:
                # Assume GeoDataFrame passed directly
                gdf = in_class_data

                # Reproject if needed
                if gdf.crs != src.crs:
                    if not quiet:
                        print(f"Reprojecting mask from {gdf.crs} to {src.crs}")
                    gdf = gdf.to_crs(src.crs)

                # Apply buffer if requested
                if buffer_radius > 0:
                    gdf.geometry = gdf.geometry.buffer(buffer_radius)

                # Rasterize entire mask up front
                shapes = [
                    (geom, value)
                    for geom, value in zip(gdf.geometry, gdf[class_value_field])
                ]
                mask_array = features.rasterize(
                    shapes,
                    out_shape=(height, width),
                    transform=src.transform,
                    all_touched=all_touched,
                    fill=0,
                    dtype=np.uint8,
                )

        # Calculate tile positions
        tile_positions = []
        for y in range(0, height - tile_size + 1, stride):
            for x in range(0, width - tile_size + 1, stride):
                tile_positions.append((x, y))

        if max_tiles:
            tile_positions = tile_positions[:max_tiles]

        # Process tiles
        pbar = tqdm(tile_positions, desc="Exporting tiles", disable=quiet)

        for tile_idx, (x, y) in enumerate(pbar):
            window = Window(x, y, tile_size, tile_size)

            # Read image tile
            image_tile = src.read(window=window)

            # Read mask tile based on data type
            mask_tile = None
            has_features = False

            if is_class_data_raster and class_src is not None:
                # For raster masks, read directly from the raster source
                # Get window transform and bounds
                window_transform = src.window_transform(window)
                minx = window_transform[2]
                maxy = window_transform[5]
                maxx = minx + tile_size * window_transform[0]
                miny = maxy + tile_size * window_transform[4]

                # Get corresponding window in class raster
                window_class = rasterio.windows.from_bounds(
                    minx, miny, maxx, maxy, class_src.transform
                )

                try:
                    # Read label data from raster
                    mask_tile = class_src.read(
                        1,
                        window=window_class,
                        boundless=True,
                        out_shape=(tile_size, tile_size),
                    )

                    # Check if tile has features
                    has_features = np.any(mask_tile > 0)
                except Exception as e:
                    if not quiet:
                        pbar.write(f"Error reading mask tile at ({x}, {y}): {e}")
                    continue

            elif mask_array is not None:
                # For vector masks (pre-rasterized)
                mask_tile = mask_array[y : y + tile_size, x : x + tile_size]
                has_features = np.any(mask_tile > 0)

            # Skip empty tiles if requested
            if skip_empty_tiles and not has_features:
                stats["tiles_skipped_empty"] += 1
                continue

            # Apply min_feature_ratio filtering if enabled
            if skip_empty_tiles and has_features and min_feature_ratio is not False:
                # Calculate ratio of non-background pixels
                total_pixels = mask_tile.size
                feature_pixels = np.sum(mask_tile > 0)
                feature_ratio = feature_pixels / total_pixels

                # Skip tile if below threshold
                if feature_ratio < min_feature_ratio:
                    stats["tiles_skipped_ratio"] += 1
                    continue

            # Save image tile
            tile_name = f"tile_{tile_idx:06d}.tif"
            image_path = images_dir / tile_name

            # Get transform for this tile
            tile_transform = src.window_transform(window)

            # Write image
            with rasterio.open(
                image_path,
                "w",
                driver="GTiff",
                height=tile_size,
                width=tile_size,
                count=src.count,
                dtype=src.dtypes[0],
                crs=src.crs,
                transform=tile_transform,
                compress="lzw",
            ) as dst:
                dst.write(image_tile)

            # Save mask tile if available
            if mask_tile is not None:
                mask_path = labels_dir / tile_name
                with rasterio.open(
                    mask_path,
                    "w",
                    driver="GTiff",
                    height=tile_size,
                    width=tile_size,
                    count=1,
                    dtype=np.uint8,
                    crs=src.crs,
                    transform=tile_transform,
                    compress="lzw",
                ) as dst:
                    dst.write(mask_tile, 1)

            stats["tiles_exported"] += 1

            # Update progress bar description with selection count
            if not quiet:
                pbar.set_description(
                    f"Exporting tiles ({stats['tiles_exported']}/{tile_idx + 1})"
                )

    # Close raster class source if opened
    if class_src is not None:
        class_src.close()

    # Print summary
    if not quiet:
        print(f"\n{'='*60}")
        print("TILE EXPORT SUMMARY")
        print(f"{'='*60}")
        print(f"Tiles exported: {stats['tiles_exported']}/{len(tile_positions)}")
        if skip_empty_tiles:
            print(f"Tiles skipped (empty): {stats['tiles_skipped_empty']}")
        if min_feature_ratio is not False:
            print(
                f"Tiles skipped (low feature ratio < {min_feature_ratio}): {stats['tiles_skipped_ratio']}"
            )
        print(f"\nOutput directories:")
        print(f"  Images: {stats['output_dirs']['images']}")
        print(f"  Labels: {stats['output_dirs']['labels']}")
        print(f"{'='*60}\n")

    return stats


# ---------------------------------------------------------------------------
# Radiometric Normalization
# ---------------------------------------------------------------------------


def _load_raster(filepath: str) -> Tuple[np.ndarray, dict]:
    """Load a multi-band raster as a (H, W, B) float64 array.

    Args:
        filepath: Path to the raster file.

    Returns:
        Tuple of (image_array, profile) where image_array has shape (H, W, B)
        and profile is the rasterio dataset profile dict.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Raster file not found: {filepath}")
    with rasterio.open(filepath) as src:
        img = src.read()  # (B, H, W)
        profile = src.profile.copy()
    img = np.moveaxis(img, 0, -1).astype(np.float64)
    if np.any(~np.isfinite(img)):
        warnings.warn(
            "Raster contains NaN or infinite values; replacing with 0.",
            stacklevel=2,
        )
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    return img, profile


def _save_raster(filepath: str, img: np.ndarray, profile: dict) -> None:
    """Save a (H, W, B) array as a multi-band GeoTIFF.

    Args:
        filepath: Output file path.
        img: Image array with shape (H, W, B).
        profile: Rasterio profile dict (from ``_load_raster``).

    Raises:
        ValueError: If *img* is not 3-dimensional.
    """
    if img.ndim != 3:
        raise ValueError(f"Expected 3-D array (H, W, B), got shape {img.shape}")
    out_profile = profile.copy()
    out_profile.update(
        dtype="float64",
        count=img.shape[2],
        height=img.shape[0],
        width=img.shape[1],
        compress="lzw",
    )
    with rasterio.open(filepath, "w", **out_profile) as dst:
        for i in range(img.shape[2]):
            dst.write(img[:, :, i], i + 1)


def _compute_distances(
    p_n: int,
    a1: np.ndarray,
    b1: np.ndarray,
    id_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Select *p_n* samples closest to the maximum value, then subsample.

    Args:
        p_n: Number of candidate samples to draw from each array.
        a1: Non-zero reference pixel values (1-D).
        b1: Non-zero subject pixel values (1-D).
        id_indices: Random indices used to subsample from the candidates.

    Returns:
        Tuple of (sub_samples, ref_samples) after subsampling.
    """
    max_ref = np.max(a1)
    idx_ref = np.argsort(np.abs(a1 - max_ref))
    ref_candidates = a1[idx_ref[:p_n]]

    max_sub = np.max(b1)
    idx_sub = np.argsort(np.abs(b1 - max_sub))
    sub_candidates = b1[idx_sub[:p_n]]

    safe_indices = id_indices[
        id_indices < min(len(sub_candidates), len(ref_candidates))
    ]
    if len(safe_indices) == 0:
        return sub_candidates, ref_candidates
    return sub_candidates[safe_indices], ref_candidates[safe_indices]


def _compute_sample(
    p_n: int,
    a1: np.ndarray,
    b1: np.ndarray,
    id_indices: np.ndarray,
    num_sampling_rounds: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Multiple rounds of distance-based sampling, concatenated (non-zero only).

    Args:
        p_n: Number of samples per round.
        a1: Non-zero reference pixel values (1-D).
        b1: Non-zero subject pixel values (1-D).
        id_indices: Random subsample indices.
        num_sampling_rounds: Number of sampling rounds.

    Returns:
        Tuple of (sub_combined, ref_combined).
    """
    pairs = [
        _compute_distances(p_n, a1, b1, id_indices) for _ in range(num_sampling_rounds)
    ]
    sub_combined = np.concatenate([s[s != 0] for s, _ in pairs])
    ref_combined = np.concatenate([r[r != 0] for _, r in pairs])
    return sub_combined, ref_combined


def _sample_selection(
    p_n: int,
    a: np.ndarray,
    b: np.ndarray,
    id_indices: np.ndarray,
    num_sampling_rounds: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Select representative sample pairs from a single quantization level.

    Args:
        p_n: Number of sample points.
        a: Flattened reference pixels (masked by quantization level; 0 = outside).
        b: Flattened subject pixels (masked by quantization level; 0 = outside).
        id_indices: Random sub-sampling indices.
        num_sampling_rounds: Number of sampling rounds.

    Returns:
        Tuple of (sub_samples, ref_samples), each 1-D.
    """
    a1 = a[a != 0]
    b1 = b[b != 0]

    if len(a1) == 0 or len(b1) == 0:
        return np.array([0.0]), np.array([0.0])

    if len(a1) < p_n or len(b1) < p_n:
        min_len = min(len(a1), len(b1))
        return b1[:min_len], a1[:min_len]

    sub_1, ref_1 = _compute_sample(p_n, a1, b1, id_indices, num_sampling_rounds)

    sub = np.concatenate([sub_1[sub_1 != 0], sub_1[sub_1 == 0]])
    ref = np.concatenate([ref_1[ref_1 != 0], ref_1[ref_1 == 0]])
    return sub, ref


def _linear_reg(
    sub_samples: np.ndarray,
    ref_samples: np.ndarray,
    image_band: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    """OLS linear regression to normalize one image band.

    Fits ``ref = intercept + slope * sub`` and applies the transformation
    to *image_band*.

    Args:
        sub_samples: Subject sample values (1-D).
        ref_samples: Reference sample values (1-D).
        image_band: Full subject band to normalize (H, W).

    Returns:
        Tuple of (normalized_band, adjusted_r_squared, rmse).

    Raises:
        ValueError: If fewer than 2 valid samples are available for regression.
    """
    mask = (sub_samples != 0) & (ref_samples != 0)
    sub_clean = sub_samples[mask]
    ref_clean = ref_samples[mask]

    if len(sub_clean) < 2:
        raise ValueError(
            f"Insufficient samples for regression: got {len(sub_clean)}, need >= 2"
        )

    X = sub_clean.reshape(-1, 1)
    y = ref_clean

    model = LinearRegression().fit(X, y)
    intercept, slope = model.intercept_, model.coef_[0]

    norm_band = intercept + slope * image_band

    y_pred = model.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    n = len(y)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
    r_adj = 1 - (1 - r2) * (n - 1) / (n - 2) if n > 2 else 0.0
    rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))

    return norm_band, r_adj, rmse


def _lirrn(
    p_n: int,
    sub_img: np.ndarray,
    ref_img: np.ndarray,
    num_quantisation_classes: int = 3,
    num_sampling_rounds: int = 3,
    subsample_ratio: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Core LIRRN algorithm.

    Implements Location-Independent Relative Radiometric Normalization.
    Processes each band independently via multi-Otsu thresholding,
    stratified sampling and per-band linear regression.

    Args:
        p_n: Number of sample points per quantization level.
        sub_img: Subject image (H, W, B) float64.
        ref_img: Reference image (H, W, B) float64.
        num_quantisation_classes: Number of brightness strata (default 3).
        num_sampling_rounds: Number of sampling rounds (default 3).
        subsample_ratio: Fraction of candidates retained (default 0.1).
        rng: Numpy random Generator for reproducibility.

    Returns:
        Tuple of (normalized_image, rmse_per_band, r_adj_per_band).
    """
    if rng is None:
        rng = np.random.default_rng()

    num_bands = sub_img.shape[2]

    id_indices = rng.integers(0, p_n, size=max(1, round(subsample_ratio * p_n)))

    norm_img = np.zeros_like(sub_img, dtype=np.float64)
    rmse = np.zeros(num_bands)
    r_adj = np.zeros(num_bands)

    # Quantize each band into brightness levels via multi-Otsu
    sub_labels = np.zeros_like(sub_img, dtype=np.int32)
    ref_labels = np.zeros_like(ref_img, dtype=np.int32)

    for j in range(num_bands):
        for img, labels in [(sub_img, sub_labels), (ref_img, ref_labels)]:
            nonzero = img[:, :, j][img[:, :, j] != 0]
            if len(nonzero) > 0:
                try:
                    thresh = threshold_multiotsu(
                        nonzero, classes=num_quantisation_classes
                    )
                    labels[:, :, j] = np.digitize(img[:, :, j], bins=thresh) + 1
                except ValueError:
                    labels[:, :, j] = 1

    # For each band: sample from quantization levels then regress
    for j in range(num_bands):
        sub_list, ref_list = [], []

        for level in range(1, num_quantisation_classes + 1):
            a = np.where(ref_labels[:, :, j] == level, ref_img[:, :, j], 0).ravel()
            b = np.where(sub_labels[:, :, j] == level, sub_img[:, :, j], 0).ravel()
            sub_s, ref_s = _sample_selection(p_n, a, b, id_indices, num_sampling_rounds)
            sub_list.append(sub_s)
            ref_list.append(ref_s)

        all_sub = np.concatenate(sub_list)
        all_ref = np.concatenate(ref_list)

        try:
            norm_img[:, :, j], r_adj[j], rmse[j] = _linear_reg(
                all_sub, all_ref, sub_img[:, :, j]
            )
        except ValueError:
            warnings.warn(
                f"Band {j}: insufficient samples for regression, "
                "returning band unchanged.",
                stacklevel=2,
            )
            norm_img[:, :, j] = sub_img[:, :, j]
            continue

        ref_band = ref_img[:, :, j]
        ref_min = ref_band.min()
        ref_max = ref_band.max()
        if ref_min == ref_max:
            warnings.warn(
                f"Band {j}: reference band has constant value "
                f"{ref_min}; skipping clipping of normalized band.",
                stacklevel=2,
            )
        else:
            norm_img[:, :, j] = np.clip(norm_img[:, :, j], ref_min, ref_max)

    return norm_img, rmse, r_adj


# ---------------------------------------------------------------------------
# Planet Labs PIF-based Radiometric Normalization
# ---------------------------------------------------------------------------


def _robust_fit(
    candidate_data: np.ndarray,
    reference_data: np.ndarray,
) -> Tuple[float, float]:
    """Cascading Huber / RANSAC robust regression (Planet Labs method).

    Tries progressively less-strict HuberRegressor epsilon values,
    falling back to RANSAC if all fail.

    Args:
        candidate_data: 1-D array of candidate pixel values.
        reference_data: 1-D array of reference pixel values.

    Returns:
        Tuple of (gain, offset) for ``reference = gain * candidate + offset``.
    """
    from sklearn.linear_model import HuberRegressor, RANSACRegressor

    X = candidate_data.reshape(-1, 1)
    y = reference_data

    for epsilon in (1.01, 1.05, 1.1, 1.35):
        try:
            model = HuberRegressor(epsilon=epsilon, max_iter=10000)
            model.fit(X, y)
            return float(model.coef_[0]), float(model.intercept_)
        except Exception:
            continue

    # Final fallback: RANSAC
    model = RANSACRegressor(max_trials=10000)
    model.fit(X, y)
    return float(model.estimator_.coef_[0]), float(model.estimator_.intercept_)


def _pca_pif_single_band(
    candidate_data: np.ndarray,
    reference_data: np.ndarray,
    threshold: float = 30.0,
) -> np.ndarray:
    """PCA-based pseudo-invariant feature detection for one band.

    Fits PCA to the (candidate, reference) scatter, then keeps pixels
    whose perpendicular distance from the first principal component is
    within *threshold*.

    Args:
        candidate_data: 1-D valid candidate pixels.
        reference_data: 1-D valid reference pixels.
        threshold: Max distance from PC1 to accept as PIF.

    Returns:
        Boolean array (length = input length); True → PIF.
    """
    from sklearn.decomposition import PCA as _PCA

    X = np.column_stack([candidate_data, reference_data])
    pca = _PCA(n_components=2).fit(X)
    minor = pca.transform(X)[:, 1]
    return np.abs(minor) <= threshold


def _planet_normalize(
    sub_img: np.ndarray,
    ref_img: np.ndarray,
    pif_method: str = "pca",
    pif_threshold: float = 30.0,
    transform_method: str = "robust",
    downsample_factor: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Planet Labs PIF-based per-band radiometric normalization.

    Pipeline: PIF detection → transformation estimation → application.

    Args:
        sub_img: Subject image (H, W, B) float64.
        ref_img: Reference image (H, W, B) float64.
        pif_method: ``"pca"`` (PCA filter) or ``"robust"`` (Huber residual
            filter). Default ``"pca"``.
        pif_threshold: Distance threshold for PIF retention.
            For PCA: perpendicular distance from PC1 (default 30).
            For robust: residual distance from fit line (default 100).
        transform_method: ``"linear_relationship"`` (std-ratio),
            ``"ols"`` (scipy linregress) or ``"robust"`` (Huber cascade).
            Default ``"robust"``.
        downsample_factor: Spatial reduction factor for PIF detection and
            transformation estimation. ``1`` = full resolution (default).
            Higher values (e.g. ``4``, ``10``) reduce memory usage at the
            cost of using fewer pixels for parameter estimation; the
            resulting gain/offset is still applied to the full-resolution
            image.

    Returns:
        Tuple of (normalized_image, gains, offsets, pif_counts).
    """
    from scipy.stats import linregress as _linregress
    from skimage.transform import resize as _resize

    num_bands = sub_img.shape[2]
    norm_img = np.zeros_like(sub_img)
    gains = np.zeros(num_bands, dtype=np.float64)
    offsets = np.zeros(num_bands, dtype=np.float64)
    pif_counts = np.zeros(num_bands, dtype=np.int64)

    # Downsample for PIF detection / transformation estimation
    ds = max(int(downsample_factor), 1)
    if ds > 1:
        sub_ds = sub_img[::ds, ::ds, :]
        ref_ds = ref_img[::ds, ::ds, :]
    else:
        sub_ds = sub_img
        ref_ds = ref_img

    # If images differ in spatial size, resample reference to subject shape
    # for PIF detection only (transformation is then applied to original subject)
    sub_h, sub_w = sub_ds.shape[:2]
    ref_h, ref_w = ref_ds.shape[:2]
    if sub_h != ref_h or sub_w != ref_w:
        ref_img_pif = _resize(
            ref_ds, (sub_h, sub_w, num_bands),
            order=1, mode='reflect', anti_aliasing=True, preserve_range=True
        )
    else:
        ref_img_pif = ref_ds

    for b in range(num_bands):
        sub_flat = sub_ds[:, :, b].ravel()
        ref_flat = ref_img_pif[:, :, b].ravel()

        # Valid mask: both non-zero
        valid = (sub_flat > 0) & (ref_flat > 0)
        sub_valid = sub_flat[valid]
        ref_valid = ref_flat[valid]

        if len(sub_valid) < 10:
            warnings.warn(
                f"Band {b}: too few valid pixels ({len(sub_valid)}); "
                "keeping band unchanged.",
                stacklevel=2,
            )
            norm_img[:, :, b] = sub_img[:, :, b]
            gains[b] = 1.0
            continue

        # --- PIF detection ---
        if pif_method == "pca":
            pif_mask = _pca_pif_single_band(
                sub_valid, ref_valid, pif_threshold
            )
        elif pif_method == "robust":
            g_tmp, o_tmp = _robust_fit(sub_valid, ref_valid)
            residuals = (
                np.abs(g_tmp * sub_valid - ref_valid + o_tmp)
                / np.sqrt(1 + g_tmp ** 2)
            )
            pif_mask = residuals < pif_threshold
        else:
            raise ValueError(
                f"Unknown pif_method {pif_method!r}; use 'pca' or 'robust'."
            )

        sub_pifs = sub_valid[pif_mask]
        ref_pifs = ref_valid[pif_mask]
        pif_counts[b] = len(sub_pifs)

        if len(sub_pifs) < 2:
            warnings.warn(
                f"Band {b}: too few PIFs ({len(sub_pifs)}); "
                "keeping band unchanged.",
                stacklevel=2,
            )
            norm_img[:, :, b] = sub_img[:, :, b]
            gains[b] = 1.0
            continue

        # --- Transformation estimation ---
        if transform_method == "linear_relationship":
            c_std = float(np.std(sub_pifs))
            r_std = float(np.std(ref_pifs))
            gain = r_std / c_std if c_std > 0 else 1.0
            offset = float(np.mean(ref_pifs)) - gain * float(np.mean(sub_pifs))
        elif transform_method == "ols":
            gain, offset, _, _, _ = _linregress(sub_pifs, ref_pifs)
            gain, offset = float(gain), float(offset)
        elif transform_method == "robust":
            gain, offset = _robust_fit(sub_pifs, ref_pifs)
        else:
            raise ValueError(
                f"Unknown transform_method {transform_method!r}; "
                "use 'linear_relationship', 'ols', or 'robust'."
            )

        gains[b] = gain
        offsets[b] = offset

        # Apply and clip to valid reference range
        norm_band = offset + gain * sub_img[:, :, b]
        ref_vals = ref_img[:, :, b]
        ref_nz = ref_vals[ref_vals > 0]
        if len(ref_nz) > 0:
            norm_band = np.clip(norm_band, 0, ref_nz.max())
        norm_img[:, :, b] = norm_band

    return norm_img, gains, offsets, pif_counts


def _block_size_from_memory(
    num_bands: int,
    dtype_bytes: int = 8,
    max_tile_memory_mb: float = 512,
    num_tiles: int = 2,
) -> int:
    """Compute the largest square block size that fits within a memory budget.

    During tiled I/O, *num_tiles* tile buffers of shape
    ``(num_bands, block_size, block_size)`` are held simultaneously.

    Args:
        num_bands: Number of raster bands.
        dtype_bytes: Bytes per pixel element (8 for float64).
        max_tile_memory_mb: Memory budget in MiB for tile buffers.
        num_tiles: Number of concurrent tile arrays (default 2: subject +
            reference).

    Returns:
        Block size in pixels (multiple of 16, at least 256).
    """
    budget_bytes = max_tile_memory_mb * 1024 * 1024
    # Each tile: num_bands * block_size^2 * dtype_bytes
    # Total: num_tiles * num_bands * block_size^2 * dtype_bytes
    pixel_cost = num_tiles * num_bands * dtype_bytes
    if pixel_cost <= 0:
        return 2048
    block = int(np.sqrt(budget_bytes / pixel_cost))
    block = max(256, block)
    # Round down to nearest multiple of 16 (required for JPEG compression)
    return (block // 16) * 16


def _estimate_planet_pif_tiled(
    sub_path: str,
    ref_path: str,
    pif_method: str = "pca",
    pif_threshold: float = 30.0,
    transform_method: str = "robust",
    downsample_factor: int = 1,
    max_tile_memory_mb: float = 512,
    max_pif_samples: int = 500_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate Planet PIF normalization parameters via tiled I/O.

    Reads spatial tiles from both rasters without loading the entire images
    into memory.  PIFs are collected per tile, then the global gain/offset
    is estimated from the pooled PIF pixels.

    Args:
        sub_path: Path to subject raster.
        ref_path: Path to reference raster.
        pif_method: ``"pca"`` or ``"robust"``.
        pif_threshold: Distance threshold for PIF retention.
        transform_method: ``"linear_relationship"``, ``"ols"``, or
            ``"robust"``.
        downsample_factor: Spatial stride within each tile (default 1).
        max_tile_memory_mb: Memory budget in MiB for tile buffers.  The
            block size is calculated automatically so that two concurrent
            tiles (subject + reference) fit within this limit. Default 512.
        max_pif_samples: Cap on PIF samples collected per band to bound
            memory.

    Returns:
        Tuple of (gains, offsets, pif_counts, ref_clip_max) — all arrays
        of length B (number of bands).
    """
    from scipy.stats import linregress as _linregress

    with rasterio.open(sub_path) as sub_src, \
         rasterio.open(ref_path) as ref_src:

        num_bands = sub_src.count
        ds = max(int(downsample_factor), 1)

        block_size = _block_size_from_memory(
            num_bands=num_bands,
            dtype_bytes=8,
            max_tile_memory_mb=max_tile_memory_mb,
            num_tiles=2,
        )

        # Per-band collectors
        pif_sub: List[List[np.ndarray]] = [[] for _ in range(num_bands)]
        pif_ref: List[List[np.ndarray]] = [[] for _ in range(num_bands)]
        pif_total = np.zeros(num_bands, dtype=np.int64)
        ref_max = np.full(num_bands, -np.inf, dtype=np.float64)

        # --- Pass 1: tile-by-tile PIF collection ---
        for j in range(0, sub_src.height, block_size):
            for i in range(0, sub_src.width, block_size):
                win_h = min(block_size, sub_src.height - j)
                win_w = min(block_size, sub_src.width - i)
                window = Window(i, j, win_w, win_h)

                sub_tile = sub_src.read(window=window).astype(np.float64)

                # Read matching geographic region from reference
                bounds = rasterio.windows.bounds(window, sub_src.transform)
                ref_window = rasterio.windows.from_bounds(
                    *bounds, ref_src.transform,
                )
                ref_tile = ref_src.read(
                    window=ref_window,
                    out_shape=(num_bands, win_h, win_w),
                    resampling=Resampling.bilinear,
                    boundless=True,
                    fill_value=0,
                ).astype(np.float64)

                # Downsample within tile
                if ds > 1:
                    sub_tile = sub_tile[:, ::ds, ::ds]
                    ref_tile = ref_tile[:, ::ds, ::ds]

                for b in range(num_bands):
                    # Track reference max
                    ref_b = ref_tile[b].ravel()
                    ref_nz = ref_b[ref_b > 0]
                    if len(ref_nz) > 0:
                        ref_max[b] = max(ref_max[b], float(ref_nz.max()))

                    # Skip band if we have enough PIFs already
                    if pif_total[b] >= max_pif_samples:
                        continue

                    sub_flat = sub_tile[b].ravel()
                    ref_flat = ref_tile[b].ravel()
                    valid = (sub_flat > 0) & (ref_flat > 0)
                    sub_valid = sub_flat[valid]
                    ref_valid = ref_flat[valid]

                    if len(sub_valid) < 10:
                        continue

                    # PIF detection
                    if pif_method == "pca":
                        pif_mask = _pca_pif_single_band(
                            sub_valid, ref_valid, pif_threshold,
                        )
                    elif pif_method == "robust":
                        g_tmp, o_tmp = _robust_fit(sub_valid, ref_valid)
                        residuals = (
                            np.abs(g_tmp * sub_valid - ref_valid + o_tmp)
                            / np.sqrt(1 + g_tmp ** 2)
                        )
                        pif_mask = residuals < pif_threshold
                    else:
                        raise ValueError(
                            f"Unknown pif_method {pif_method!r}; "
                            "use 'pca' or 'robust'."
                        )

                    sp = sub_valid[pif_mask]
                    rp = ref_valid[pif_mask]
                    if len(sp) == 0:
                        continue
                    remaining = max_pif_samples - pif_total[b]
                    pif_sub[b].append(sp[:remaining])
                    pif_ref[b].append(rp[:remaining])
                    pif_total[b] += min(len(sp), remaining)

    # --- Estimate global per-band transforms ---
    gains = np.ones(num_bands, dtype=np.float64)
    offsets = np.zeros(num_bands, dtype=np.float64)
    pif_counts = pif_total.copy()

    for b in range(num_bands):
        if not pif_sub[b]:
            warnings.warn(
                f"Band {b}: no PIFs found; keeping band unchanged.",
                stacklevel=2,
            )
            continue
        sub_pifs = np.concatenate(pif_sub[b])
        ref_pifs = np.concatenate(pif_ref[b])
        if len(sub_pifs) < 2:
            warnings.warn(
                f"Band {b}: too few PIFs ({len(sub_pifs)}); "
                "keeping band unchanged.",
                stacklevel=2,
            )
            continue

        if transform_method == "linear_relationship":
            c_std = float(np.std(sub_pifs))
            r_std = float(np.std(ref_pifs))
            gain = r_std / c_std if c_std > 0 else 1.0
            offset = float(np.mean(ref_pifs)) - gain * float(
                np.mean(sub_pifs)
            )
        elif transform_method == "ols":
            gain, offset, _, _, _ = _linregress(sub_pifs, ref_pifs)
            gain, offset = float(gain), float(offset)
        elif transform_method == "robust":
            gain, offset = _robust_fit(sub_pifs, ref_pifs)
        else:
            raise ValueError(
                f"Unknown transform_method {transform_method!r}; "
                "use 'linear_relationship', 'ols', or 'robust'."
            )
        gains[b] = gain
        offsets[b] = offset

    ref_clip_max = np.where(
        ref_max > -np.inf, ref_max, np.finfo(np.float64).max,
    )
    return gains, offsets, pif_counts, ref_clip_max


def _plot_radiometric_normalization_histogram(
    subject: np.ndarray,
    normalized: np.ndarray,
    reference: np.ndarray,
    output_path: str,
    title: str = "Radiometric Normalization",
) -> None:
    """Generate per-band before/after histogram comparison figure.

    Args:
        subject: Subject image (H, W, B) before normalization.
        normalized: Normalized subject image (H, W, B).
        reference: Reference image (H, W, B).
        output_path: Path to save the PNG figure.
        title: Figure title.
    """
    num_bands = subject.shape[2]
    fig, axes = plt.subplots(
        num_bands, 2, figsize=(14, 4 * num_bands), squeeze=False,
    )
    fig.suptitle(title, fontsize=14, y=1.01)

    colors = {'subject': 'steelblue', 'normalized': 'darkorange', 'reference': 'seagreen'}

    for b in range(num_bands):
        ax_before = axes[b][0]
        ax_after = axes[b][1]
        ax_before.set_title(f"Band {b + 1} — Before Normalization")
        ax_after.set_title(f"Band {b + 1} — After Normalization")

        # Before: subject vs reference
        for data, label, color in [
            (subject[:, :, b], 'Subject', colors['subject']),
            (reference[:, :, b], 'Reference', colors['reference']),
        ]:
            vals = data.ravel()
            vals = vals[vals > 0]
            if len(vals) > 0:
                lo, hi = np.percentile(vals, [1, 99])
                ax_before.hist(
                    vals, bins=128, range=(lo, hi), density=True,
                    alpha=0.5, color=color, label=label,
                )

        # After: normalized vs reference
        for data, label, color in [
            (normalized[:, :, b], 'Normalized', colors['normalized']),
            (reference[:, :, b], 'Reference', colors['reference']),
        ]:
            vals = data.ravel()
            vals = vals[vals > 0]
            if len(vals) > 0:
                lo, hi = np.percentile(vals, [1, 99])
                ax_after.hist(
                    vals, bins=128, range=(lo, hi), density=True,
                    alpha=0.5, color=color, label=label,
                )

        ax_before.legend(fontsize=8)
        ax_after.legend(fontsize=8)
        ax_before.set_xlabel("Pixel value")
        ax_after.set_xlabel("Pixel value")
        ax_before.set_ylabel("Density")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_radiometric_histogram_report(
    subject: np.ndarray,
    normalized: np.ndarray,
    reference: np.ndarray,
    output_path: str,
    title: str = "Radiometric Normalization",
    n_bins: int = 256,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """Save histogram comparison and compute Bhattacharyya quality indices.

    Returns per-band metrics before and after normalization:
    - Bhattacharyya distance (lower is better)
    - Bhattacharyya similarity index (higher is better, in [0, 1])
    """
    _plot_radiometric_normalization_histogram(
        subject=subject,
        normalized=normalized,
        reference=reference,
        output_path=output_path,
        title=title,
    )

    before_distance = _bhattacharyya_distance(subject, reference, n_bins=n_bins)
    after_distance = _bhattacharyya_distance(normalized, reference, n_bins=n_bins)

    before_similarity = np.exp(-before_distance)
    after_similarity = np.exp(-after_distance)

    # Guard against +/-inf from empty histograms
    before_similarity = np.where(np.isfinite(before_similarity), before_similarity, 0.0)
    after_similarity = np.where(np.isfinite(after_similarity), after_similarity, 0.0)

    report = {
        "bhattacharyya_distance_before": before_distance,
        "bhattacharyya_distance_after": after_distance,
        "bhattacharyya_similarity_before": before_similarity,
        "bhattacharyya_similarity_after": after_similarity,
        "bhattacharyya_distance_improvement": before_distance - after_distance,
        "bhattacharyya_similarity_gain": after_similarity - before_similarity,
    }

    if verbose:
        print("Histogram saved:", output_path)
        print("Bhattacharyya metrics by band:")
        for i in range(len(before_distance)):
            print(
                "  Band "
                f"{i + 1}: "
                f"distance {before_distance[i]:.6f} -> {after_distance[i]:.6f}, "
                f"similarity {before_similarity[i]:.6f} -> {after_similarity[i]:.6f}"
            )

    return report


def _resolve_radiometric_inputs(
    subject_image: Union[str, np.ndarray],
    reference_image: Union[str, np.ndarray],
    output_path: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, Optional[dict], bool, bool]:
    """Load/validate subject and reference inputs for radiometric normalization."""
    profile = None
    is_file_sub = isinstance(subject_image, str)
    is_file_ref = isinstance(reference_image, str)

    if is_file_sub:
        sub_arr, profile = _load_raster(subject_image)
    else:
        sub_arr = np.asarray(subject_image, dtype=np.float64)
        if sub_arr.ndim != 3:
            raise ValueError(
                f"subject_image must be 3-D (H, W, B), got {sub_arr.ndim}-D"
            )

    if is_file_ref:
        ref_arr, _ = _load_raster(reference_image)
    else:
        ref_arr = np.asarray(reference_image, dtype=np.float64)
        if ref_arr.ndim != 3:
            raise ValueError(
                f"reference_image must be 3-D (H, W, B), got {ref_arr.ndim}-D"
            )

    if output_path is not None and profile is None:
        raise ValueError(
            "output_path requires subject_image to be a file path "
            "(not an array) so that spatial metadata is available."
        )

    if sub_arr.shape[2] != ref_arr.shape[2]:
        raise ValueError(
            f"Band count mismatch: subject has {sub_arr.shape[2]} bands, "
            f"reference has {ref_arr.shape[2]} bands."
        )

    if np.any(~np.isfinite(sub_arr)):
        warnings.warn(
            "subject_image contains NaN or infinite values; replacing with 0.",
            stacklevel=2,
        )
        sub_arr = np.nan_to_num(sub_arr, nan=0.0, posinf=0.0, neginf=0.0)

    if np.any(~np.isfinite(ref_arr)):
        warnings.warn(
            "reference_image contains NaN or infinite values; replacing with 0.",
            stacklevel=2,
        )
        ref_arr = np.nan_to_num(ref_arr, nan=0.0, posinf=0.0, neginf=0.0)

    return sub_arr, ref_arr, profile, is_file_sub, is_file_ref


def normalize_radiometric_lirrn(
    subject_image: Union[str, np.ndarray],
    reference_image: Union[str, np.ndarray],
    output_path: Optional[str] = None,
    p_n: int = 500,
    num_quantisation_classes: int = 3,
    num_sampling_rounds: int = 3,
    subsample_ratio: float = 0.1,
    random_state: Optional[Union[int, np.random.Generator]] = None,
    save_histogram: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Run LIRRN normalization with optional histogram/Bhattacharyya reporting."""
    if p_n < 1:
        raise ValueError(f"p_n must be >= 1, got {p_n}")
    if num_sampling_rounds < 1:
        raise ValueError(
            f"num_sampling_rounds must be >= 1, got {num_sampling_rounds}"
        )
    if subsample_ratio <= 0 or subsample_ratio > 1:
        raise ValueError(
            f"subsample_ratio must be in (0, 1], got {subsample_ratio}"
        )

    # Auto-generate output path from subject filename if not provided
    if output_path is None and isinstance(subject_image, str):
        output_path = str(
            Path(subject_image).parent / (Path(subject_image).stem + "_norm.tif")
        )

    sub_arr, ref_arr, profile, _, _ = _resolve_radiometric_inputs(
        subject_image, reference_image, output_path,
    )

    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    t0 = time.time()
    norm_img, rmse, r_adj = _lirrn(
        p_n,
        sub_arr,
        ref_arr,
        num_quantisation_classes=num_quantisation_classes,
        num_sampling_rounds=num_sampling_rounds,
        subsample_ratio=subsample_ratio,
        rng=rng,
    )
    metrics: Dict[str, np.ndarray] = {"rmse": rmse, "r_adj": r_adj}

    if output_path is not None:
        _save_raster(output_path, norm_img, profile)
        metrics["output_path"] = output_path

    if save_histogram and output_path is not None:
        hist_path = str(Path(output_path).with_suffix("")) + "_histogram.png"
        hist_metrics = generate_radiometric_histogram_report(
            subject=sub_arr,
            normalized=norm_img,
            reference=ref_arr,
            output_path=hist_path,
            title=f"Radiometric Normalization (LIRRN): {Path(output_path).stem}",
            verbose=verbose,
        )
        metrics.update(hist_metrics)

    if verbose:
        elapsed = round(time.time() - t0, 2)
        print(f"LIRRN normalization complete in {elapsed}s")
        for i, (r, r2) in enumerate(zip(metrics['rmse'], metrics['r_adj'])):
            print(f"  Band {i+1}:  RMSE = {r:.4f},  Adj R\u00b2 = {r2:.4f}")
        if 'bhattacharyya_similarity_after' in metrics:
            for i, s in enumerate(metrics['bhattacharyya_similarity_after']):
                print(f"  Band {i+1}:  Bhattacharyya similarity (after) = {s:.4f}")
        if output_path is not None:
            print(f"Saved -> {output_path}")

    return norm_img, metrics


def normalize_radiometric_planet(
    subject_image: Union[str, np.ndarray],
    reference_image: Union[str, np.ndarray],
    output_path: Optional[str] = None,
    pif_method: str = "pca",
    pif_threshold: Optional[float] = None,
    transform_method: str = "robust",
    downsample_factor: int = 1,
    max_tile_memory_mb: float = 512,
    save_histogram: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Run Planet PIF normalization with optional histogram/Bhattacharyya reporting."""
    # Auto-generate output path from subject filename if not provided
    if output_path is None and isinstance(subject_image, str):
        output_path = str(
            Path(subject_image).parent / (Path(subject_image).stem + "_norm.tif")
        )

    is_file_sub = isinstance(subject_image, str)
    is_file_ref = isinstance(reference_image, str)
    pif_thresh = pif_threshold if pif_threshold is not None else (30.0 if pif_method == "pca" else 100.0)

    if verbose:
        print("Planet PIF normalization configuration:")
        print(f"  pif_method={pif_method}, pif_threshold={pif_thresh}")
        print(f"  transform_method={transform_method}, downsample_factor={downsample_factor}")
        print(f"  max_tile_memory_mb={max_tile_memory_mb}")

    t0 = time.time()

    # File-path mode: tiled estimation + tiled write (memory-safe)
    if is_file_sub and is_file_ref:
        if not os.path.isfile(subject_image):
            raise FileNotFoundError(f"Raster file not found: {subject_image}")
        if not os.path.isfile(reference_image):
            raise FileNotFoundError(f"Raster file not found: {reference_image}")

        with rasterio.open(subject_image) as ssrc, rasterio.open(reference_image) as rsrc:
            if ssrc.count != rsrc.count:
                raise ValueError(
                    f"Band count mismatch: subject has {ssrc.count} bands, reference has {rsrc.count} bands."
                )

            gains, offsets, pif_counts, ref_clip_max = _estimate_planet_pif_tiled(
                subject_image,
                reference_image,
                pif_method=pif_method,
                pif_threshold=pif_thresh,
                transform_method=transform_method,
                downsample_factor=downsample_factor,
                max_tile_memory_mb=max_tile_memory_mb,
            )

            metrics: Dict[str, np.ndarray] = {
                "gains": gains,
                "offsets": offsets,
                "pif_counts": pif_counts,
            }

            if output_path is not None:
                write_block = _block_size_from_memory(
                    num_bands=ssrc.count,
                    dtype_bytes=8,
                    max_tile_memory_mb=max_tile_memory_mb,
                    num_tiles=2,
                )

                clip_min = np.zeros_like(gains)
                _apply_linear_transform(
                    subject_image,
                    output_path,
                    slopes=gains,
                    intercepts=offsets,
                    clip_min=clip_min,
                    clip_max=ref_clip_max,
                    block_size=write_block,
                )
                metrics["output_path"] = output_path

                if save_histogram:
                    # Use downsampled arrays to keep reporting lightweight.
                    sub_ds, _, _ = _load_raster_downsampled(subject_image, factor=10)
                    ref_ds, _, _ = _load_raster_downsampled(reference_image, factor=10)
                    norm_ds, _, _ = _load_raster_downsampled(output_path, factor=10)
                    hist_path = str(Path(output_path).with_suffix("")) + "_histogram.png"
                    hist_metrics = generate_radiometric_histogram_report(
                        subject=sub_ds,
                        normalized=norm_ds,
                        reference=ref_ds,
                        output_path=hist_path,
                        title=f"Radiometric Normalization (Planet PIF): {Path(output_path).stem}",
                        verbose=verbose,
                    )
                    metrics.update(hist_metrics)

            # Preserve existing behavior for tiled mode (image not held in memory).
            norm_img = np.empty((0, 0, 0), dtype=np.float64)

    if verbose:
        elapsed = round(time.time() - t0, 2)
        print(f"Planet PIF normalization complete in {elapsed}s")
        for i, (g, o, n) in enumerate(zip(metrics['gains'], metrics['offsets'], metrics['pif_counts'])):
            print(f"  Band {i+1}:  gain = {g:.4f},  offset = {o:.4f},  PIFs = {n}")
        if 'bhattacharyya_similarity_after' in metrics:
            for i, s in enumerate(metrics['bhattacharyya_similarity_after']):
                print(f"  Band {i+1}:  Bhattacharyya similarity (after) = {s:.4f}")
        if output_path is not None:
            print(f"Saved -> {output_path}")

    if is_file_sub and is_file_ref:
        return norm_img, metrics

    # Array or mixed mode: load in memory, then run _planet_normalize directly.
    sub_arr, ref_arr, profile, _, _ = _resolve_radiometric_inputs(
        subject_image, reference_image, output_path,
    )

    norm_img, gains, offsets, pif_counts = _planet_normalize(
        sub_arr,
        ref_arr,
        pif_method=pif_method,
        pif_threshold=pif_thresh,
        transform_method=transform_method,
        downsample_factor=downsample_factor,
    )
    metrics = {
        "gains": gains,
        "offsets": offsets,
        "pif_counts": pif_counts,
    }

    if output_path is not None:
        _save_raster(output_path, norm_img, profile)
        metrics["output_path"] = output_path

    if save_histogram and output_path is not None:
        hist_path = str(Path(output_path).with_suffix("")) + "_histogram.png"
        hist_metrics = generate_radiometric_histogram_report(
            subject=sub_arr,
            normalized=norm_img,
            reference=ref_arr,
            output_path=hist_path,
            title=f"Radiometric Normalization (Planet PIF): {Path(output_path).stem}",
            verbose=verbose,
        )
        metrics.update(hist_metrics)

    if verbose:
        elapsed = round(time.time() - t0, 2)
        print(f"Planet PIF normalization complete in {elapsed}s")
        for i, (g, o, n) in enumerate(zip(metrics['gains'], metrics['offsets'], metrics['pif_counts'])):
            print(f"  Band {i+1}:  gain = {g:.4f},  offset = {o:.4f},  PIFs = {n}")
        if 'bhattacharyya_similarity_after' in metrics:
            for i, s in enumerate(metrics['bhattacharyya_similarity_after']):
                print(f"  Band {i+1}:  Bhattacharyya similarity (after) = {s:.4f}")
        if output_path is not None:
            print(f"Saved -> {output_path}")

    return norm_img, metrics


def normalize_radiometric(
    subject_image: Union[str, np.ndarray],
    reference_image: Union[str, np.ndarray],
    output_path: Optional[str] = None,
    method: str = "lirrn",
    p_n: int = 500,
    num_quantisation_classes: int = 3,
    num_sampling_rounds: int = 3,
    subsample_ratio: float = 0.1,
    random_state: Optional[Union[int, np.random.Generator]] = None,
    pif_method: str = "pca",
    pif_threshold: Optional[float] = None,
    transform_method: str = "robust",
    downsample_factor: int = 1,
    max_tile_memory_mb: float = 512,
    save_histogram: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Normalize subject image radiometry to match a reference image.

    Adjusts brightness and contrast of the subject image so that its pixel
    value distribution matches the reference image. This is essential for
    multi-temporal analysis where images are acquired under different
    atmospheric conditions, sensor calibrations, or illumination angles.

    Supports two methods:

    - ``"lirrn"`` — Location-Independent Relative Radiometric
      Normalization.  Multi-Otsu thresholding + stratified sampling +
      per-band OLS regression.  Reference: doi:10.3390/s24072272
    - ``"planet_pif"`` — Planet Labs PIF-based normalization.
      PCA or robust-fit pseudo-invariant feature detection followed by
      linear-relationship, OLS, or Huber/RANSAC regression per band.

    Args:
        subject_image: Path to the subject GeoTIFF or numpy array with
            shape (H, W, B). The image to be normalized.
        reference_image: Path to the reference GeoTIFF or numpy array with
            shape (H, W, B). The target radiometry to match.
        output_path: Path to save the normalized image as GeoTIFF. Only
            applicable when *subject_image* is a file path (so spatial
            metadata is available). If None, the array is returned without
            saving. Default: None.
        method: ``"lirrn"`` or ``"planet_pif"``. Default ``"lirrn"``.
        p_n: (LIRRN) Samples per quantization level. Default: 500.
        num_quantisation_classes: (LIRRN) Brightness strata. Default: 3.
        num_sampling_rounds: (LIRRN) Sampling rounds. Default: 3.
        subsample_ratio: (LIRRN) Fraction of candidates for regression.
            Default: 0.1.
        random_state: Seed or numpy Generator for reproducible results.
            Default: None.
        pif_method: (planet_pif) PIF detection method — ``"pca"`` or
            ``"robust"``. Default ``"pca"``.
        pif_threshold: (planet_pif) Distance threshold for PIF retention.
            Default ``None`` → 30 for PCA, 100 for robust.
        transform_method: (planet_pif) Transformation method —
            ``"linear_relationship"``, ``"ols"``, or ``"robust"``.
            Default ``"robust"``.
        downsample_factor: (planet_pif) Spatial downsampling factor for PIF
            detection and transformation estimation. ``1`` = full resolution
            (default). Higher values (e.g. ``4``, ``10``) reduce memory
            usage; the gain/offset is still applied to the full image.
        max_tile_memory_mb: (planet_pif) Memory budget in MiB for tile
            buffers during tiled windowed I/O.  The tile size is computed
            automatically so that two concurrent tiles (subject + reference)
            fit within this limit.  When file paths are provided, PIF
            detection is done tile-by-tile so the full images are never
            loaded simultaneously.  Ignored when numpy arrays are passed
            (uses ``downsample_factor`` instead). Default ``512``.
        save_histogram: Save a before/after histogram comparison figure.
            The histogram is saved to ``<output_path_stem>_histogram.png``.
            If ``output_path`` is None, no histogram is saved regardless
            of this setting. Default ``True``.

    Returns:
        Tuple of (normalized_image, metrics) where:
            - normalized_image: numpy array (H, W, B) float64.
            - metrics: dict whose keys depend on method.
              LIRRN: ``"rmse"``, ``"r_adj"``.
              planet_pif: ``"gains"``, ``"offsets"``, ``"pif_counts"``.

    Raises:
        ValueError: If *method* is not supported.
        ValueError: If subject and reference have different band counts.
        ValueError: If input arrays are not 3-dimensional.
        ValueError: If *output_path* is set but *subject_image* is an array.
        FileNotFoundError: If file paths do not point to existing files.

    Examples:
        Normalize a satellite image using file paths:

        >>> from geoai import normalize_radiometric
        >>> norm_img, metrics = normalize_radiometric(
        ...     "subject.tif",
        ...     "reference.tif",
        ...     output_path="normalized.tif",
        ... )
        >>> print(f"RMSE per band: {metrics['rmse']}")

        Normalize using numpy arrays:

        >>> import numpy as np
        >>> subject = np.random.rand(100, 100, 4)
        >>> reference = np.random.rand(120, 120, 4)
        >>> norm_img, metrics = normalize_radiometric(subject, reference)
        >>> norm_img.shape
        (100, 100, 4)

    Note:
        The subject and reference images must have the same number of bands
        but may have different spatial dimensions (height and width).
    """
    warnings.warn(
        "normalize_radiometric(method=...) is retained for backward compatibility. "
        "Prefer normalize_radiometric_lirrn(...) or normalize_radiometric_planet(...).",
        DeprecationWarning,
        stacklevel=2,
    )

    if method == "lirrn":
        return normalize_radiometric_lirrn(
            subject_image=subject_image,
            reference_image=reference_image,
            output_path=output_path,
            p_n=p_n,
            num_quantisation_classes=num_quantisation_classes,
            num_sampling_rounds=num_sampling_rounds,
            subsample_ratio=subsample_ratio,
            random_state=random_state,
            save_histogram=save_histogram,
            verbose=True,
        )

    if method == "planet_pif":
        return normalize_radiometric_planet(
            subject_image=subject_image,
            reference_image=reference_image,
            output_path=output_path,
            pif_method=pif_method,
            pif_threshold=pif_threshold,
            transform_method=transform_method,
            downsample_factor=downsample_factor,
            max_tile_memory_mb=max_tile_memory_mb,
            save_histogram=save_histogram,
            verbose=True,
        )

    raise ValueError(
        f"Unsupported normalization method {method!r}. "
        "Choose from ('lirrn', 'planet_pif')."
    )


# ---------------------------------------------------------------------------
# Multi-Temporal Iterative Normalization
# ---------------------------------------------------------------------------

def _load_raster_downsampled(
    filepath: str,
    factor: int = 10,
) -> Tuple[np.ndarray, dict, float]:
    """Load a raster at reduced resolution for parameter estimation.

    Args:
        filepath: Path to the raster file.
        factor: Downsample factor (e.g., 10 → every 10th pixel).

    Returns:
        Tuple of (image_array, profile, actual_factor) where image_array
        has shape (H//factor, W//factor, B) in float64.
    """
    with rasterio.open(filepath) as src:
        h, w = src.height, src.width
        new_h = max(1, h // factor)
        new_w = max(1, w // factor)
        img = src.read(out_shape=(src.count, new_h, new_w)).astype(np.float64)
        profile = src.profile.copy()
    img = np.moveaxis(img, 0, -1)  # (B, H, W) → (H, W, B)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    actual_factor = h / new_h
    return img, profile, actual_factor


def _estimate_lirrn_parameters(
    sub_img: np.ndarray,
    ref_img: np.ndarray,
    p_n: int = 500,
    num_quantisation_classes: int = 3,
    num_sampling_rounds: int = 3,
    subsample_ratio: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run LIRRN on (typically downsampled) images and extract per-band
    linear regression parameters instead of the full normalised image.

    Args:
        sub_img: Subject image (H, W, B) float64.
        ref_img: Reference image (H, W, B) float64.
        p_n, num_quantisation_classes, num_sampling_rounds, subsample_ratio,
        rng: Same semantics as ``_lirrn``.

    Returns:
        Tuple of (slopes, intercepts, rmse, r_adj), each of shape (B,).
    """
    if rng is None:
        rng = np.random.default_rng()

    num_bands = sub_img.shape[2]
    id_indices = rng.integers(0, p_n, size=max(1, round(subsample_ratio * p_n)))

    slopes = np.ones(num_bands, dtype=np.float64)
    intercepts = np.zeros(num_bands, dtype=np.float64)
    rmse = np.zeros(num_bands, dtype=np.float64)
    r_adj = np.zeros(num_bands, dtype=np.float64)

    # Quantize each band into brightness levels via multi-Otsu
    sub_labels = np.zeros_like(sub_img, dtype=np.int32)
    ref_labels = np.zeros_like(ref_img, dtype=np.int32)

    for j in range(num_bands):
        for img_arr, labels in [(sub_img, sub_labels), (ref_img, ref_labels)]:
            nonzero = img_arr[:, :, j][img_arr[:, :, j] != 0]
            if len(nonzero) > 0:
                try:
                    thresh = threshold_multiotsu(
                        nonzero, classes=num_quantisation_classes
                    )
                    labels[:, :, j] = np.digitize(img_arr[:, :, j], bins=thresh) + 1
                except ValueError:
                    labels[:, :, j] = 1

    for j in range(num_bands):
        sub_list, ref_list = [], []
        for level in range(1, num_quantisation_classes + 1):
            a = np.where(ref_labels[:, :, j] == level, ref_img[:, :, j], 0).ravel()
            b = np.where(sub_labels[:, :, j] == level, sub_img[:, :, j], 0).ravel()
            sub_s, ref_s = _sample_selection(p_n, a, b, id_indices, num_sampling_rounds)
            sub_list.append(sub_s)
            ref_list.append(ref_s)

        all_sub = np.concatenate(sub_list)
        all_ref = np.concatenate(ref_list)

        mask = (all_sub != 0) & (all_ref != 0)
        sub_clean = all_sub[mask]
        ref_clean = all_ref[mask]

        if len(sub_clean) < 2:
            warnings.warn(
                f"Band {j}: insufficient samples for regression; "
                "keeping identity transform (slope=1, intercept=0).",
                stacklevel=2,
            )
            continue

        X = sub_clean.reshape(-1, 1)
        y = ref_clean
        model = LinearRegression().fit(X, y)
        intercepts[j] = model.intercept_
        slopes[j] = model.coef_[0]

        y_pred = model.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        n = len(y)
        r_adj[j] = (1 - (1 - (1 - ss_res / ss_tot)) * (n - 1) / (n - 2)
                     if ss_tot != 0 and n > 2 else 0.0)
        rmse[j] = float(np.sqrt(np.mean((y - y_pred) ** 2)))

    return slopes, intercepts, rmse, r_adj


def _apply_linear_transform(
    input_path: str,
    output_path: str,
    slopes: np.ndarray,
    intercepts: np.ndarray,
    clip_min: Optional[np.ndarray] = None,
    clip_max: Optional[np.ndarray] = None,
    block_size: int = 1024,
) -> None:
    """Apply per-band linear transform to a raster via windowed I/O.

    For each band *b*: ``out[b] = intercepts[b] + slopes[b] * in[b]``,
    optionally clipped to ``[clip_min[b], clip_max[b]]``.

    Args:
        input_path: Source raster path.
        output_path: Destination path (GeoTIFF, LZW compressed).
        slopes: Per-band slope array of length B.
        intercepts: Per-band intercept array of length B.
        clip_min: Optional per-band minimum values.
        clip_max: Optional per-band maximum values.
        block_size: Tile size for windowed processing.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with rasterio.open(input_path) as src:
        src_dtype = src.dtypes[0]
        # Build a clean GTiff profile to avoid inheriting JP2-specific keys
        # and to preserve the source dtype (prevents unnecessary BigTIFF).
        out_profile: Dict[str, Any] = {
            "driver": "GTiff",
            "dtype": src_dtype,
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "crs": src.crs,
            "transform": src.transform,
            "tiled": True,
            "blockxsize": min(block_size, src.width),
            "blockysize": min(block_size, src.height),
        }
        if src.nodata is not None:
            out_profile["nodata"] = src.nodata
        _is_int = np.issubdtype(np.dtype(src_dtype), np.integer)
        # JPEG compression for uint8 (matches JP2-level sizes on aerial imagery).
        # Fall back to DEFLATE for other dtypes (JPEG is uint8-only in GeoTIFF).
        if src_dtype == "uint8":
            out_profile["compress"] = "jpeg"
            out_profile["jpeg_quality"] = 90
        else:
            out_profile["compress"] = "deflate"
            out_profile["predictor"] = 2 if _is_int else 3
            out_profile["zlevel"] = 6  # balanced speed/size (1=fast, 9=max)
        if _is_int:
            _dtype_min = float(np.iinfo(np.dtype(src_dtype)).min)
            _dtype_max = float(np.iinfo(np.dtype(src_dtype)).max)
        with rasterio.open(output_path, "w", **out_profile) as dst:
            for j in range(0, src.height, block_size):
                for i in range(0, src.width, block_size):
                    win_h = min(block_size, src.height - j)
                    win_w = min(block_size, src.width - i)
                    window = Window(i, j, win_w, win_h)
                    data = src.read(window=window).astype(np.float64)
                    for b in range(data.shape[0]):
                        data[b] = intercepts[b] + slopes[b] * data[b]
                        if clip_min is not None:
                            data[b] = np.maximum(data[b], clip_min[b])
                        if clip_max is not None:
                            data[b] = np.minimum(data[b], clip_max[b])
                        if _is_int:
                            data[b] = np.round(data[b]).clip(_dtype_min, _dtype_max)
                    dst.write(data.astype(src_dtype), window=window)


def _resample_raster(
    input_path: str,
    output_path: str,
    target_res: float,
    resampling_method: Resampling = Resampling.bilinear,
) -> str:
    """Resample a raster to a target resolution.

    Args:
        input_path: Source raster path.
        output_path: Destination path for the resampled raster.
        target_res: Target pixel size in the raster's CRS units (metres).
        resampling_method: Rasterio resampling algorithm.

    Returns:
        Path to the resampled raster.
    """
    with rasterio.open(input_path) as src:
        cur_res = abs(src.transform[0])
        if abs(cur_res - target_res) < 1e-6:
            # Already at target resolution — just copy
            import shutil
            shutil.copy2(input_path, output_path)
            return output_path

        transform, width, height = calculate_default_transform(
            src.crs, src.crs, src.width, src.height,
            *src.bounds,
            resolution=target_res,
        )
        out_profile = src.profile.copy()
        out_profile.update(
            transform=transform,
            width=width,
            height=height,
            compress="lzw",
        )
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with rasterio.open(output_path, "w", **out_profile) as dst:
            for band_idx in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=src.crs,
                    resampling=resampling_method,
                )
    return output_path


def _bhattacharyya_distance(
    img_a: np.ndarray,
    img_b: np.ndarray,
    n_bins: int = 256,
) -> np.ndarray:
    """Compute per-band Bhattacharyya distance between two images.

    Zero-valued pixels (nodata) are excluded from the histograms.

    Args:
        img_a: Image array (H, W, B) float64.
        img_b: Image array (H, W, B) float64.
        n_bins: Number of histogram bins.

    Returns:
        1-D array of distances, one per band.  A value of 0 means
        identical distributions; larger values indicate greater divergence.
    """
    num_bands = img_a.shape[2]
    distances = np.zeros(num_bands, dtype=np.float64)

    for b in range(num_bands):
        a_vals = img_a[:, :, b].ravel()
        b_vals = img_b[:, :, b].ravel()
        # Exclude nodata (zeros)
        a_vals = a_vals[a_vals != 0]
        b_vals = b_vals[b_vals != 0]

        if len(a_vals) == 0 or len(b_vals) == 0:
            distances[b] = float("inf")
            continue

        lo = min(a_vals.min(), b_vals.min())
        hi = max(a_vals.max(), b_vals.max())
        if lo == hi:
            distances[b] = 0.0
            continue

        hist_a, _ = np.histogram(a_vals, bins=n_bins, range=(lo, hi), density=True)
        hist_b, _ = np.histogram(b_vals, bins=n_bins, range=(lo, hi), density=True)

        # Normalise to sum=1 (probability vectors)
        hist_a = hist_a / (hist_a.sum() + 1e-12)
        hist_b = hist_b / (hist_b.sum() + 1e-12)

        bc = np.sum(np.sqrt(hist_a * hist_b))
        bc = np.clip(bc, 1e-12, 1.0)
        distances[b] = -np.log(bc)

    return distances


def _normalize_tile_group(
    tile_paths: Dict[str, str],
    reference_year: str,
    output_dir: str,
    p_n: int,
    num_quantisation_classes: int,
    num_sampling_rounds: int,
    subsample_ratio: float,
    rng: np.random.Generator,
    convergence_threshold: float,
    max_iterations: int,
    downsample_factor: int,
    quiet: bool,
) -> Dict[str, Any]:
    """Iteratively normalise a single spatial tile across years.

    Args:
        tile_paths: ``{year_label: raster_path}`` for one (x, y) tile.
        reference_year: Year label used as the initial radiometric anchor.
        output_dir: Base output directory; per-year sub-folders are created.
        p_n .. rng: LIRRN algorithm parameters.
        convergence_threshold: Max Bhattacharyya distance to stop.
        max_iterations: Safety cap on iteration count.
        downsample_factor: Factor for parameter estimation.
        quiet: Suppress progress messages.

    Returns:
        Dict with convergence history, output paths, and per-band metrics.
    """
    years = sorted(tile_paths.keys())
    tile_name = Path(tile_paths[reference_year]).stem

    if reference_year not in tile_paths:
        raise ValueError(
            f"Reference year '{reference_year}' not found in tile group "
            f"for tile {tile_name}. Available: {years}"
        )

    # --- Check resolution consistency; resample if needed ---
    ref_path = tile_paths[reference_year]
    with rasterio.open(ref_path) as src:
        ref_res = abs(src.transform[0])
        ref_bands = src.count

    # Build working copies (resample if necessary)
    work_dir = os.path.join(output_dir, "_work", tile_name)
    os.makedirs(work_dir, exist_ok=True)

    working_paths: Dict[str, str] = {}
    for yr, path in tile_paths.items():
        with rasterio.open(path) as src:
            cur_res = abs(src.transform[0])
            cur_bands = src.count
        if cur_bands != ref_bands:
            raise ValueError(
                f"Band count mismatch for tile {tile_name}: "
                f"'{yr}' has {cur_bands} bands but reference "
                f"'{reference_year}' has {ref_bands} bands."
            )
        if abs(cur_res - ref_res) > 1e-6:
            if not quiet:
                print(
                    f"  Resampling {yr} tile from {cur_res:.2f}m to "
                    f"{ref_res:.2f}m..."
                )
            resampled = os.path.join(work_dir, f"{yr}_resampled.tif")
            _resample_raster(path, resampled, ref_res)
            working_paths[yr] = resampled
        else:
            working_paths[yr] = path

    # --- Load downsampled images for parameter estimation ---
    def _load_ds(path):
        img, _, _ = _load_raster_downsampled(path, factor=downsample_factor)
        return img

    ds_images: Dict[str, np.ndarray] = {}
    for yr, path in working_paths.items():
        ds_images[yr] = _load_ds(path)

    # Store originals for histogram comparison
    ds_originals = {yr: img.copy() for yr, img in ds_images.items()}

    lirrn_kwargs = dict(
        p_n=p_n,
        num_quantisation_classes=num_quantisation_classes,
        num_sampling_rounds=num_sampling_rounds,
        subsample_ratio=subsample_ratio,
    )

    convergence_history: List[Dict[str, Any]] = []

    # Cumulative transforms: final pixel = intercept + slope * original
    cum_slopes = {yr: np.ones(ref_bands, dtype=np.float64) for yr in years}
    cum_intercepts = {yr: np.zeros(ref_bands, dtype=np.float64) for yr in years}

    def _update_cumulative(yr, new_slopes, new_intercepts):
        """Chain: new(cum(x)) = new_int + new_slope * (cum_int + cum_slope * x)
        = (new_int + new_slope * cum_int) + (new_slope * cum_slope) * x
        """
        cum_intercepts[yr] = new_intercepts + new_slopes * cum_intercepts[yr]
        cum_slopes[yr] = new_slopes * cum_slopes[yr]

    def _apply_ds(yr, slopes, intercepts):
        """Apply a linear transform to the downsampled image in-place."""
        for b in range(ref_bands):
            ds_images[yr][:, :, b] = (
                intercepts[b] + slopes[b] * ds_images[yr][:, :, b]
            )
        _update_cumulative(yr, slopes, intercepts)

    def _compute_max_distance():
        """Compute maximum pairwise Bhattacharyya distance across all pairs."""
        max_dist = 0.0
        pair_dists = {}
        for i, y1 in enumerate(years):
            for y2 in years[i + 1:]:
                d = _bhattacharyya_distance(ds_images[y1], ds_images[y2])
                pair_key = f"{y1}-{y2}"
                pair_dists[pair_key] = d.tolist()
                max_dist = max(max_dist, float(d.max()))
        return max_dist, pair_dists

    # --- Phase 1: Anchor pass — normalize all → reference ---
    if not quiet:
        print(f"  Phase 1: Anchor pass (all → {reference_year})")
    ref_ds = ds_images[reference_year]
    for yr in years:
        if yr == reference_year:
            continue
        slopes, intercepts, _, _ = _estimate_lirrn_parameters(
            ds_images[yr], ref_ds, rng=rng, **lirrn_kwargs
        )
        _apply_ds(yr, slopes, intercepts)

    max_d, pair_d = _compute_max_distance()
    convergence_history.append({
        "iteration": 0, "phase": "anchor",
        "max_bhattacharyya": max_d, "pairs": pair_d,
    })
    if not quiet:
        print(f"    Max Bhattacharyya distance: {max_d:.4f}")

    if max_d < convergence_threshold:
        if not quiet:
            print(f"  Converged after anchor pass.")
    else:
        # --- Phase 2: Reverse pass — reference → mean of all normalised targets ---
        if not quiet:
            print(f"  Phase 2: Reverse pass ({reference_year} → each)")
        target_years = [y for y in years if y != reference_year]
        if target_years:
            mean_target = np.mean([ds_images[y] for y in target_years], axis=0)
            slopes, intercepts, _, _ = _estimate_lirrn_parameters(
                ds_images[reference_year], mean_target, rng=rng, **lirrn_kwargs
            )
            _apply_ds(reference_year, slopes, intercepts)

        max_d, pair_d = _compute_max_distance()
        convergence_history.append({
            "iteration": 0, "phase": "reverse",
            "max_bhattacharyya": max_d, "pairs": pair_d,
        })
        if not quiet:
            print(f"    Max Bhattacharyya distance: {max_d:.4f}")

        if max_d >= convergence_threshold and not quiet:
            print(
                f"  Normalization complete. Max Bhattacharyya distance: {max_d:.4f} "
                f"(threshold: {convergence_threshold}). "
                f"Linear correction cannot reduce this further."
            )

    # --- Apply cumulative transforms to full-resolution rasters ---
    output_paths: Dict[str, str] = {}
    for yr in years:
        yr_out_dir = os.path.join(output_dir, yr)
        os.makedirs(yr_out_dir, exist_ok=True)
        out_name = Path(tile_paths[yr]).stem + "_norm.tif"
        out_path = os.path.join(yr_out_dir, out_name)

        # Determine per-band clip range from the reference original
        with rasterio.open(working_paths[reference_year]) as src:
            clip_max = np.zeros(ref_bands, dtype=np.float64)
            clip_min = np.zeros(ref_bands, dtype=np.float64)
            for b in range(ref_bands):
                band_data = src.read(b + 1).astype(np.float64)
                valid = band_data[band_data != 0]
                if len(valid) > 0:
                    clip_min[b] = 0.0
                    clip_max[b] = valid.max()
                else:
                    clip_min[b] = 0.0
                    clip_max[b] = 255.0

        if not quiet:
            print(f"  Writing {yr} → {out_path}")
        _apply_linear_transform(
            working_paths[yr],
            out_path,
            slopes=cum_slopes[yr],
            intercepts=cum_intercepts[yr],
            clip_min=clip_min,
            clip_max=clip_max,
        )
        output_paths[yr] = out_path

    return {
        "tile_name": tile_name,
        "convergence_history": convergence_history,
        "output_paths": output_paths,
        "ds_originals": ds_originals,
        "ds_normalized": {yr: img.copy() for yr, img in ds_images.items()},
    }


def _plot_normalization_comparison(
    ds_originals: Dict[str, np.ndarray],
    ds_normalized: Dict[str, np.ndarray],
    tile_name: str,
    output_path: str,
) -> None:
    """Generate per-band before/after histogram comparison figure.

    Args:
        ds_originals: ``{year: original_downsampled_image}``.
        ds_normalized: ``{year: normalised_downsampled_image}``.
        tile_name: Tile identifier for the figure title.
        output_path: Path to save the PNG figure.
    """
    years = sorted(ds_originals.keys())
    num_bands = ds_originals[years[0]].shape[2]
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(years), 3)))

    fig, axes = plt.subplots(
        num_bands, 2, figsize=(14, 4 * num_bands), squeeze=False,
    )
    fig.suptitle(f"Radiometric Normalization: {tile_name}", fontsize=14, y=1.01)

    for b in range(num_bands):
        ax_before = axes[b][0]
        ax_after = axes[b][1]
        ax_before.set_title(f"Band {b + 1} — Before")
        ax_after.set_title(f"Band {b + 1} — After")

        for idx, yr in enumerate(years):
            # Before
            vals_b = ds_originals[yr][:, :, b].ravel()
            vals_b = vals_b[vals_b != 0]
            if len(vals_b) > 0:
                lo, hi = np.percentile(vals_b, [1, 99])
                ax_before.hist(
                    vals_b, bins=128, range=(lo, hi), density=True,
                    alpha=0.5, color=colors[idx], label=yr,
                )
            # After
            vals_a = ds_normalized[yr][:, :, b].ravel()
            vals_a = vals_a[vals_a != 0]
            if len(vals_a) > 0:
                lo, hi = np.percentile(vals_a, [1, 99])
                ax_after.hist(
                    vals_a, bins=128, range=(lo, hi), density=True,
                    alpha=0.5, color=colors[idx], label=yr,
                )

        ax_before.legend(fontsize=8)
        ax_after.legend(fontsize=8)
        ax_before.set_xlabel("Pixel value")
        ax_after.set_xlabel("Pixel value")
        ax_before.set_ylabel("Density")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def normalize_multitemporal(
    reference_path: str,
    target_paths: Union[str, List[str], Dict[str, str]],
    output_dir: str,
    method: str = "lirrn",
    # LIRRN parameters
    p_n: int = 500,
    num_quantisation_classes: int = 3,
    num_sampling_rounds: int = 3,
    subsample_ratio: float = 0.1,
    random_state: Optional[Union[int, np.random.Generator]] = None,
    convergence_threshold: float = 0.05,
    max_iterations: int = 10,
    downsample_factor: int = 10,
    # Planet parameters
    pif_method: str = "pca",
    pif_threshold: Optional[float] = None,
    transform_method: str = "robust",
    max_tile_memory_mb: float = 512,
    # Common
    save_histograms: bool = True,
    quiet: bool = False,
) -> Dict[str, Any]:
    """Iteratively normalise one or more target rasters to a reference
    raster using LIRRN with Bhattacharyya convergence.

    The algorithm proceeds in three phases:

    1. **Anchor pass** — normalise every target to the reference.
    2. **Reverse pass** — normalise the reference toward each (now
       normalised) target to reduce anchor bias.
    3. **Iterative cross-normalisation** — round-robin normalise each
       raster toward the mean of all others until the per-band
       Bhattacharyya distance falls below *convergence_threshold* or
       *max_iterations* is reached.

    LIRRN regression parameters are estimated on images downsampled by
    *downsample_factor* (memory-safe for large files), then the resulting
    linear transforms are applied to the full-resolution rasters via
    windowed I/O.

    Args:
        reference_path: Path to the reference raster (radiometric anchor).
        target_paths: One or more rasters to normalise toward the reference.
            Can be:
            - A single file path string.
            - A list of file path strings (labels derived from filenames).
            - A ``{label: path}`` dict for explicit labelling.
        output_dir: Base directory for normalised outputs.  Per-label
            sub-folders are created automatically.
        p_n: LIRRN sample count per quantisation level (default 500).
        num_quantisation_classes: Multi-Otsu brightness strata (default 3).
        num_sampling_rounds: LIRRN sampling rounds (default 3).
        subsample_ratio: Fraction of candidates for regression (default 0.1).
        random_state: Integer seed or ``np.random.Generator`` for
            reproducibility.
        convergence_threshold: Maximum Bhattacharyya distance at which to
            stop iterating (default 0.05).
        max_iterations: Hard cap on cross-normalisation rounds (default 10).
        downsample_factor: Spatial reduction factor for parameter estimation
            (default 10 → every 10th pixel used).
        save_histograms: Save before/after histogram PNGs to
            ``<output_dir>/histograms/`` (default True).
        quiet: Suppress progress output (default False).

    Returns:
        Dictionary containing:
            - ``convergence_history``: list of per-iteration distance records.
            - ``output_paths``: ``{label: normalised_output_path}``.
            - ``tile_name``: stem used for file naming.
            - ``output_dir``: the base output directory.

    Raises:
        FileNotFoundError: If *reference_path* or any target path does not
            exist.
        ValueError: If *target_paths* is empty or band counts are
            inconsistent.

    Examples:
        >>> from geoai.landcover_utils import normalize_multitemporal
        >>> results = normalize_multitemporal(
        ...     reference_path="data/2021_ortho.tif",
        ...     target_paths=[
        ...         "data/2024_ortho.tif",
        ...         "data/2015_ortho.tif",
        ...     ],
        ...     output_dir="normalized_output",
        ... )
        >>> # With explicit labels
        >>> results = normalize_multitemporal(
        ...     reference_path="data/2021_ortho.tif",
        ...     target_paths={"2024": "data/2024_ortho.tif"},
        ...     output_dir="normalized_output",
        ... )
    """
    # --- Normalise target_paths to a dict ---
    _RASTER_EXTS = {".tif", ".tiff", ".jp2", ".ers", ".img", ".vrt"}
    if isinstance(target_paths, str):
        _tp = Path(target_paths)
        if _tp.is_dir():
            target_paths = {
                f.stem: str(f)
                for f in sorted(_tp.iterdir())
                if f.suffix.lower() in _RASTER_EXTS
            }
            if not target_paths:
                raise ValueError(
                    f"No raster files found in folder: {_tp}. "
                    f"Expected extensions: {sorted(_RASTER_EXTS)}"
                )
        else:
            target_paths = {_tp.stem: str(_tp)}
    elif isinstance(target_paths, list):
        target_paths = {Path(p).stem: str(p) for p in target_paths}
    # else: already a dict — use as-is

    if not target_paths:
        raise ValueError("target_paths must contain at least one raster path.")

    # --- Validate paths exist ---
    missing = []
    if not Path(reference_path).exists():
        missing.append(reference_path)
    for lbl, p in target_paths.items():
        if not Path(p).exists():
            missing.append(p)
    if missing:
        raise FileNotFoundError(
            "The following raster paths do not exist:\n"
            + "\n".join(f"  {p}" for p in missing)
        )

    # --- Validate other params ---
    if method not in ("lirrn", "planet"):
        raise ValueError(f"method must be 'lirrn' or 'planet', got {method!r}")
    if p_n < 1:
        raise ValueError(f"p_n must be >= 1, got {p_n}")
    if num_sampling_rounds < 1:
        raise ValueError(f"num_sampling_rounds must be >= 1, got {num_sampling_rounds}")
    if subsample_ratio <= 0 or subsample_ratio > 1:
        raise ValueError(f"subsample_ratio must be in (0, 1], got {subsample_ratio}")
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")
    if downsample_factor < 1:
        raise ValueError(f"downsample_factor must be >= 1, got {downsample_factor}")

    # --- Derive reference label ---
    import re
    ref_label = Path(reference_path).stem
    if ref_label in target_paths:
        ref_label = ref_label + "_reference"
    year_match = re.search(r'(\d{4})', ref_label)
    if year_match:
        ref_label = year_match.group(1)

    output_dir = str(Path(output_dir))

    t0 = time.time()

    if not quiet:
        print(f"Method: {method}")
        print(f"Reference: {Path(reference_path).name}  (label: '{ref_label}')")
        for lbl, p in target_paths.items():
            print(f"  Target '{lbl}': {Path(p).name}")

    # -----------------------------------------------------------------------
    # Planet method — single-pass PIF normalization (one pass per target)
    # -----------------------------------------------------------------------
    if method == "planet":
        _pif_threshold = pif_threshold
        if _pif_threshold is None:
            _pif_threshold = 100.0 if pif_method == "robust" else 30.0

        planet_output_paths: Dict[str, str] = {}
        planet_metrics: Dict[str, Any] = {}
        ds_originals_p: Dict[str, np.ndarray] = {}
        ds_normalized_p: Dict[str, np.ndarray] = {}

        if save_histograms:
            ref_ds, _, _ = _load_raster_downsampled(reference_path, factor=10)
            ds_originals_p[ref_label] = ref_ds
            ds_normalized_p[ref_label] = ref_ds

        for lbl, sub_path in target_paths.items():
            lbl_out_dir = os.path.join(output_dir, lbl)
            os.makedirs(lbl_out_dir, exist_ok=True)
            out_name = Path(sub_path).stem + "_norm.tif"
            out_path = os.path.join(lbl_out_dir, out_name)

            if not quiet:
                print(f"  Normalizing '{lbl}': {Path(sub_path).name} ...")

            gains, offsets, pif_counts, ref_clip_max = _estimate_planet_pif_tiled(
                sub_path=sub_path,
                ref_path=reference_path,
                pif_method=pif_method,
                pif_threshold=_pif_threshold,
                transform_method=transform_method,
                downsample_factor=1,
                max_tile_memory_mb=max_tile_memory_mb,
            )

            if not quiet:
                for b, (g, o, c) in enumerate(zip(gains, offsets, pif_counts)):
                    print(
                        f"    Band {b + 1}: gain={g:.4f}, offset={o:.4f}, "
                        f"PIFs={int(c)}"
                    )

            _apply_linear_transform(
                sub_path,
                out_path,
                slopes=gains,
                intercepts=offsets,
                clip_min=np.zeros_like(gains),
                clip_max=ref_clip_max,
            )
            planet_output_paths[lbl] = out_path
            planet_metrics[lbl] = {
                "gains": gains.tolist(),
                "offsets": offsets.tolist(),
                "pif_counts": pif_counts.tolist(),
            }

            if save_histograms:
                orig_ds, _, _ = _load_raster_downsampled(sub_path, factor=10)
                norm_ds, _, _ = _load_raster_downsampled(out_path, factor=10)
                ds_originals_p[lbl] = orig_ds
                ds_normalized_p[lbl] = norm_ds

        result: Dict[str, Any] = {
            "tile_name": ref_label,
            "convergence_history": [],
            "output_paths": planet_output_paths,
            "metrics": planet_metrics,
            "ds_originals": ds_originals_p,
            "ds_normalized": ds_normalized_p,
        }

    # -----------------------------------------------------------------------
    # LIRRN method — iterative normalization with Bhattacharyya convergence
    # -----------------------------------------------------------------------
    else:
        # --- Build RNG ---
        if isinstance(random_state, np.random.Generator):
            rng = random_state
        else:
            rng = np.random.default_rng(random_state)

        tile_paths: Dict[str, str] = {ref_label: reference_path}
        tile_paths.update(target_paths)

        result = _normalize_tile_group(
            tile_paths=tile_paths,
            reference_year=ref_label,
            output_dir=output_dir,
            p_n=p_n,
            num_quantisation_classes=num_quantisation_classes,
            num_sampling_rounds=num_sampling_rounds,
            subsample_ratio=subsample_ratio,
            rng=rng,
            convergence_threshold=convergence_threshold,
            max_iterations=max_iterations,
            downsample_factor=downsample_factor,
            quiet=quiet,
        )

    # --- Save histogram comparison ---
    if save_histograms and result.get("ds_originals"):
        hist_path = os.path.join(
            output_dir, "histograms",
            f"{result['tile_name']}_comparison.png",
        )
        _plot_normalization_comparison(
            result["ds_originals"],
            result["ds_normalized"],
            result["tile_name"],
            hist_path,
        )
        if not quiet:
            print(f"Histogram saved: {hist_path}")

    elapsed = round(time.time() - t0, 1)

    # --- Summary ---
    if not quiet:
        print(f"\n{'='*60}")
        print("MULTI-TEMPORAL NORMALIZATION COMPLETE")
        print(f"{'='*60}")
        print(f"Completed in {elapsed}s")
        if result["convergence_history"]:
            last = result["convergence_history"][-1]
            print(
                f"Final max Bhattacharyya distance: "
                f"{last['max_bhattacharyya']:.4f} "
                f"({last['phase']}, iter {last['iteration']})"
            )
        for lbl, p in result["output_paths"].items():
            print(f"  {lbl}: {p}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}\n")

    result["output_dir"] = output_dir
    result["elapsed"] = elapsed
    return result
