"""
Landcover Classification Utilities - Enhanced Tile Export and Normalization Module

This module extends the base geoai functionality with specialized utilities
for landcover classification. It provides enhanced tile generation
with background filtering capabilities to improve training efficiency,
and radiometric normalization methods for image comparability.

Key Features:
- Enhanced tile filtering with configurable feature ratio thresholds
- Separate statistics tracking for different skip reasons
- Radiometric normalization for multi-temporal/multi-sensor image comparability
- Maintains full compatibility with base geoai workflow
- Optimized for landcover classification tasks

Date: November 2025
"""

import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from rasterio.windows import Window
from sklearn.linear_model import LinearRegression
from skimage.filters import threshold_multiotsu
from tqdm import tqdm


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


# ----------------------------------------------------------------------------
# Radiometric Normalization Functions
# ----------------------------------------------------------------------------

def compute_distances(p_n, a1, b1, id_indices):
    """Select P_N samples closest to the maximum value, then subsample with *id_indices*.

    Parameters
    ----------
    p_n : int
        Number of candidate samples to draw from each array.
    a1 : np.ndarray
        Non-zero reference pixel values (1-D).
    b1 : np.ndarray
        Non-zero subject pixel values (1-D).
    id_indices : np.ndarray
        Random indices used to subsample from the P_N candidates.

    Returns
    -------
    sub, ref : np.ndarray
        Subsampled subject and reference values.
    """
    # Reference: P_N values closest to max
    max_ref = np.max(a1)
    idx_ref = np.argsort(np.abs(a1 - max_ref))
    ref_candidates = a1[idx_ref[:p_n]]

    # Subject: P_N values closest to max
    max_sub = np.max(b1)
    idx_sub = np.argsort(np.abs(b1 - max_sub))
    sub_candidates = b1[idx_sub[:p_n]]

    return sub_candidates[id_indices], ref_candidates[id_indices]


def compute_sample(p_n, a1, b1, id_indices, num_sampling_rounds=3):
    """Three rounds of distance-based sampling, concatenated (non-zero only).

    In the original MATLAB all three calls are identical (max-based).
    The combined pool gives a larger, slightly varied sample set because
    *id_indices* was drawn randomly once at the start.
    """
    pairs = [compute_distances(p_n, a1, b1, id_indices) for _ in range(num_sampling_rounds)]
    sub_combined = np.concatenate([s[s != 0] for s, _ in pairs])
    ref_combined = np.concatenate([r[r != 0] for _, r in pairs])
    return sub_combined, ref_combined


def sample_selection(p_n, a, b, id_indices, num_sampling_rounds=3):
    """Select representative sample pairs from a single quantisation level.

    Parameters
    ----------
    p_n : int
        Number of sample points.
    a : np.ndarray
        Flattened reference pixels (masked by quantisation level; 0 = outside).
    b : np.ndarray
        Flattened subject pixels (masked by quantisation level; 0 = outside).
    id_indices : np.ndarray
        Random sub-sampling indices.
    num_sampling_rounds : int
        Number of sampling rounds.

    Returns
    -------
    sub, ref : np.ndarray   (1-D each)
    """
    a1 = a[a != 0]
    b1 = b[b != 0]

    # Guard: not enough pixels in this quantisation level
    if len(a1) < p_n or len(b1) < p_n:
        min_len = min(len(a1), len(b1))
        if min_len == 0:
            return np.array([0.0]), np.array([0.0])
        return b1[:min_len], a1[:min_len]

    sub_1, ref_1 = compute_sample(p_n, a1, b1, id_indices, num_sampling_rounds)

    # Non-zeros first, then zeros (matches MATLAB behaviour)
    sub = np.concatenate([sub_1[sub_1 != 0], sub_1[sub_1 == 0]])
    ref = np.concatenate([ref_1[ref_1 != 0], ref_1[ref_1 == 0]])
    return sub, ref


def linear_reg(sub_samples, ref_samples, image_band):
    """OLS linear regression: ``ref = intercept + slope * sub``, applied to *image_band*.

    Parameters
    ----------
    sub_samples : np.ndarray  (1-D)
    ref_samples : np.ndarray  (1-D)
    image_band  : np.ndarray  (H, W)

    Returns
    -------
    norm_band : np.ndarray (H, W)
    r_adj     : float   – adjusted R²
    rmse      : float
    """
    X = sub_samples.reshape(-1, 1)
    y = ref_samples

    model = LinearRegression().fit(X, y)
    intercept, slope = model.intercept_, model.coef_[0]

    # Apply transformation
    norm_band = intercept + slope * image_band

    # Goodness-of-fit metrics
    y_pred = model.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    n = len(y)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
    r_adj = 1 - (1 - r2) * (n - 1) / (n - 2) if n > 2 else 0.0
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))

    return norm_band, r_adj, rmse


def lirrn(p_n, sub_img, ref_img, num_quantisation_classes=3, num_sampling_rounds=3, subsample_ratio=0.1, random_seed=None):
    """Location-Independent Relative Radiometric Normalization.

    Parameters
    ----------
    p_n : int
        Number of sample points per quantisation level.
    sub_img : np.ndarray  (H, W, B) float64
        Subject image.
    ref_img : np.ndarray  (H, W, B) float64
        Reference image.
    num_quantisation_classes : int
        Number of brightness strata (default: 3).
    num_sampling_rounds : int
        Number of sampling rounds (default: 3).
    subsample_ratio : float
        Fraction of candidates retained (default: 0.1).
    random_seed : int or None
        Random seed for reproducibility (default: None).

    Returns
    -------
    norm_img : np.ndarray (H, W, B)
    rmse     : np.ndarray (B,)
    r_adj    : np.ndarray (B,)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    num_bands = sub_img.shape[2]

    # Random sub-sampling indices (SUBSAMPLE_RATIO % of p_n), drawn once as in the MATLAB code
    id_indices = np.random.randint(0, p_n, size=max(1, round(subsample_ratio * p_n)))

    norm_img = np.zeros_like(sub_img, dtype=np.float64)
    rmse = np.zeros(num_bands)
    r_adj = np.zeros(num_bands)

    # --- Quantise each band into NUM_QUANTISATION_CLASSES levels (multi-Otsu) ---
    sub_labels = np.zeros_like(sub_img, dtype=np.int32)
    ref_labels = np.zeros_like(ref_img, dtype=np.int32)

    for j in range(num_bands):
        for img, labels in [(sub_img, sub_labels), (ref_img, ref_labels)]:
            nonzero = img[:, :, j][img[:, :, j] != 0]
            if len(nonzero) > 0:
                try:
                    thresh = threshold_multiotsu(nonzero, classes=num_quantisation_classes)
                    # np.digitize + 1 gives labels 1, 2, 3... like MATLAB imquantize
                    labels[:, :, j] = np.digitize(img[:, :, j], bins=thresh) + 1
                except ValueError:
                    labels[:, :, j] = 1

    # --- For each band: sample from NUM_QUANTISATION_CLASSES levels then regress ---
    for j in range(num_bands):
        sub_list, ref_list = [], []

        for level in range(1, num_quantisation_classes + 1):
            a = np.where(ref_labels[:, :, j] == level, ref_img[:, :, j], 0).ravel()
            b = np.where(sub_labels[:, :, j] == level, sub_img[:, :, j], 0).ravel()
            sub_s, ref_s = sample_selection(p_n, a, b, id_indices, num_sampling_rounds)
            sub_list.append(sub_s)
            ref_list.append(ref_s)

        all_sub = np.concatenate(sub_list)
        all_ref = np.concatenate(ref_list)
        norm_img[:, :, j], r_adj[j], rmse[j] = linear_reg(all_sub, all_ref, sub_img[:, :, j])

    return norm_img, rmse, r_adj


def normalize_radiometric(
    subject_image,
    reference_image,
    method='lirrn',
    p_n=1000,
    num_quantisation_classes=3,
    num_sampling_rounds=3,
    subsample_ratio=0.1,
    random_seed=None
):
    """Normalize subject image to match reference image radiometrically for improved comparability.

    Radiometric normalization adjusts the brightness and contrast of images to make them
    comparable across different acquisition times, sensors, or atmospheric conditions.
    This is crucial for landcover classification tasks using multi-temporal or multi-sensor data.

    Currently supports LIRRN (Location-Independent Relative Radiometric Normalization),
    which uses stratified sampling and linear regression per band to achieve robust normalization.

    Parameters
    ----------
    subject_image : np.ndarray
        Subject image array (H, W, B) to be normalized.
    reference_image : np.ndarray
        Reference image array (H, W, B) with desired radiometric properties.
    method : str, default 'lirrn'
        Normalization method. Currently only 'lirrn' is supported.
    p_n : int, default 1000
        Number of sample points per quantisation level for LIRRN (NUM_SAMPLE_POINTS).
    num_quantisation_classes : int, default 3
        Number of brightness strata for stratified sampling in LIRRN (NUM_QUANTISATION_CLASSES).
    num_sampling_rounds : int, default 3
        Number of sampling rounds for robustness in LIRRN (NUM_SAMPLING_ROUNDS).
    subsample_ratio : float, default 0.1
        Fraction of candidates retained for regression in LIRRN (SUBSAMPLE_RATIO).
    random_seed : int or None, default None
        Random seed for reproducible results (RANDOM_SEED).

    Returns
    -------
    normalized_image : np.ndarray
        Normalized image with radiometric properties matching the reference.
    metrics : dict
        Dictionary containing:
        - 'rmse': Root Mean Square Error for each band
        - 'r_adj': Adjusted R² for each band

    Raises
    ------
    NotImplementedError
        If an unsupported normalization method is specified.

    Examples
    --------
    >>> import numpy as np
    >>> # Assuming subject_img and ref_img are loaded numpy arrays
    >>> norm_img, metrics = normalize_radiometric(subject_img, ref_img)
    >>> print(f"RMSE per band: {metrics['rmse']}")
    """
    if method == 'lirrn':
        norm_img, rmse, r_adj = lirrn(
            p_n, subject_image, reference_image,
            num_quantisation_classes=num_quantisation_classes,
            num_sampling_rounds=num_sampling_rounds,
            subsample_ratio=subsample_ratio,
            random_seed=random_seed
        )
        metrics = {'rmse': rmse, 'r_adj': r_adj}
        return norm_img, metrics
    else:
        raise NotImplementedError(f"Normalization method '{method}' not implemented.")
