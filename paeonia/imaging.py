"""The 12 image descriptors used in the visible15 panel."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from scipy import ndimage
from PIL import Image


def load_rgb(path: Path, max_side: int) -> tuple[np.ndarray, tuple[int, int], float]:
    with Image.open(path) as image:
        image = image.convert("RGB")
        original_size = image.size
        scale = min(1.0, float(max_side) / max(original_size))
        if scale < 1.0:
            new_size = (
                max(1, int(round(original_size[0] * scale))),
                max(1, int(round(original_size[1] * scale))),
            )
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        arr = np.asarray(image, dtype=np.uint8)
    return arr, original_size, scale


def otsu_threshold(values: np.ndarray) -> float:
    hist, bin_edges = np.histogram(values.reshape(-1), bins=256, range=(0, 255))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 20.0

    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    sum_bg = np.cumsum(hist * centers)
    sum_total = sum_bg[-1]

    valid = (weight_bg > 0) & (weight_fg > 0)
    mean_bg = np.zeros_like(centers)
    mean_fg = np.zeros_like(centers)
    mean_bg[valid] = sum_bg[valid] / weight_bg[valid]
    mean_fg[valid] = (sum_total - sum_bg[valid]) / weight_fg[valid]
    between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    return float(centers[int(np.argmax(between))])


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    labels, n_labels = ndimage.label(mask)
    if n_labels == 0:
        return mask
    areas = np.bincount(labels.reshape(-1))
    keep = areas >= min_area
    keep[0] = False
    return keep[labels]


def build_foreground_mask(rgb: np.ndarray, min_component_area: int) -> tuple[np.ndarray, float]:
    rgb_float = rgb.astype(np.float32)
    luminance = (
        0.2126 * rgb_float[:, :, 0]
        + 0.7152 * rgb_float[:, :, 1]
        + 0.0722 * rgb_float[:, :, 2]
    )
    threshold = min(max(otsu_threshold(luminance), 12.0), 75.0)
    mask = luminance > threshold

    structure = np.ones((3, 3), dtype=bool)
    mask = ndimage.binary_opening(mask, structure=structure)
    mask = ndimage.binary_closing(mask, structure=np.ones((5, 5), dtype=bool))
    mask = ndimage.binary_fill_holes(mask)
    image_scaled_min_area = max(min_component_area, int(mask.size * 1e-5))
    mask = remove_small_components(mask, image_scaled_min_area)
    mask = ndimage.binary_fill_holes(mask)
    return mask.astype(bool), threshold


def erode_for_interior(mask: np.ndarray) -> np.ndarray:
    if int(mask.sum()) == 0:
        return mask
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), iterations=2)
    if int(eroded.sum()) < int(mask.sum()) * 0.45:
        return mask
    return eroded.astype(bool)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    rgb_norm = rgb.astype(np.float32) / 255.0
    linear = np.where(
        rgb_norm <= 0.04045,
        rgb_norm / 12.92,
        ((rgb_norm + 0.055) / 1.055) ** 2.4,
    )
    matrix = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float32,
    )
    xyz = linear @ matrix.T
    xyz = xyz / np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)
    delta = 6 / 29
    f_xyz = np.where(xyz > delta**3, np.cbrt(xyz), xyz / (3 * delta**2) + 4 / 29)
    lab = np.empty_like(f_xyz, dtype=np.float32)
    lab[:, :, 0] = 116 * f_xyz[:, :, 1] - 16
    lab[:, :, 1] = 500 * (f_xyz[:, :, 0] - f_xyz[:, :, 1])
    lab[:, :, 2] = 200 * (f_xyz[:, :, 1] - f_xyz[:, :, 2])
    return lab


def weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return float("nan")
    mean = np.average(values[valid], weights=weights[valid])
    return float(np.sqrt(np.average((values[valid] - mean) ** 2, weights=weights[valid])))


def component_arrays(mask: np.ndarray) -> tuple[np.ndarray, int, np.ndarray]:
    labels, n_labels = ndimage.label(mask)
    if n_labels == 0:
        return labels, 0, np.array([], dtype=float)
    areas = np.bincount(labels.reshape(-1))[1:].astype(float)
    return labels, n_labels, areas


def edge_ring_mask(mask: np.ndarray, iterations: int = 4) -> np.ndarray:
    if int(mask.sum()) == 0:
        return mask
    eroded = ndimage.binary_erosion(
        mask, structure=np.ones((3, 3), dtype=bool), iterations=iterations
    )
    edge = mask & ~eroded
    if int(edge.sum()) < 20:
        return mask
    return edge.astype(bool)


def masked_local_std(image: np.ndarray, mask: np.ndarray, window: int) -> np.ndarray:
    mask_float = mask.astype(np.float32)
    size = (window, window)
    count = ndimage.uniform_filter(mask_float, size=size, mode="constant") * (window**2)
    summed = ndimage.uniform_filter(
        image.astype(np.float32) * mask_float, size=size, mode="constant"
    )
    summed *= window**2
    sum_sq = ndimage.uniform_filter(
        (image.astype(np.float32) ** 2) * mask_float, size=size, mode="constant"
    )
    sum_sq *= window**2
    valid = (mask & (count >= window**2 * 0.5)) & (count > 0)
    mean = np.zeros_like(image, dtype=np.float32)
    variance = np.zeros_like(image, dtype=np.float32)
    mean[valid] = summed[valid] / count[valid]
    variance[valid] = sum_sq[valid] / count[valid] - mean[valid] ** 2
    variance = np.maximum(variance, 0.0)
    return np.sqrt(variance)[valid]


def glcm_entropy(l_values: np.ndarray, mask: np.ndarray, levels: int = 32) -> float:
    quantized = np.clip((l_values / 100.0 * levels).astype(np.int16), 0, levels - 1)
    offsets = [(0, 1), (1, 0), (1, 1), (1, -1)]
    glcm = np.zeros((levels, levels), dtype=np.float64)
    for dy, dx in offsets:
        if dy >= 0:
            y1 = slice(0, -dy or None)
            y2 = slice(dy, None)
        else:
            y1 = slice(-dy, None)
            y2 = slice(0, dy)
        if dx >= 0:
            x1 = slice(0, -dx or None)
            x2 = slice(dx, None)
        else:
            x1 = slice(-dx, None)
            x2 = slice(0, dx)
        valid = mask[y1, x1] & mask[y2, x2]
        if not np.any(valid):
            continue
        left = quantized[y1, x1][valid].reshape(-1)
        right = quantized[y2, x2][valid].reshape(-1)
        counts = np.bincount(left * levels + right, minlength=levels * levels)
        matrix = counts.reshape(levels, levels).astype(np.float64)
        glcm += matrix + matrix.T

    total = glcm.sum()
    if total <= 0:
        return float("nan")
    p = glcm / total
    nonzero = p[p > 0]
    return float(-np.sum(nonzero * np.log2(nonzero)))


def process_image(path: Path, args: argparse.Namespace) -> dict[str, float]:
    """Extract selected descriptors without using grade or chemical measurements."""
    rgb, _, _ = load_rgb(path, args.max_image_side)
    mask, _ = build_foreground_mask(rgb, args.min_component_area)
    if not mask.any():
        raise ValueError("No foreground pieces detected")
    interior = erode_for_interior(mask)
    edge = edge_ring_mask(mask)
    lab = rgb_to_lab(rgb)
    pixels = lab[mask]
    labels, _, areas = component_arrays(mask)
    l_values = pixels[:, 0]
    cutoff = np.median(l_values) - 1.5 * (np.nanpercentile(l_values, 75) - np.nanpercentile(l_values, 25))
    dark = remove_small_components(mask & (lab[:, :, 0] < cutoff), max(4, int(mask.sum() * 2e-5)))
    center = np.median(pixels, axis=0)
    delta_e = np.sqrt(np.sum((lab - center) ** 2, axis=2))[mask]
    object_l = []
    for label_id, bbox in enumerate(ndimage.find_objects(labels), start=1):
        if bbox is not None:
            component = labels[bbox] == label_id
            object_l.append(float(np.mean(lab[bbox][component], axis=0)[0]))
    lightness = lab[:, :, 0].astype(np.float32)
    fine = np.abs(lightness - ndimage.gaussian_filter(lightness, sigma=1.0, mode="nearest"))[interior]
    coarse = np.abs(ndimage.gaussian_filter(lightness, sigma=3.0, mode="nearest") - ndimage.gaussian_filter(lightness, sigma=7.0, mode="nearest"))[interior]
    fine_mean = float(np.mean(fine))
    return {
        "visible_mask_component_area_cv": float(np.std(areas, ddof=0) / np.mean(areas)),
        "visible_lab_L_mean": float(np.mean(pixels[:, 0])),
        "visible_lab_a_mean": float(np.mean(pixels[:, 1])),
        "visible_lab_b_mean": float(np.mean(pixels[:, 2])),
        "visible_color_delta_e_fraction_gt_12": float(np.mean(delta_e > 12.0)),
        "visible_object_L_between_std": weighted_std(np.asarray(object_l), areas),
        "visible_relative_dark_spot_area_fraction": float(dark.sum() / max(float(mask.sum()), 1.0)),
        "visible_edge_minus_inner_L_mean": float(float(np.mean(lab[edge][:, 0])) - float(np.mean(lab[interior][:, 0]))),
        "visible_texture_local_L_std_w5_mean": float(np.mean(masked_local_std(lightness, interior, 5))),
        "visible_texture_local_L_std_w33_mean": float(np.mean(masked_local_std(lightness, interior, 33))),
        "visible_texture_coarse_to_fine_ratio": float(float(np.mean(coarse)) / fine_mean) if fine_mean else float("nan"),
        "visible_glcm_entropy": glcm_entropy(lightness, interior),
    }
