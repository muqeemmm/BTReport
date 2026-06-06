"""
Transition Zone Thickness — Two Complementary Methods
======================================================

Method A (PRIMARY, recommended):
    Distance-transform local thickness on the geometric transition zone.
    - Purely geometric, segmentation-only.
    - Reproducible across scanners and intensity protocols.
    - Based on Hildebrand-Ruegsegger local thickness (well-established in
      bone microarchitecture imaging).

Method B (SECONDARY):
    Ray-casting along outward normals with intensity-band stopping criterion.
    - Intensity-aware "biological" transition depth.
    - Use with FLAIR (or T2) for meaningful infiltration depth — NOT T1CE,
      since edema has no characteristic T1CE signature.
    - Kept here in fixed form, but treat as a research / exploratory metric.

BraTS-style label convention assumed:
    1 = NCR/NET (necrotic / non-enhancing tumor core)
    2 = ED      (peritumoral edema)
    3 = (legacy / unused in some BraTS versions)
    4 = ET      (enhancing tumor)
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    distance_transform_edt,
    generate_binary_structure,
    label,
    map_coordinates,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Shared helpers
# =============================================================================

def _build_masks(seg_array: np.ndarray) -> dict:
    """
    Build region masks from a BraTS-style integer segmentation.

    Returns
    -------
    dict with boolean masks:
        et   : enhancing tumor          (label 4)
        ncr  : necrotic / non-enhancing (label 1)
        ed   : peritumoral edema        (label 2)
        tc   : tumor core               (NCR + ET)  -> labels {1, 3, 4}
        wt   : whole tumor              (NCR + ED + ET)
        tz   : transition zone          (WT - TC)  ~= edema region
    """
    et  = np.isin(seg_array, [3, 4])
    ncr = np.isin(seg_array, [1])
    ed  = np.isin(seg_array, [2])
    tc  = np.isin(seg_array, [1, 3, 4])
    wt  = np.isin(seg_array, [1, 2, 3, 4])
    tz  = wt & ~tc
    return {"et": et, "ncr": ncr, "ed": ed, "tc": tc, "wt": wt, "tz": tz}


def _clean_mask(mask: np.ndarray, closing_radius: int = 1,
                min_voxels: int = 27) -> np.ndarray:
    """Light morphological cleanup: closing + small-component removal."""
    if not mask.any():
        return mask
    struct = generate_binary_structure(3, 1)
    cleaned = binary_closing(mask, structure=struct, iterations=closing_radius)

    # remove small connected components
    lab, n = label(cleaned, structure=generate_binary_structure(3, 2))
    if n == 0:
        return cleaned
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0  # background
    keep = sizes >= min_voxels
    return keep[lab]


# =============================================================================
# Method A — Distance-transform local thickness  (PRIMARY, recommended)
# =============================================================================

def compute_local_thickness_map(mask: np.ndarray,
                                 voxel_spacing: Tuple[float, float, float]
                                 ) -> np.ndarray:
    """
    Local thickness at each voxel = 2 x (distance to nearest boundary).

    This is the classical Hildebrand-Ruegsegger approximation: for a voxel
    inside the mask, the inscribed-sphere-diameter at that voxel is twice
    its distance to the nearest non-mask voxel. For convex-like shells
    (which a peritumoral transition zone usually is), this is an accurate
    and well-validated thickness measure.

    Returns thickness in mm everywhere inside the mask, 0 outside.
    """
    if not mask.any():
        return np.zeros_like(mask, dtype=np.float32)
    dist = distance_transform_edt(mask, sampling=voxel_spacing)
    return (2.0 * dist).astype(np.float32)


def transition_zone_thickness_distance_transform(
    seg_array: np.ndarray,
    voxel_spacing: Tuple[float, float, float],
    *,
    clean_segmentation: bool = True,
) -> dict:
    """
    Geometric transition zone thickness via distance transform.

    Transition zone = WT - TC  (i.e. the peritumoral / non-core region
    inside the whole-tumor envelope; on BraTS this corresponds to edema).

    Parameters
    ----------
    seg_array : 3D int array, BraTS-style labels.
    voxel_spacing : (sx, sy, sz) in mm.
    clean_segmentation : whether to apply light morphological cleanup
        before measurement.

    Returns
    -------
    dict with thickness statistics (mm), volume, and the per-voxel
    thickness map (for visualization).
    """
    masks = _build_masks(seg_array)
    tz = masks["tz"] # Basically edema region

    if clean_segmentation:
        tz = _clean_mask(tz, closing_radius=1, min_voxels=27)

    if not tz.any():
        return {
            "method": "distance_transform_local_thickness",
            "transition_zone_present": False,
            "mean_mm": np.nan,
            "median_mm": np.nan,
            "std_mm": np.nan,
            "min_mm": np.nan,
            "max_mm": np.nan,
            "p95_mm": np.nan,
            "volume_mm3": 0.0,
            "voxel_spacing": voxel_spacing,
            "thickness_map": None,
        }

    thickness_map = compute_local_thickness_map(tz, voxel_spacing)
    values = thickness_map[tz]

    voxel_volume = float(np.prod(voxel_spacing))
    return {
        "method": "distance_transform_local_thickness",
        "transition_zone_present": True,
        "mean_mm":   float(np.mean(values)),
        "median_mm": float(np.median(values)),
        "std_mm":    float(np.std(values)),
        "min_mm":    float(np.min(values)),
        "max_mm":    float(np.max(values)),
        "p95_mm":    float(np.percentile(values, 95)),
        "volume_mm3": float(tz.sum() * voxel_volume),
        "voxel_spacing": tuple(float(s) for s in voxel_spacing),
        "thickness_map": thickness_map,
    }


# =============================================================================
# Method B — Ray-casting along outward normals  (SECONDARY, intensity-aware)
# =============================================================================

def compute_outward_normals(tc_mask: np.ndarray,
                             voxel_spacing: Tuple[float, float, float]
                             ) -> np.ndarray:
    """
    Outward unit normals (in PHYSICAL/mm space) at every voxel of the mask,
    computed as the negative gradient of the interior distance transform.

    Returns
    -------
    ndarray of shape (X, Y, Z, 3): unit-norm outward direction in mm-space
    at each voxel; zero where the mask interior is degenerate.
    """
    tc_mask = tc_mask.astype(bool)
    voxel_spacing = np.asarray(voxel_spacing, dtype=float)

    # Distance transform with physical spacing -> distances in mm
    dist_in = distance_transform_edt(tc_mask, sampling=voxel_spacing)

    # np.gradient with spacing returns d/dmm along each axis
    grads = np.gradient(dist_in, *voxel_spacing, edge_order=1)
    grad = np.stack(grads, axis=-1)  # (X, Y, Z, 3) in 1/mm sense

    outward = -grad  # points outward (from interior to boundary)
    mag = np.linalg.norm(outward, axis=-1, keepdims=True)
    return np.divide(
        outward,
        np.maximum(mag, 1e-12),
        out=np.zeros_like(outward),
        where=mag > 1e-12,
    )


def _nn_in_mask(mask: np.ndarray, p_vox: np.ndarray) -> bool:
    """Nearest-neighbor membership test for a sub-voxel coordinate."""
    idx = np.round(p_vox).astype(int)
    if np.any(idx < 0) or np.any(idx >= np.asarray(mask.shape)):
        return False
    return bool(mask[tuple(idx)])


def transition_zone_thickness_raycast(
    intensity_array: np.ndarray,
    seg_array: np.ndarray,
    voxel_spacing: Tuple[float, float, float],
    *,
    step_mm: float | None = None,
    max_dist_mm: float = 80.0,
    mad_scale: float = 1.4826,
    clean_segmentation: bool = True,
) -> dict:
    """
    Intensity-aware transition zone thickness via outward-normal ray casting.

    For each voxel on the tumor-core boundary that is adjacent to edema, cast
    a ray outward along the local surface normal. Walk the ray in physical
    (mm) space, sampling intensity with trilinear interpolation. Stop when
    the ray reaches a point that is (a) inside the edema mask and (b) has an
    intensity within a robust band derived from the edema region itself.

    The intensity stopping band is defined as:
        ed_lower = median(ed_vals) - mad_scale * MAD(ed_vals)
        ed_upper = median(ed_vals) + mad_scale * MAD(ed_vals)
    and then clamped to [min(ed_vals), max(ed_vals)].

    NOTE on modality choice:
      The intensity band is meaningful only on a sequence where edema has
      a characteristic signal (FLAIR, T2). On T1CE, edema has no specific
      signature, so this metric is hard to interpret. Prefer FLAIR.

    Parameters
    ----------
    intensity_array : 3D float array, same grid as seg_array.
    seg_array       : 3D int array, BraTS-style labels.
    voxel_spacing   : (sx, sy, sz) in mm.
    step_mm         : ray step size in mm. Default = 0.25 * min(spacing).
    max_dist_mm     : max ray length in mm.
    mad_scale       : multiplier applied to the MAD when building the
        intensity band (default 1.4826, consistent with normal-distribution
        std-dev equivalence).
    clean_segmentation : light morphological cleanup before measurement.

    Returns
    -------
    dict with thickness statistics and ray-outcome diagnostics.
    """
    voxel_spacing = tuple(float(s) for s in voxel_spacing)
    spacing_arr = np.asarray(voxel_spacing, dtype=float)

    masks = _build_masks(seg_array)
    tc_mask = masks["tc"]
    ed_mask = masks["ed"]

    if clean_segmentation:
        tc_mask = _clean_mask(tc_mask, closing_radius=1, min_voxels=27)
        ed_mask = _clean_mask(ed_mask, closing_radius=1, min_voxels=27)

    empty_result = {
        "method": "raycast_intensity_band",
        "transition_zone_thickness_mm": np.nan,
        "mean_mm": np.nan,
        "median_mm": np.nan,
        "std_mm": np.nan,
        "p95_mm": np.nan,
        "n_boundary_voxels": 0,
        "n_successful_rays": 0,
        "ray_outcomes": {"hit": 0, "exited_no_hit": 0, "hit_max_dist": 0,
                         "degenerate_normal": 0},
        "edema_intensity_band": (np.nan, np.nan),
        "voxel_spacing": voxel_spacing,
    }

    if not tc_mask.any():
        empty_result["reason"] = "Empty tumor core."
        return empty_result
    if not ed_mask.any():
        empty_result["reason"] = "Empty edema region."
        return empty_result

    # tumor-core boundary adjacent to edema
    edema_dil = binary_dilation(ed_mask, iterations=1)
    boundary = tc_mask & edema_dil & ~ed_mask
    boundary_coords = np.argwhere(boundary)

    # Add a field inside the structred dict for the number of boundary coordinates
    empty_result["n_boundary_voxels"] = int(boundary_coords.shape[0])

    if boundary_coords.size == 0:
        empty_result["reason"] = "No tumor-core boundary adjacent to edema."
        return empty_result

    # robust edema intensity band
    ed_vals = intensity_array[ed_mask]

    # MAD-based intensity band, clamped to the observed range of edema values
    ed_med = float(np.median(ed_vals))
    mad    = float(np.median(np.abs(ed_vals - ed_med))) + 1e-6
    ed_lower = ed_med - mad_scale * mad
    ed_upper = ed_med + mad_scale * mad
    ed_lower = float(max(ed_lower, float(np.min(ed_vals))))
    ed_upper = float(min(ed_upper, float(np.max(ed_vals))))

    # outward normals (in mm-space)
    outward_normals = compute_outward_normals(tc_mask, voxel_spacing)

    # ray marching
    if step_mm is None:
        step_mm = 0.25 * float(np.min(spacing_arr))
    n_steps = int(np.ceil(max_dist_mm / step_mm))

    thicknesses = []
    outcomes = {"hit": 0, "exited_no_hit": 0, "hit_max_dist": 0,
                "degenerate_normal": 0}

    for xyz in boundary_coords:
        p_boundary_vox = xyz.astype(float)

        u_mm = outward_normals[tuple(xyz)].astype(float)
        if np.linalg.norm(u_mm) < 1e-8:
            outcomes["degenerate_normal"] += 1
            continue

        # convert mm-step to voxel-step: dp_vox = (u_mm * step_mm) / spacing
        d_vox_step = (u_mm * step_mm) / spacing_arr

        # first sample just outside the core boundary
        p_vox = p_boundary_vox + d_vox_step
        prev_int = float(map_coordinates(
            intensity_array,
            [[p_vox[0]], [p_vox[1]], [p_vox[2]]],
            order=1, mode="nearest",
        )[0])
        prev_d_mm = 0.0

        # accept zero thickness if immediately in edema band
        if _nn_in_mask(ed_mask, p_vox) and (ed_lower <= prev_int <= ed_upper):
            thicknesses.append(0.0)
            outcomes["hit"] += 1
            continue

        hit_recorded = False
        for k in range(1, n_steps + 1):
            p_vox_k = p_boundary_vox + d_vox_step * k
            d_mm = step_mm * k

            I = float(map_coordinates(
                intensity_array,
                [[p_vox_k[0]], [p_vox_k[1]], [p_vox_k[2]]],
                order=1, mode="nearest",
            )[0])

            in_tc = _nn_in_mask(tc_mask, p_vox_k)
            in_ed = _nn_in_mask(ed_mask, p_vox_k)

            # left core but never reached edema -> ray fails
            if (not in_tc) and (not in_ed):
                outcomes["exited_no_hit"] += 1
                break

            # valid stop: inside edema with intensity in band
            if in_ed and (ed_lower <= I <= ed_upper):
                if abs(I - prev_int) > 1e-6:
                    if prev_int > ed_upper and I <= ed_upper:
                        target = ed_upper
                    elif prev_int < ed_lower and I >= ed_lower:
                        target = ed_lower
                    else:
                        target = ed_med
                    alpha = (target - prev_int) / (I - prev_int + 1e-12)
                    alpha = float(np.clip(alpha, 0.0, 1.0))
                    d_mm = prev_d_mm + alpha * (d_mm - prev_d_mm)

                thicknesses.append(float(d_mm))
                outcomes["hit"] += 1
                hit_recorded = True
                break

            prev_int, prev_d_mm = I, d_mm

    total_rays_attempted = int(boundary_coords.shape[0]) - outcomes["degenerate_normal"]
    outcomes["hit_max_dist"] = max(
        0,
        total_rays_attempted - outcomes["hit"] - outcomes["exited_no_hit"],
    )

    if len(thicknesses) == 0:
        return {
            **empty_result,
            "n_boundary_voxels": int(boundary_coords.shape[0]),
            "ray_outcomes": outcomes,
            "edema_intensity_band": (ed_lower, ed_upper),
            "reason": "No rays produced a valid hit.",
        }

    arr = np.asarray(thicknesses, dtype=float)
    return {
        "method": "raycast_intensity_band",
        "transition_zone_thickness_mm": float(np.median(arr)),
        "mean_mm":   float(np.mean(arr)),
        "median_mm": float(np.median(arr)),
        "std_mm":    float(np.std(arr)),
        "p95_mm":    float(np.percentile(arr, 95)),
        "n_boundary_voxels": int(boundary_coords.shape[0]),
        "n_successful_rays": int(len(thicknesses)),
        "ray_outcomes": outcomes,
        "edema_intensity_band": (ed_lower, ed_upper),
        "voxel_spacing": voxel_spacing,
    }
