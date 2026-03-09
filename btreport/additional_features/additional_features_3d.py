import trimesh, ants
import numpy as np
from typing import Tuple
from skimage import measure
from vasari_features.vasari_auto_v2 import NiftiImage
from scipy.ndimage import (
    label,
    binary_dilation,
    binary_erosion,
    generate_binary_structure,
    distance_transform_edt,
)

def compute_sphericity(seg_path: str) -> float:
    """
    Compute sphericity of a binary mask.

    Sphericity = (pi^(1/3) * (6 * V)^(2/3)) / A
    where V is volume and A is surface area.
    """
    seg_img = NiftiImage(seg_path)
    seg_array = seg_img.array.astype(np.int16)
    spacing = tuple(float(s) for s in seg_img.spacing)

    wt_mask        = np.isin(seg_array, [1, 2, 3, 4])       # Whole Tumor
    tc_mask        = np.isin(seg_array, [1, 3, 4])          # Tumor Core
    if not wt_mask.any():
        return np.nan

    wt_sphericity = np.nan
    tc_sphericity = np.nan
    # Compute surface area using marching cubes
    if wt_mask.sum() > 0:
        verts, faces, _, _ = measure.marching_cubes(wt_mask.astype(np.uint8), 
                                                    level = 0.5, spacing = spacing,
                                                    allow_degenerate = False)

        mesh = trimesh.Trimesh(vertices = verts, faces = faces, process = True)

        surface_area = float(mesh.area)  # in mm^2
        if mesh.is_watertight and np.isfinite(mesh.volume):
            volume = float(abs(mesh.volume))
        else:
            volume = float(wt_mask.sum()) * float(np.prod(spacing))  # in mm^3

        if (np.isfinite(surface_area) and np.isfinite(volume)) and surface_area > 0 and volume > 0:
            wt_sphericity = (np.pi ** (1/3)) * ((6 * volume) ** (2/3)) / surface_area
        else:
            wt_sphericity = np.nan

    if tc_mask.sum() > 0:
        verts, faces, _, _ = measure.marching_cubes(tc_mask.astype(np.uint8), 
                                                    level = 0.5, spacing = spacing,
                                                    allow_degenerate = False)

        mesh = trimesh.Trimesh(vertices = verts, faces = faces, process = True)

        surface_area = float(mesh.area)  # in mm^2
        if mesh.is_watertight and np.isfinite(mesh.volume):
            volume = float(abs(mesh.volume))
        else:
            volume = float(tc_mask.sum()) * float(np.prod(spacing))  # in mm^3

        if (np.isfinite(surface_area) and np.isfinite(volume)) and surface_area > 0 and volume > 0:
            tc_sphericity = (np.pi ** (1/3)) * ((6 * volume) ** (2/3)) / surface_area
        else:
            tc_sphericity = np.nan

    return {"whole_tumor_sphericity": wt_sphericity, "tumor_core_sphericity": tc_sphericity}

def _robust_zscore(x: np.ndarray, mask: np.ndarray | None = None, clip=(1, 99), eps=1e-8) -> np.ndarray:
    """Robust intensity normalization: percentile clip + median/MAD scaling."""
    if mask is None:
        mask = np.isfinite(x)
    vals = x[mask]
    if vals.size == 0:
        return x.astype(np.float32)
    lo, hi = np.percentile(vals, clip)
    xc = np.clip(x, lo, hi)
    med = np.median(xc[mask])
    mad = np.median(np.abs(xc[mask] - med))
    scale = 1.4826 * mad + eps
    return ((xc - med) / scale).astype(np.float32)


def count_components_3d(
    mask: np.ndarray,
    *,
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    connectivity: int = 2,  # 1=6-neigh, 2=18-neigh, 3=26-neigh
    min_component_mm3: float = 0.0,
) -> dict:
    """Connected-component count with optional physical-size filtering."""
    st = generate_binary_structure(3, connectivity)
    mask = mask.astype(bool)
    if not np.any(mask):
        return {"n_components": 0, "component_sizes_mm3": []}

    voxel_mm3 = float(spacing_mm[0] * spacing_mm[1] * spacing_mm[2])
    min_vox = int(np.ceil(max(min_component_mm3, 0.0) / voxel_mm3)) if voxel_mm3 > 0 else 0

    lab, n = label(mask, structure=st)
    if n == 0:
        return {"n_components": 0, "component_sizes_mm3": []}

    sizes_vox = [int(np.sum(lab == i)) for i in range(1, n + 1)]
    if min_vox > 0:
        sizes_vox = [s for s in sizes_vox if s >= min_vox]

    sizes_mm3 = [float(s * voxel_mm3) for s in sizes_vox]
    return {"n_components": int(len(sizes_vox)), "component_sizes_mm3": sizes_mm3}


def compute_rim_core_adjacency(
    seg: np.ndarray,
    *,
    enhancing_label: int = 4,      # BraTS ET
    nonenhancing_label: int = 1,   # BraTS NCR/NET (core)
    connectivity: int = 2,
) -> dict:
    """
    Rim–core adjacency: fraction of the non-enhancing core boundary that touches enhancing tumor.
    (High values are consistent with a ring/rim enhancement pattern around core.)
    """
    st = generate_binary_structure(3, connectivity)
    rim = (seg == enhancing_label)
    core = (seg == nonenhancing_label)

    if not np.any(core) or not np.any(rim):
        return {
            "rim_touches_core": False,
            "core_boundary_voxels": 0,
            "contact_voxels": 0,
            "adjacency_fraction": 0.0,
        }

    core_boundary = core & ~binary_erosion(core, structure=st, border_value=0)
    rim_dil = binary_dilation(rim, structure=st)
    contact = core_boundary & rim_dil

    denom = int(np.sum(core_boundary))
    num = int(np.sum(contact))
    return {
        "rim_touches_core": bool(num > 0),
        "core_boundary_voxels": denom,
        "contact_voxels": num,
        "adjacency_fraction": float(num / (denom + 1e-9)),
    }


def compute_boundary_sharpness(
    intensity: np.ndarray,
    tumor_mask: np.ndarray,
    *,
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    brain_mask: np.ndarray | None = None,
    connectivity: int = 2,
    shell_mm: tuple[float, float] = (2.0, 5.0),
) -> dict:
    """
    Boundary sharpness from intensity gradient magnitude at the tumor boundary,
    normalized by gradient magnitude in a peritumoral shell (reduces sensitivity to image texture/noise).

    Score = median(|∇I| on boundary) / median(|∇I| in outside shell).
    """
    st = generate_binary_structure(3, connectivity)
    tumor_mask = tumor_mask.astype(bool)

    if not np.any(tumor_mask):
        return {"boundary_sharpness": float("nan"), "reason": "Empty tumor mask."}

    if brain_mask is None:
        # Conservative default (matches typical NIfTI background=0)
        brain_mask = np.isfinite(intensity) & (intensity != 0)

    x = _robust_zscore(intensity, mask=brain_mask)

    gx, gy, gz = np.gradient(x, spacing_mm[0], spacing_mm[1], spacing_mm[2])
    grad = np.sqrt(gx * gx + gy * gy + gz * gz)

    boundary = tumor_mask & ~binary_erosion(tumor_mask, structure=st, border_value=0)

    outside = (~tumor_mask) & brain_mask
    # distance from outside voxels to nearest tumor voxel (0 distance at boundary-adjacent outside voxels)
    dt_out = distance_transform_edt(outside, sampling=spacing_mm)
    shell = outside & (dt_out >= float(shell_mm[0])) & (dt_out <= float(shell_mm[1]))

    bvals = grad[boundary]
    svals = grad[shell]

    if bvals.size == 0:
        return {"boundary_sharpness": float("nan"), "reason": "No boundary voxels."}

    shell_med = float(np.median(svals)) if svals.size else 0.0
    score = float(np.median(bvals) / (shell_med + 1e-9))

    return {
        "boundary_sharpness": score,
        "boundary_grad_median": float(np.median(bvals)),
        "boundary_grad_iqr": float(np.percentile(bvals, 75) - np.percentile(bvals, 25)),
        "peritumoral_shell_grad_median": shell_med,
        "n_boundary_vox": int(np.sum(boundary)),
        "n_shell_vox": int(np.sum(shell)),
        "shell_mm": (float(shell_mm[0]), float(shell_mm[1])),
    }

def compute_vasari_style_morphometrics(
    tumorseg_ss_path: str,
    *,
    # optional intensity image for boundary sharpness:
    intensity_path: str | None = None,
    brain_mask_path: str | None = None,
    enhancing_label: int = 4,
    nonenhancing_label: int = 1,
    oedema_label: int = 2,
    connectivity: int = 2,
    min_component_mm3: float = 250.0,   # helps ignore tiny islands / label noise
    shell_mm: tuple[float, float] = (2.0, 5.0),
) -> dict:
    """
    Computes:
      - boundary sharpness (if intensity_path given)
      - rim–core adjacency (ET touching NCR boundary)
      - number of components for enhancing (ET) and non-enhancing (NCR)
    using the same conventions/classes as your VASARI code.
    """
    seg_img = NiftiImage(tumorseg_ss_path)
    seg = seg_img.array.astype(np.int16)
    spacing = tuple(float(s) for s in seg_img.spacing)

    et = (seg == enhancing_label)
    ncr = (seg == nonenhancing_label)
    tumor = (seg > 0)  # BraTS tumor union (ET/NCR/ED)

    out = {
        "spacing_mm": spacing,
        "n_components_enhancing": None,
        "n_components_nonenhancing": None,
        "rim_core_adjacency": None,
        "boundary_sharpness": None,
    }

    # --- number of components (robust physical-size filtering) ---
    out["n_components_enhancing"] = count_components_3d(
        et, spacing_mm=spacing, connectivity=connectivity, min_component_mm3=min_component_mm3
    )
    out["n_components_nonenhancing"] = count_components_3d(
        ncr, spacing_mm=spacing, connectivity=connectivity, min_component_mm3=min_component_mm3
    )

    # --- rim-core adjacency ---
    out["rim_core_adjacency"] = compute_rim_core_adjacency(
        seg, enhancing_label=enhancing_label, nonenhancing_label=nonenhancing_label, connectivity=connectivity
    )

    # --- boundary sharpness (requires intensity image) ---
    if intensity_path is not None:
        I = NiftiImage(intensity_path).array.astype(np.float32)
        brain_mask = None
        if brain_mask_path is not None:
            brain_mask = (NiftiImage(brain_mask_path).array != 0)

        out["boundary_sharpness"] = compute_boundary_sharpness(
            I, tumor, spacing_mm=spacing, brain_mask=brain_mask, connectivity=connectivity, shell_mm=shell_mm
        )

    return out
