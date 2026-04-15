import trimesh, ants
import numpy as np
import pandas as pd
import time
import os, logging
from typing import Tuple
from skimage import measure
from vasari_features.vasari_auto_v2 import NiftiImage
from scipy.ndimage import (
    label,
    binary_dilation,
    binary_erosion,
    generate_binary_structure,
    distance_transform_edt,
    map_coordinates,
)
import attrs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

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

# ---------------------------------------------------------------------------------------------- #

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
    seg     = seg_img.array.astype(np.int16)
    spacing = tuple(float(s) for s in seg_img.spacing)

    et      = (seg == enhancing_label)
    ncr     = (seg == nonenhancing_label)
    tumor   = (seg > 0)  # BraTS tumor union (ET/NCR/ED)

    out = {
        "spacing_mm": spacing,
        "n_components_enhancing": None,
        "n_components_nonenhancing": None,
        "rim_core_adjacency": None,
        "boundary_sharpness": None,
    }

    # --- number of components (robust physical-size filtering) ---
    out["n_components_enhancing"] = count_components_3d(
        et, spacing_mm = spacing, connectivity = connectivity, min_component_mm3 = min_component_mm3
    )
    out["n_components_nonenhancing"] = count_components_3d(
        ncr, spacing_mm = spacing, connectivity = connectivity, min_component_mm3 = min_component_mm3
    )

    # --- rim-core adjacency ---
    out["rim_core_adjacency"] = compute_rim_core_adjacency(
        seg, enhancing_label = enhancing_label, nonenhancing_label = nonenhancing_label, connectivity = connectivity
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
# ---------------------------------------------------------------------------------------------- #

def compute_transition_zone_thickness(t1ce_pth, seg_pth,
                              step_vox = 0.25, max_dist_mm = 80.0,) -> float:
    
    t1ce_img    = NiftiImage(t1ce_pth)
    t1ce_img    = t1ce_img.array.astype(np.float32)

    seg_img   = NiftiImage(seg_pth)
    seg_array = seg_img.array.astype(np.int16)
    spacing   = tuple(float(s) for s in seg_img.spacing)

    # Define tumor regions
    tc_mask     = np.isin(seg_array, [1, 3, 4]).astype(bool)          # Tumor Core
    ed_mask     = np.isin(seg_array, [2]).astype(bool)                # Edema

    # Helper: nearest-neighbor membership test for floating (z,y,x)
    def nn_mask(mask, p):
        iz, iy, ix = np.round(p).astype(int)
        if (iz < 0 or iy < 0 or ix < 0 or
            iz >= mask.shape[0] or iy >= mask.shape[1] or ix >= mask.shape[2]):
            return False
        return bool(mask[iz, iy, ix])

    # 1) boundary voxels: enhancing adjacent to edema
    edema_dil = binary_dilation(ed_mask, iterations = 1)
    boundary  = tc_mask & edema_dil & ~ed_mask

    boundary_coords = np.argwhere(boundary)
    if boundary_coords.size == 0:
        return np.nan

    # 2) edema intensity band 
    ed_vals = t1ce_img[ed_mask]
    if ed_vals.size == 0:
        return np.nan

    ed_med = np.median(ed_vals)
    mad    = np.median(np.abs(ed_vals - ed_med)) + 1e-6
    abs_band = max(1.4826 * mad, 1e-6)  # robust sigma-like width
    ed_lower, ed_upper = ed_med - abs_band, ed_med + abs_band

    if ed_lower < np.min(ed_vals):
        ed_lower = np.min(ed_vals)
        
    if ed_upper > np.max(ed_vals):
        ed_upper = np.max(ed_vals)

    # 3) ray direction = nearest edema voxel
    inv_edema = ~ed_mask
    _, (zi, yi, xi) = distance_transform_edt(inv_edema, return_indices = True, sampling = spacing)
    nearest_edema_idx = np.stack([zi, yi, xi], axis = -1)

    spacing = np.asarray(spacing, dtype=float)  # (z,y,x)

    thicknesses = []
    for zyx in boundary_coords:
        z0, y0, x0 = zyx.astype(float)

        ze, ye, xe = nearest_edema_idx[tuple(zyx)]
        v = np.array([ze - z0, ye - y0, xe - x0], dtype=float)
        nv = np.linalg.norm(v)
        if nv == 0:
            thicknesses.append(0.0)
            continue

        u = v / nv
        mm_per_step = np.linalg.norm(u * spacing)
        if mm_per_step == 0:
            continue
        n_steps = int(np.ceil(max_dist_mm / (step_vox * mm_per_step)))

        p0 = np.array([z0, y0, x0], dtype=float)
        prev_int = map_coordinates(t1ce_img, [[z0], [y0], [x0]], order = 1, mode = "nearest")[0]
        prev_d_mm = 0.0

        # only accept a "hit" if we're in edema OR we've exited tc_mask
        in_edema0 = nn_mask(ed_mask, p0)
        out_of_tc0 = not nn_mask(tc_mask, p0)
        if (ed_lower <= prev_int <= ed_upper) and (in_edema0 or out_of_tc0):
            thicknesses.append(prev_d_mm)
            continue

        for k in range(1, n_steps + 1):
            p = p0 + u * (k * step_vox)
            I = map_coordinates(t1ce_img, [[p[0]], [p[1]], [p[2]]], order=1, mode="nearest")[0]
            d_mm = k * step_vox * np.linalg.norm(u * spacing)

            in_edema = nn_mask(ed_mask, p)
            out_of_tc = not nn_mask(tc_mask, p)

            if (ed_lower <= I <= ed_upper) and (in_edema or out_of_tc):
                # optional linear interpolation between prev and current
                if abs(I - prev_int) > 1e-6:
                    if prev_int > ed_upper and I <= ed_upper:
                        target = ed_upper
                    elif prev_int < ed_lower and I >= ed_lower:
                        target = ed_lower
                    else:
                        target = ed_med  # fallback
                    alpha = (target - prev_int) / (I - prev_int + 1e-12)
                    alpha = float(np.clip(alpha, 0.0, 1.0))
                    d_mm = prev_d_mm + alpha * (d_mm - prev_d_mm)
                thicknesses.append(d_mm)
                break

            prev_int, prev_d_mm = I, d_mm

    return {"transition_zone_thickness_in_mm": float(np.nanmedian(thicknesses)) if len(thicknesses) else np.nan}

# ---------------------------------------------------------------------------------------------- #
@attrs.define
class ExtractT2FLAIRMismatch:

    verbose: bool = False

    # BraTS-style labels
    enhancing_label: int = 4
    nonenhancing_label: int = 1
    oedema_label: int = 2

    # thresholds / heuristics
    min_core_voxels: int = 100
    min_center_voxels: int = 30
    min_rim_voxels: int = 30
    min_wt_voxels: int = 100

    # normalized intensity thresholds
    center_t2_high_thresh: float = 0.60
    center_flair_low_thresh: float = 0.50
    rim_flair_high_thresh: float = 0.55

    # decision threshold for final binary flag
    mismatch_score_thresh: float = 0.12

    COL_NAMES = [
        "T2-FLAIR Mismatch Present",
        "T2-FLAIR Mismatch Score",
        "T2-FLAIR Mismatch Degree",
        "Central Core T2 Mean",
        "Central Core FLAIR Mean",
        "Peripheral Rim FLAIR Mean",
        "Tumor Core Volume (mL)",
        "Whole Tumor Volume (mL)",
    ]

    def robust_normalize(self, img: np.ndarray, brain_mask: np.ndarray) -> np.ndarray:
        """
        Robust min-max normalization to [0,1] inside a brain mask.
        """
        arr = np.asarray(img, dtype=np.float32)
        vals = arr[brain_mask]

        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return np.zeros_like(arr, dtype=np.float32)

        lo = np.percentile(vals, 1)
        hi = np.percentile(vals, 99)

        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros_like(arr, dtype=np.float32)

        arr = np.clip(arr, lo, hi)
        arr = (arr - lo) / (hi - lo + 1e-8)
        return arr.astype(np.float32)

    def get_whole_tumor_mask(self, segmentation: NiftiImage) -> np.ndarray:
        """
        Whole tumor = NCR/NET + ED + ET
        """
        return np.isin(
            segmentation.array,
            [self.nonenhancing_label, self.oedema_label, self.enhancing_label],
        )

    def get_tumor_core_mask(self, segmentation: NiftiImage) -> np.ndarray:
        """
        Tumor core = NCR/NET + ET
        Excludes edema.
        """
        return np.isin(
            segmentation.array,
            [self.nonenhancing_label, self.enhancing_label],
        )

    def get_center_and_rim_masks(self, core_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Split tumor core into a central component and a peripheral rim.

        - center: voxels sufficiently far from the boundary
        - rim: shell near the boundary

        Uses a distance transform so it behaves more consistently across lesion sizes.
        """
        core_mask = core_mask.astype(bool)

        if not core_mask.any():
            return np.zeros_like(core_mask, dtype=bool), np.zeros_like(core_mask, dtype=bool)

        dt = distance_transform_edt(core_mask)

        # center = deeper half of the lesion thickness (at least >0)
        max_dt = float(dt.max())
        if max_dt <= 0:
            return np.zeros_like(core_mask, dtype=bool), np.zeros_like(core_mask, dtype=bool)

        center_mask = core_mask & (dt >= 0.25 * max_dt)
        rim_mask = core_mask & (~center_mask)

        # fallback if center/rim are too small
        if center_mask.sum() < self.min_center_voxels:
            eroded = binary_erosion(core_mask, iterations=1)
            if eroded.sum() > 0:
                center_mask = eroded
                rim_mask = core_mask & (~center_mask)

        return center_mask, rim_mask

    def compute_mismatch_score(
        self,
        t2_norm: np.ndarray,
        flair_norm: np.ndarray,
        center_mask: np.ndarray,
        rim_mask: np.ndarray,
    ) -> tuple[float, float, float, float]:
        """
        Continuous heuristic score for T2-FLAIR mismatch.

        Desired pattern:
        - central T2 high
        - central FLAIR relatively low
        - peripheral rim FLAIR relatively high

        Score is higher when this pattern is stronger.
        """
        if center_mask.sum() == 0 or rim_mask.sum() == 0:
            return np.nan, np.nan, np.nan, np.nan

        center_t2_mean = float(np.mean(t2_norm[center_mask]))
        center_flair_mean = float(np.mean(flair_norm[center_mask]))
        rim_flair_mean = float(np.mean(flair_norm[rim_mask]))

        # components of the mismatch pattern
        central_t2_component = max(0.0, center_t2_mean - self.center_t2_high_thresh)
        central_suppression_component = max(0.0, self.center_flair_low_thresh - center_flair_mean)
        rim_component = max(0.0, rim_flair_mean - self.rim_flair_high_thresh)

        mismatch_score = (
            central_t2_component
            + central_suppression_component
            + rim_component
        ) / 3.0

        return (
            float(mismatch_score),
            float(center_t2_mean),
            float(center_flair_mean),
            float(rim_flair_mean),
        )

    def extract_t2_flair_mismatch(
        self,
        tumorseg_ss: str,
        t2_path: str,
        flair_path: str,
        brain_mask_path: str,
        laterility: str | None = None,
        merged_seg: str | None = None,
    ) -> pd.DataFrame:
        """
        Run T2-FLAIR mismatch extraction in subject space.

        Inputs should all be in the same space:
        - tumorseg_ss
        - T2
        - FLAIR
        """
        start_time = time.time()

        segmentation = NiftiImage(tumorseg_ss)
        merged_segmentation = NiftiImage(merged_seg)
        t2 = NiftiImage(t2_path)
        flair = NiftiImage(flair_path)

        brain_mask = NiftiImage(brain_mask_path).array.astype(bool)

        whole_tumor_mask = self.get_whole_tumor_mask(segmentation)
        tumor_core_mask = self.get_tumor_core_mask(segmentation)

        merged_seg_array = merged_segmentation.array.astype(np.int16)
        wm_left = np.isin(merged_seg_array, [2, 7])  # left hemisphere white matter
        wm_right = np.isin(merged_seg_array, [41, 46])  # right hemisphere white matter
        if laterility == "left":
            cnwm_mask = wm_right
        elif laterility == "right":
            cnwm_mask = wm_left
        else:
            cnwm_mask = wm_left | wm_right

        voxel_volume_mm3 = float(np.prod(segmentation.spacing))
        whole_tumor_volume_ml = float(whole_tumor_mask.sum() * voxel_volume_mm3 / 1000.0)
        tumor_core_volume_ml = float(tumor_core_mask.sum() * voxel_volume_mm3 / 1000.0)

        if self.verbose:
            logger.debug(f"Whole tumor voxels: {int(whole_tumor_mask.sum())}")
            logger.debug(f"Tumor core voxels: {int(tumor_core_mask.sum())}")

        result = pd.DataFrame(columns=self.COL_NAMES)

        if tumor_core_mask.sum() < self.min_core_voxels:
            if self.verbose:
                logger.debug("Tumor core too small for reliable T2-FLAIR mismatch assessment")

            result.loc[len(result)] = {
                "T2-FLAIR Mismatch Present": np.nan,
                "T2-FLAIR Mismatch Score": np.nan,
                "T2-FLAIR Mismatch Degree": np.nan,
                "Central Core T2 Mean": np.nan,
                "Central Core FLAIR Mean": np.nan,
                "Peripheral Rim FLAIR Mean": np.nan,
                "Tumor Core Volume (mL)": tumor_core_volume_ml,
                "Whole Tumor Volume (mL)": whole_tumor_volume_ml,
            }
            return result

        t2_norm = self.robust_normalize(t2.array, brain_mask)
        flair_norm = self.robust_normalize(flair.array, brain_mask)

        # Add assert statement later on

        center_mask, rim_mask = self.get_center_and_rim_masks(tumor_core_mask)

        if self.verbose:
            logger.debug(f"Center voxels: {int(center_mask.sum())}")
            logger.debug(f"Rim voxels: {int(rim_mask.sum())}")

        if center_mask.sum() < self.min_center_voxels or rim_mask.sum() < self.min_rim_voxels:
            mismatch_score = np.nan
            center_t2_mean = np.nan
            center_flair_mean = np.nan
            rim_flair_mean = np.nan
            mismatch_present = np.nan
        else:
            (
                mismatch_score,
                center_t2_mean,
                center_flair_mean,
                rim_flair_mean,
            ) = self.compute_mismatch_score(
                t2_norm=t2_norm,
                flair_norm=flair_norm,
                center_mask=center_mask,
                rim_mask=rim_mask,
            )

            # Binary decision:
            # central T2 high, central FLAIR lower, rim FLAIR higher, and overall score above threshold
            mismatch_present = int(
                (center_t2_mean >= self.center_t2_high_thresh)
                and (center_flair_mean <= self.center_flair_low_thresh)
                and (rim_flair_mean >= self.rim_flair_high_thresh)
                and (mismatch_score >= self.mismatch_score_thresh)
            )
        
            # Mismatch degree
            t2_tumour_mean = float(np.mean(t2.array[tumor_core_mask]))
            flair_tumour_mean = float(np.mean(flair.array[tumor_core_mask]))
            t2_cnwm_mean = float(np.mean(t2.array[cnwm_mask]))
            flair_cnwm_mean = float(np.mean(flair.array[cnwm_mask]))

            mismatch_degree = (t2_tumour_mean / (t2_cnwm_mean + 1e-8)) - (flair_tumour_mean / (flair_cnwm_mean + 1e-8))


        result.loc[len(result)] = {
            "T2-FLAIR Mismatch Present": mismatch_present,
            "T2-FLAIR Mismatch Score": mismatch_score,
            "T2-FLAIR Mismatch Degree": mismatch_degree,
            "Central Core T2 Mean": center_t2_mean,
            "Central Core FLAIR Mean": center_flair_mean,
            "Peripheral Rim FLAIR Mean": rim_flair_mean,
            "Tumor Core Volume (mL)": tumor_core_volume_ml,
            "Whole Tumor Volume (mL)": whole_tumor_volume_ml,
        }

        end_time = time.time()
        time_taken = np.round(end_time - start_time, 2)

        if self.verbose:
            logger.debug("Time taken: " + str(time_taken) + " seconds")
            logger.debug(f"T2-FLAIR mismatch result: {result.iloc[0].to_dict()}")

        return result

    def __call__(self, tumorseg_ss: str, t2_path: str, flair_path: str, laterility: str, merged_seg: str, brain_mask_path: str) -> dict:
        report = self.extract_t2_flair_mismatch(
            tumorseg_ss=tumorseg_ss,
            t2_path=t2_path,
            flair_path=flair_path,
            laterility=laterility,
            merged_seg=merged_seg,
            brain_mask_path=brain_mask_path
        )
        logger.info("* Finished T2-FLAIR mismatch extraction!")
        return report.iloc[0].to_dict()