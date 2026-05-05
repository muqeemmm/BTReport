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
    binary_closing,
    generate_binary_structure,
    distance_transform_edt,
)
import attrs
from .transition_zone_thickness import (
    transition_zone_thickness_distance_transform,
    transition_zone_thickness_raycast,
)

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

def _robust_zscore(
    x: np.ndarray,
    mask: np.ndarray | None = None,
    clip: tuple[float, float] | None = (1, 99),
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Robust intensity normalization: optional percentile clip + median/MAD scaling.

    Pass ``clip=None`` to skip percentile clipping (useful when downstream code
    needs to preserve the high-intensity tail, e.g. for gradient computations
    across sharp tissue boundaries where clipping would compress the very
    contrast we want to measure).
    """
    if mask is None:
        mask = np.isfinite(x)
    vals = x[mask]
    if vals.size == 0:
        return x.astype(np.float32)
    if clip is not None:
        lo, hi = np.percentile(vals, clip)
        xc = np.clip(x, lo, hi)
    else:
        xc = x
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

def _empty_boundary_sharpness_result(
    shell_mm: tuple[float, float],
    reason: str,
) -> dict:
    """Standardized return dict for boundary-sharpness failure modes (keeps the
    schema stable so downstream DataFrame aggregation never hits KeyError)."""
    return {
        "boundary_sharpness": float("nan"),
        "boundary_grad_median": float("nan"),
        "boundary_grad_p90": float("nan"),
        "boundary_grad_iqr": float("nan"),
        "peritumoral_shell_grad_median": float("nan"),
        "peritumoral_shell_grad_p90": float("nan"),
        "n_boundary_vox": 0,
        "n_shell_vox": 0,
        "shell_mm": (float(shell_mm[0]), float(shell_mm[1])),
        "reason": reason,
    }


def compute_boundary_sharpness(
    intensity: np.ndarray,
    tumor_mask: np.ndarray,
    *,
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    brain_mask: np.ndarray | None = None,
    connectivity: int = 1,
    shell_mm: tuple[float, float] = (2.0, 5.0),
    normalize: bool = True,
    shell_grad_floor: float = 1e-6,
) -> dict:
    """
    Relative gradient ratio at the tumor boundary (a proxy for "boundary
    sharpness"), comparing intensity gradient magnitude on a thin band straddling
    the tumor surface against the gradient magnitude in a peritumoral shell.

    Score = median(|∇I| on boundary band) / median(|∇I| in outside shell).

    Notes & caveats
    ---------------
    * The score is a unitless ratio, not a calibrated physical measurement, and
      should not be compared directly across protocols/scanners without
      harmonization.
    * ``intensity`` axes are assumed to align with ``spacing_mm`` element-wise
      (i.e. ``spacing_mm[i]`` is the spacing along ``intensity.shape[i]``). If
      your loader returns a permuted ``spacing`` tuple this will silently
      corrupt the gradient magnitude.
    * ``shell_mm = (2.0, 5.0)`` mm is a default; on highly anisotropic data the
      inner offset may be only one voxel thick along the coarse axis. Override
      explicitly when voxel size deviates strongly from ~1 mm³.

    Parameters
    ----------
    intensity   : 3-D intensity volume.
    tumor_mask  : Boolean (or coercible) tumor mask in the same grid as ``intensity``.
    spacing_mm  : Voxel spacing per axis, matching ``intensity.shape`` order.
    brain_mask  : Brain mask in the same grid; strongly recommended. If None, a
                  conservative default of ``isfinite(intensity) & (intensity != 0)``
                  is used and a warning is logged.
    connectivity: Structuring-element connectivity for the inner boundary erosion.
    shell_mm    : (min_dist, max_dist) in mm defining the peritumoral shell.
    normalize   : If True, apply median/MAD normalization (no percentile clip)
                  before differentiation. The ratio cancels global scale either
                  way; normalization mainly stabilizes the per-subject absolute
                  gradient values that are also returned.
    shell_grad_floor : If the peritumoral shell median gradient falls below
                  this floor, the score is returned as NaN (rather than blowing
                  up via a tiny denominator).

    Returns
    -------
    dict with a fixed schema (NaNs in failure cases) — see
    ``_empty_boundary_sharpness_result``. Includes p90 statistics in addition
    to medians/IQR because the discriminative signal for "sharp vs diffuse"
    edges sits in the upper tail of |∇I|.
    """
    tumor_mask = tumor_mask.astype(bool)

    if not np.any(tumor_mask):
        return _empty_boundary_sharpness_result(shell_mm, reason="Empty tumor mask.")

    if intensity.shape != tumor_mask.shape:
        return _empty_boundary_sharpness_result(
            shell_mm,
            reason=f"Shape mismatch: intensity {intensity.shape} vs mask {tumor_mask.shape}.",
        )

    if brain_mask is None:
        logger.warning(
            "compute_boundary_sharpness: brain_mask not provided; falling back to "
            "(intensity != 0) & isfinite. This is unreliable for non-skull-stripped "
            "or bias-corrected volumes — pass brain_mask explicitly when possible."
        )
        brain_mask = np.isfinite(intensity) & (intensity != 0)
    else:
        brain_mask = brain_mask.astype(bool)
        if brain_mask.shape != intensity.shape:
            return _empty_boundary_sharpness_result(
                shell_mm,
                reason=f"brain_mask shape {brain_mask.shape} does not match intensity {intensity.shape}.",
            )

    # Optional median/MAD normalization (no percentile clip — clipping would
    # compress the tumor/edema contrast we are trying to measure).
    if normalize:
        x = _robust_zscore(intensity, mask=brain_mask, clip=None)
    else:
        x = intensity.astype(np.float32)

    gx, gy, gz = np.gradient(x, spacing_mm[0], spacing_mm[1], spacing_mm[2])
    grad = np.sqrt(gx * gx + gy * gy + gz * gz)

    # --- Boundary band: one voxel inside + one voxel outside the surface.
    # A pure inner ring under-samples the gradient peak, which often straddles
    # the partial-volume voxel just outside the mask, especially at oblique
    # interfaces.
    st = generate_binary_structure(3, connectivity)
    inner_ring = tumor_mask & ~binary_erosion(tumor_mask, structure=st, border_value=0)
    outer_ring = binary_dilation(tumor_mask, structure=st) & ~tumor_mask & brain_mask
    boundary = inner_ring | outer_ring

    # --- Peritumoral shell: distance to the *tumor* (not "distance to the
    # nearest non-tumor voxel within the outside region"), then intersect with
    # brain. The previous implementation passed `outside` to the EDT which
    # produced a distance-to-(tumor OR brain-edge) map, biasing the shell near
    # peripheral lesions.
    dt_to_tumor = distance_transform_edt(~tumor_mask, sampling=spacing_mm)
    
    # Reference region
    shell = (
        brain_mask
        & ~tumor_mask
        & (dt_to_tumor >= float(shell_mm[0]))
        & (dt_to_tumor <= float(shell_mm[1]))
    )

    bvals = grad[boundary]
    svals = grad[shell]

    if bvals.size == 0:
        return _empty_boundary_sharpness_result(shell_mm, reason="No boundary voxels.")
    if svals.size == 0:
        return _empty_boundary_sharpness_result(
            shell_mm,
            reason="Empty peritumoral shell (lesion may abut brain edge — try smaller shell_mm).",
        )

    boundary_med = float(np.median(bvals))
    boundary_p90 = float(np.percentile(bvals, 90))
    boundary_iqr = float(np.percentile(bvals, 75) - np.percentile(bvals, 25))
    shell_med = float(np.median(svals))
    shell_p90 = float(np.percentile(svals, 90))

    if not np.isfinite(shell_med) or shell_med < shell_grad_floor:
        score = float("nan")
    else:
        score = boundary_med / shell_med

    return {
        "boundary_sharpness": score,
        "boundary_grad_median": boundary_med,
        "boundary_grad_p90": boundary_p90,
        "boundary_grad_iqr": boundary_iqr,
        "peritumoral_shell_grad_median": shell_med,
        "peritumoral_shell_grad_p90": shell_p90,
        "n_boundary_vox": int(np.sum(boundary)),
        "n_shell_vox": int(np.sum(shell)),
        "shell_mm": (float(shell_mm[0]), float(shell_mm[1])),
        "reason": "ok",
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

def compute_transition_zone_thickness(
    seg_path: str,
    flair_path: str | None = None,
    t1ce_path: str | None = None,
    method: str = "method_a",
) -> dict:
    """
    Wrapper around the two transition zone thickness methods.

    Parameters
    ----------
    seg_path   : path to BraTS-style segmentation NIfTI.
    flair_path : path to FLAIR NIfTI — required for method_b.
    t1ce_path  : path to T1CE NIfTI — used by method_b to compute a separate
                 intensity-aware thickness in the T1CE sequence.
    method     : "method_a" (distance-transform, default, recommended) or
                 "method_b" (ray-casting along outward normals, intensity-aware).

    Returns
    -------
    For method_a: dict with key "Transition Zone Thickness" (geometric, modality-independent).
    For method_b: dict with keys "Transition Zone Thickness (FLAIR)" and, if t1ce_path
    is provided, "Transition Zone Thickness (T1CE)".
    The thickness_map array (Method A) is excluded as it is not JSON-serialisable.
    """
    seg_img = NiftiImage(seg_path, canonical=True)
    seg_array = seg_img.array.astype(np.int16)
    voxel_spacing = tuple(float(s) for s in seg_img.spacing)

    if method == "method_a":
        result = transition_zone_thickness_distance_transform(seg_array, voxel_spacing)
        result.pop("thickness_map", None)  # ndarray — not JSON-serialisable
        return {"Transition Zone Thickness": result}

    elif method == "method_b":
        if flair_path is None:
            raise ValueError("flair_path must be provided for method_b (ray-casting).")
        flair_img = NiftiImage(flair_path, canonical=True)
        flair_array = flair_img.array.astype(np.float32)
        flair_result = transition_zone_thickness_raycast(flair_array, seg_array, voxel_spacing)

        out = {"Transition Zone Thickness (FLAIR)": flair_result}

        if t1ce_path is not None:
            t1ce_img = NiftiImage(t1ce_path, canonical=True)
            t1ce_array = t1ce_img.array.astype(np.float32)
            t1ce_result = transition_zone_thickness_raycast(t1ce_array, seg_array, voxel_spacing)
            out["Transition Zone Thickness (T1CE)"] = t1ce_result

        return out

    else:
        raise ValueError(
            f"Unknown tz_method '{method}'. Choose 'method_a' or 'method_b'."
        )

# ---------------------------------------------------------------------------------------------- #
@attrs.define
class ExtractT2FLAIRMismatch:

    verbose: bool = False

    # BraTS-style labels
    enhancing_label: int = 4
    nonenhancing_label: int = 1
    oedema_label: int = 2

    # thresholds / heuristics
    min_core_voxels: int = 100  # tumor core of the tumor
    min_center_voxels: int = 30 # non-enhancing part of the tumor
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

        dt = distance_transform_edt(~core_mask)

        # center = deeper half of the lesion thickness (at least >0)
        max_dt = float(dt.max())
        if max_dt <= 0:
            return np.zeros_like(core_mask, dtype=bool), np.zeros_like(core_mask, dtype=bool)

        center_mask = core_mask & (dt >= 0.10 * max_dt)
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
        wm_left = np.isin(merged_seg_array, [2, 7])     # left hemisphere white matter
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