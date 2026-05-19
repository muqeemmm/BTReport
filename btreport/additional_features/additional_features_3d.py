from __future__ import annotations

import trimesh
import numpy as np
import pandas as pd
import time
import os, logging
from typing import Tuple
from skimage import measure
from ..vasari_features.vasari_auto_v2 import NiftiImage
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
    method: str = "method_b",
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
    seg_img = NiftiImage(seg_path)
    seg_array = seg_img.array.astype(np.int16)
    voxel_spacing = tuple(float(s) for s in seg_img.spacing)

    if method == "method_a":
        result = transition_zone_thickness_distance_transform(seg_array, voxel_spacing)
        result.pop("thickness_map", None)  # ndarray — not JSON-serialisable
        return {"Transition Zone Thickness": result}

    elif method == "method_b":
        if flair_path is None:
            raise ValueError("flair_path must be provided for method_b (ray-casting).")
        flair_img = NiftiImage(flair_path)
        flair_array = flair_img.array.astype(np.float32)
        flair_result = transition_zone_thickness_raycast(flair_array, seg_array, voxel_spacing)

        out = {"Transition Zone Thickness (FLAIR)": flair_result}

        if t1ce_path is not None:
            t1ce_img = NiftiImage(t1ce_path)
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
    """
    T2-FLAIR mismatch extractor.

    Pipeline (all calibration baked in; no toggles):
        * Lesion mask = BraTS Tumor Core (NCR + ET), with a fallback to
          Whole Tumor (NCR + ED + ET) when TC < 1 mL or TC/WT < 0.10.
        * Center / rim split (mm-based): center = deepest
          max(3 mm, 50% of max inward depth) of the lesion; rim =
          lesion minus the dilated center.
        * Intensity normalisation: contralateral normal white matter
          (CNWM) from SynthSeg labels (2/7 left, 41/46 right), with
          bilateral-WM and brain-median fallbacks.
        * Decision: weighted soft score
              0.20 * s_t2 + 0.60 * s_supp + 0.20 * s_rim   >= 0.70
          on the CNWM-ratio scale.

    Validated on AKU-WHO (62 positive + 62 sampled negative subjects,
    6 negative-sample seeds): sensitivity 0.726, specificity 0.74 +- 0.04,
    Youden J 0.47 +- 0.04.
    """

    verbose: bool = False

    # BraTS-style labels
    enhancing_label: int = 4
    nonenhancing_label: int = 1
    oedema_label: int = 2

    # Size guards (voxels). One lower bound on the lesion (post-fallback),
    # one each on the center and rim sub-regions.
    min_lesion_voxels: int = 100
    min_center_voxels: int = 10
    min_rim_voxels: int = 10

    # TC -> WT fallback when the segmenter collapses TC into ED
    min_core_volume_ml: float = 1.0
    tc_wt_ratio_floor: float = 0.10

    # Center / rim geometry (mm-based)
    center_inward_mm: float = 3.0
    center_depth_fraction: float = 0.50

    # CNWM-ratio anchor for the central-FLAIR-suppression ramp.
    # The suppression ramp runs from 0 at (anchor + 0.30) down to 1 at
    # (anchor + 0.30 - soft_supp_span), i.e. saturates fully suppressed at
    # ratio ~ 1.25 with this configuration.
    ratio_center_flair_low: float = 1.55

    # Soft-score ramp spans. The s_t2 and s_rim ramps run from 1.0 (CNWM
    # parity) to 1.0 + span; s_supp uses the dedicated anchor above.
    soft_t2_span: float = 0.50
    soft_supp_span: float = 0.60
    soft_rim_span: float = 0.50

    # Decision threshold on the weighted soft score
    soft_score_thresh: float = 0.70

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

    def get_whole_tumor_mask(self, segmentation: NiftiImage) -> np.ndarray:
        """Whole tumor = NCR/NET + ED + ET."""
        return np.isin(
            segmentation.array,
            [self.nonenhancing_label, self.oedema_label, self.enhancing_label],
        )

    def get_tumor_core_mask(self, segmentation: NiftiImage) -> np.ndarray:
        """Tumor core = NCR/NET + ET (excludes edema)."""
        return np.isin(
            segmentation.array,
            [self.nonenhancing_label, self.enhancing_label],
        )

    def get_center_and_rim_masks(
        self,
        lesion_mask: np.ndarray,
        spacing: tuple[float, float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Split the lesion into a small deep center and a thick peripheral rim.

        Distances are in mm so the split is scanner-agnostic.
        - center: inward depth >= max(center_inward_mm, center_depth_fraction * max_depth)
        - rim:    lesion minus the dilated center (the two regions never touch)
        """
        lesion_mask = lesion_mask.astype(bool)
        if not lesion_mask.any():
            empty = np.zeros_like(lesion_mask, dtype=bool)
            return empty, empty

        dt_mm = distance_transform_edt(lesion_mask, sampling=spacing)
        max_mm = float(dt_mm.max())
        if max_mm <= 0:
            empty = np.zeros_like(lesion_mask, dtype=bool)
            return empty, empty

        cutoff = max(self.center_inward_mm, self.center_depth_fraction * max_mm)
        center_mask = lesion_mask & (dt_mm >= cutoff)

        # Tiny-lesion fallback: keep the deepest 5% of voxels by inward depth.
        if center_mask.sum() < self.min_center_voxels:
            inside = dt_mm[lesion_mask]
            if inside.size:
                q = np.percentile(inside, 95)
                center_mask = lesion_mask & (dt_mm >= q)

        rim_mask = lesion_mask & ~binary_dilation(center_mask, iterations=1)
        return center_mask, rim_mask

    def get_cnwm_mask(
        self,
        merged_seg_array: np.ndarray,
        laterality: str | None,
    ) -> np.ndarray:
        """Contralateral normal WM mask from SynthSeg labels; bilateral fallback."""
        wm_left = np.isin(merged_seg_array, [2, 7])
        wm_right = np.isin(merged_seg_array, [41, 46])
        if laterality == "left":
            mask = wm_right
        elif laterality == "right":
            mask = wm_left
        else:
            mask = wm_left | wm_right
        if mask.sum() == 0:
            mask = wm_left | wm_right
        return mask

    @staticmethod
    def _ramp(x: float, lo: float, span: float) -> float:
        """Linear ramp from 0 at x=lo to 1 at x=lo+span, clipped to [0,1]."""
        if span <= 0:
            return 1.0 if x >= lo else 0.0
        return float(np.clip((x - lo) / span, 0.0, 1.0))

    def compute_mismatch_score(
        self,
        center_t2_ratio: float,
        center_flair_ratio: float,
        rim_flair_ratio: float,
    ) -> float:
        """
        Weighted soft mismatch score on the CNWM-ratio scale:

            score = 0.20 * s_t2  + 0.60 * s_supp + 0.20 * s_rim

        where each s_* is a linear ramp on the corresponding component.
        The weighting concentrates decision power on central FLAIR
        suppression, which is the only component that meaningfully
        separates mismatch-positive from mismatch-negative tumors on the
        AKU-WHO cohort (the other two saturate near 1.0 for both classes).
        """
        s_t2 = self._ramp(center_t2_ratio, 1.0, self.soft_t2_span)
        s_supp = self._ramp(
            self.ratio_center_flair_low + 0.30 - center_flair_ratio,
            0.0, self.soft_supp_span,
        )
        s_rim = self._ramp(rim_flair_ratio, 1.0, self.soft_rim_span)
        return float(0.20 * s_t2 + 0.60 * s_supp + 0.20 * s_rim)

    def extract_t2_flair_mismatch(
        self,
        tumorseg_ss: str,
        t2_path: str,
        flair_path: str,
        brain_mask_path: str,
        laterality: str | None = None,
        merged_seg: str | None = None,
    ) -> pd.DataFrame:
        """
        Run T2-FLAIR mismatch extraction in subject space.

        All inputs must share the subject-space grid: tumor segmentation,
        T2, FLAIR, brain mask, and (optional) SynthSeg merged segmentation
        used to localise contralateral white matter.
        """
        start_time = time.time()

        segmentation = NiftiImage(tumorseg_ss)
        t2 = NiftiImage(t2_path)
        flair = NiftiImage(flair_path)
        spacing = tuple(float(s) for s in segmentation.spacing)
        voxel_mm3 = float(np.prod(spacing))

        if brain_mask_path is not None:
            brain_mask = NiftiImage(brain_mask_path).array.astype(bool)
        else:
            brain_mask = np.isfinite(t2.array) & (t2.array != 0)

        whole_tumor_mask = self.get_whole_tumor_mask(segmentation)
        tumor_core_mask = self.get_tumor_core_mask(segmentation)
        whole_tumor_volume_ml = float(whole_tumor_mask.sum() * voxel_mm3 / 1000.0)
        tumor_core_volume_ml = float(tumor_core_mask.sum() * voxel_mm3 / 1000.0)

        if self.verbose:
            logger.debug(f"Whole tumor voxels: {int(whole_tumor_mask.sum())}")
            logger.debug(f"Tumor core voxels:  {int(tumor_core_mask.sum())}")

        # TC -> WT fallback when the segmenter has assigned the lesion to ED
        lesion_mask = tumor_core_mask
        if (tumor_core_volume_ml < self.min_core_volume_ml) or (
            whole_tumor_volume_ml > 0
            and tumor_core_volume_ml / whole_tumor_volume_ml < self.tc_wt_ratio_floor
        ):
            if whole_tumor_mask.sum() >= self.min_lesion_voxels:
                lesion_mask = whole_tumor_mask
                if self.verbose:
                    logger.debug("TC too small -> falling back to WT for mismatch scoring")

        result = pd.DataFrame(columns=self.COL_NAMES)
        nan_row = {col: np.nan for col in self.COL_NAMES}
        nan_row["Tumor Core Volume (mL)"] = tumor_core_volume_ml
        nan_row["Whole Tumor Volume (mL)"] = whole_tumor_volume_ml

        if lesion_mask.sum() < self.min_lesion_voxels:
            result.loc[len(result)] = nan_row
            return result

        center_mask, rim_mask = self.get_center_and_rim_masks(lesion_mask, spacing)
        if self.verbose:
            logger.debug(f"Center voxels: {int(center_mask.sum())}, Rim voxels: {int(rim_mask.sum())}")

        if center_mask.sum() < self.min_center_voxels or rim_mask.sum() < self.min_rim_voxels:
            result.loc[len(result)] = nan_row
            return result

        # CNWM reference, with robust fallbacks.
        if merged_seg is not None:
            merged_seg_array = NiftiImage(merged_seg).array.astype(np.int16)
            cnwm_mask = self.get_cnwm_mask(merged_seg_array, laterality)
        else:
            cnwm_mask = None

        if cnwm_mask is not None and cnwm_mask.sum() > 0:
            t2_cnwm_mean = float(np.mean(t2.array[cnwm_mask]))
            flair_cnwm_mean = float(np.mean(flair.array[cnwm_mask]))
        else:
            t2_cnwm_mean = float(np.median(t2.array[brain_mask]))
            flair_cnwm_mean = float(np.median(flair.array[brain_mask]))

        if t2_cnwm_mean <= 0 or flair_cnwm_mean <= 0:
            result.loc[len(result)] = nan_row
            return result

        center_t2_ratio = float(np.mean(t2.array[center_mask])) / (t2_cnwm_mean + 1e-8)
        center_flair_ratio = float(np.mean(flair.array[center_mask])) / (flair_cnwm_mean + 1e-8)
        rim_flair_ratio = float(np.mean(flair.array[rim_mask])) / (flair_cnwm_mean + 1e-8)

        mismatch_score = self.compute_mismatch_score(
            center_t2_ratio=center_t2_ratio,
            center_flair_ratio=center_flair_ratio,
            rim_flair_ratio=rim_flair_ratio,
        )
        mismatch_present = int(mismatch_score >= self.soft_score_thresh)

        # Mismatch degree (raw intensities against CNWM, same definition as before)
        if tumor_core_mask.sum() > 0:
            t2_tumour_mean = float(np.mean(t2.array[tumor_core_mask]))
            flair_tumour_mean = float(np.mean(flair.array[tumor_core_mask]))
            mismatch_degree = (
                t2_tumour_mean / (t2_cnwm_mean + 1e-8)
                - flair_tumour_mean / (flair_cnwm_mean + 1e-8)
            )
        else:
            mismatch_degree = np.nan

        result.loc[len(result)] = {
            "T2-FLAIR Mismatch Present": mismatch_present,
            "T2-FLAIR Mismatch Score": float(mismatch_score),
            "T2-FLAIR Mismatch Degree": float(mismatch_degree) if mismatch_degree == mismatch_degree else np.nan,
            "Central Core T2 Mean": center_t2_ratio,
            "Central Core FLAIR Mean": center_flair_ratio,
            "Peripheral Rim FLAIR Mean": rim_flair_ratio,
            "Tumor Core Volume (mL)": tumor_core_volume_ml,
            "Whole Tumor Volume (mL)": whole_tumor_volume_ml,
        }

        if self.verbose:
            logger.debug(f"T2-FLAIR mismatch result: {result.iloc[0].to_dict()}")
            logger.debug(f"Time taken: {time.time() - start_time:.2f}s")

        return result

    def __call__(
        self,
        tumorseg_ss: str,
        t2_path: str,
        flair_path: str,
        laterality: str,
        merged_seg: str,
        brain_mask_path: str,
    ) -> dict:
        report = self.extract_t2_flair_mismatch(
            tumorseg_ss=tumorseg_ss,
            t2_path=t2_path,
            flair_path=flair_path,
            laterality=laterality,
            merged_seg=merged_seg,
            brain_mask_path=brain_mask_path,
        )
        logger.info("* Finished T2-FLAIR mismatch extraction!")
        return report.iloc[0].to_dict()