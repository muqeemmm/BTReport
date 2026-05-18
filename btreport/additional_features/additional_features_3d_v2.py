"""
T2-FLAIR mismatch extractor — v2 (step-toggleable).

This module reimplements ``ExtractT2FLAIRMismatch`` from
``additional_features_3d.py`` with six independently togglable remediation
steps from the diagnostic review:

    step1  Fall back from BraTS Tumor Core (NCR+ET) to Whole Tumor
           (NCR+ED+ET) ONLY when TC is empty / pathologically small
           (TC < min_core_volume_ml mL or TC/WT < tc_wt_ratio_floor).
    step2  Center/rim geometry: a small DEEP center (physical mm based)
           and a thick peripheral rim, replacing the original
           "dt >= 0.10*max_dt" rule (which assigned ~90% of the core to
           "center").
    step3  Reference intensity normalisation to contralateral normal
           white matter (CNWM) instead of [1,99]-percentile robust
           min-max over the whole brain mask. This puts thresholds on a
           subject-relative, scanner-agnostic ratio scale.
    step4  Threshold values re-tuned for the CNWM-ratio scale used by
           step3 (only meaningful if step3 is also on; the code still
           lets you turn it on alone but will warn).
    step5  Replace the strict AND of four hard thresholds with a soft
           weighted score (smooth sigmoids per component, single decision
           threshold).
    step6  Robust fallbacks: if CNWM mask is empty, fall back to any
           SynthSeg WM label; if still empty, fall back to brain median;
           never silently return NaN for a structural failure.

With ``StepFlags()`` (all False) the class reproduces the original
behaviour and can be used as a drop-in replacement to verify parity
before turning on individual steps.

The module also exposes a Step-7 validation harness via the CLI:

    python -m btreport.additional_features.additional_features_3d_v2 validate \\
        --excel       /path/to/aku_annotations_with_duplicates.xlsx \\
        --dataset     /path/to/Dataset_AKU_WHO \\
        --presets     S0,S1,S12,S123,S1234,S12345,S123456 \\
        --n_negatives 62 \\
        --output      /path/to/validation_results.csv

For each preset the harness prints sensitivity / specificity / agreement
and writes per-subject predictions to disk so improvements are auditable
step by step.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
)

# ---- NIfTI loader shim -------------------------------------------------------
# Prefer the project's loader when available; fall back to a nibabel-only
# stand-in so this script can be invoked outside the btreport package
# (e.g. during quick validation runs).
try:
    from btreport.vasari_features.vasari_auto_v2 import NiftiImage  # type: ignore
except Exception:  # pragma: no cover — falls back outside the package
    import nibabel as nib

    class NiftiImage:  # type: ignore[no-redef]
        """Minimal stand-in matching the subset of NiftiImage used here."""

        def __init__(self, path: str) -> None:
            self._img = nib.load(path)
            self.array = np.asarray(self._img.dataobj)
            zooms = self._img.header.get_zooms()
            self.spacing = tuple(float(z) for z in zooms[:3])


logger = logging.getLogger("t2flair_v2")


# ---- Step flags --------------------------------------------------------------
@dataclass
class StepFlags:
    """One bool per remediation step. All False -> original behaviour."""

    step1_wt_fallback: bool = False
    step2_center_rim_geometry: bool = False
    step3_cnwm_normalization: bool = False
    step4_ratio_thresholds: bool = False
    step5_soft_score: bool = False
    step6_robust_fallbacks: bool = False

    _ORDER = (
        "step1_wt_fallback",
        "step2_center_rim_geometry",
        "step3_cnwm_normalization",
        "step4_ratio_thresholds",
        "step5_soft_score",
        "step6_robust_fallbacks",
    )

    @classmethod
    def from_string(cls, s: str) -> "StepFlags":
        """Parse 'S0', 'S123', 'S1,S2,S3', 'all', 'none'."""
        s = (s or "").strip().lower()
        if s in ("", "none", "0", "s0", "s0_baseline"):
            return cls()
        if s == "all":
            return cls(*([True] * 6))
        s = s.lstrip("s")
        # Accept "1,2,3" or "123"
        tokens = [t for t in s.replace(",", "").strip() if t.isdigit()]
        flags = cls()
        for ch in tokens:
            n = int(ch)
            if not 1 <= n <= 6:
                raise ValueError(f"Unknown step number: {n}")
            setattr(flags, cls._ORDER[n - 1], True)
        return flags

    def label(self) -> str:
        active = [str(i + 1) for i, name in enumerate(self._ORDER) if getattr(self, name)]
        return "S" + "".join(active) if active else "S0_baseline"


# ---- Extractor ---------------------------------------------------------------
@dataclass
class T2FlairMismatchV2:
    """Step-toggleable T2-FLAIR mismatch extractor.

    Outputs the same column set as the original implementation, plus two
    audit fields: ``Lesion Mask Used`` and ``Decision Reason``.
    """

    flags: StepFlags = field(default_factory=StepFlags)
    verbose: bool = False

    # BraTS-style labels (kept identical to the original)
    enhancing_label: int = 4
    nonenhancing_label: int = 1
    oedema_label: int = 2

    # Size guards (originals preserved)
    min_core_voxels: int = 100
    min_wt_voxels: int = 100

    # Step 1 — TC -> WT fallback
    min_core_volume_ml: float = 1.0
    tc_wt_ratio_floor: float = 0.10

    # Step 2 — center/rim geometry (used when ON)
    center_inward_mm: float = 3.0
    center_depth_fraction: float = 0.50
    min_center_voxels: int = 10
    min_rim_voxels: int = 10

    # Original center/rim parameters (used when Step 2 is OFF) — match original
    legacy_min_center_voxels: int = 30
    legacy_min_rim_voxels: int = 30

    # Original thresholds (used when Step 4 is OFF)
    center_t2_high_thresh: float = 0.60
    center_flair_low_thresh: float = 0.50
    rim_flair_high_thresh: float = 0.55
    mismatch_score_thresh: float = 0.12

    # Step 4 — ratio thresholds (used when Step 4 is ON)
    ratio_center_t2_high: float = 1.50
    ratio_center_flair_low: float = 1.55
    ratio_rim_flair_high: float = 1.20
    ratio_score_thresh: float = 0.10

    # Step 5 — soft score threshold (used when Step 5 is ON)
    soft_score_thresh: float = 0.40
    soft_t2_span: float = 0.50      # ratio span over which s_t2 ramps 0->1
    soft_supp_span: float = 0.60    # ratio span for s_supp
    soft_rim_span: float = 0.50     # ratio span for s_rim
    # When Step 3 is OFF and Step 5 is ON, the soft score interprets
    # normalised values [0,1]; spans below are used in that case.
    soft_t2_span_nonratio: float = 0.40
    soft_supp_span_nonratio: float = 0.40
    soft_rim_span_nonratio: float = 0.45

    COL_NAMES = [
        "T2-FLAIR Mismatch Present",
        "T2-FLAIR Mismatch Score",
        "T2-FLAIR Mismatch Degree",
        "Central Core T2 Mean",
        "Central Core FLAIR Mean",
        "Peripheral Rim FLAIR Mean",
        "Tumor Core Volume (mL)",
        "Whole Tumor Volume (mL)",
        "Lesion Mask Used",
        "Decision Reason",
    ]

    # -------------------------------------------------------------------------
    # Intensity normalisation
    # -------------------------------------------------------------------------
    def robust_normalize(self, img: np.ndarray, brain_mask: np.ndarray) -> np.ndarray:
        """Original [1,99]-percentile robust min-max into [0,1]."""
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
        return ((arr - lo) / (hi - lo + 1e-8)).astype(np.float32)

    # -------------------------------------------------------------------------
    # Mask helpers
    # -------------------------------------------------------------------------
    def whole_tumor_mask(self, seg: np.ndarray) -> np.ndarray:
        return np.isin(seg, [self.nonenhancing_label, self.oedema_label, self.enhancing_label])

    def tumor_core_mask(self, seg: np.ndarray) -> np.ndarray:
        return np.isin(seg, [self.nonenhancing_label, self.enhancing_label])

    # -------------------------------------------------------------------------
    # Center/rim splitting
    # -------------------------------------------------------------------------
    def _center_rim_legacy(self, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Original center/rim split (Step 2 OFF)."""
        mask = mask.astype(bool)
        if not mask.any():
            empty = np.zeros_like(mask, dtype=bool)
            return empty, empty

        dt = distance_transform_edt(mask)
        max_dt = float(dt.max())
        if max_dt <= 0:
            empty = np.zeros_like(mask, dtype=bool)
            return empty, empty

        center = mask & (dt >= 0.10 * max_dt)
        rim = mask & (~center)

        # Fallback: identical to original
        if center.sum() < self.legacy_min_center_voxels:
            eroded = binary_erosion(mask, iterations=1)
            if eroded.sum() > 0:
                center = eroded
                rim = mask & (~center)
        return center, rim

    def _center_rim_v2(
        self, mask: np.ndarray, spacing: tuple[float, float, float]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Step 2 geometry: small deep center (mm-based), thick peripheral rim."""
        mask = mask.astype(bool)
        if not mask.any():
            empty = np.zeros_like(mask, dtype=bool)
            return empty, empty

        # Distance transform with anisotropic sampling so the "depth" is in mm.
        dt_mm = distance_transform_edt(mask, sampling=spacing)
        max_mm = float(dt_mm.max())
        if max_mm <= 0:
            empty = np.zeros_like(mask, dtype=bool)
            return empty, empty

        cutoff = max(self.center_inward_mm, self.center_depth_fraction * max_mm)
        center = mask & (dt_mm >= cutoff)

        # Guarantee a small but non-empty center even for tiny lesions: take
        # the deepest 5% of voxels by inward depth if the cutoff is too strict.
        if center.sum() < self.min_center_voxels:
            inside_vals = dt_mm[mask]
            if inside_vals.size:
                q = np.percentile(inside_vals, 95)
                center = mask & (dt_mm >= q)

        # Rim is everything in the lesion that is not in (or adjacent to) center.
        rim = mask & ~binary_dilation(center, iterations=1)
        return center, rim

    # -------------------------------------------------------------------------
    # CNWM mask
    # -------------------------------------------------------------------------
    def _cnwm_mask(
        self,
        merged_seg: np.ndarray,
        laterality: Optional[str],
    ) -> np.ndarray:
        """Contralateral normal WM mask from SynthSeg merged labels.

        Left hemisphere WM labels: 2, 7 ; right hemisphere WM labels: 41, 46.
        """
        wm_left = np.isin(merged_seg, [2, 7])
        wm_right = np.isin(merged_seg, [41, 46])
        if laterality == "left":
            return wm_right
        if laterality == "right":
            return wm_left
        return wm_left | wm_right

    def _wm_fallback_mask(self, merged_seg: np.ndarray) -> np.ndarray:
        """All WM labels — used by Step 6 when contralateral mask is empty."""
        return np.isin(merged_seg, [2, 7, 41, 46])

    # -------------------------------------------------------------------------
    # Soft component scores (Step 5)
    # -------------------------------------------------------------------------
    @staticmethod
    def _soft_ramp(x: float, lo: float, span: float) -> float:
        """Linear ramp from 0 at x=lo to 1 at x=lo+span, clipped to [0,1]."""
        if span <= 0:
            return 1.0 if x >= lo else 0.0
        return float(np.clip((x - lo) / span, 0.0, 1.0))

    def _soft_score(
        self,
        center_t2: float,
        center_flair: float,
        rim_flair: float,
        ratio_mode: bool,
    ) -> float:
        if ratio_mode:
            s_t2 = self._soft_ramp(center_t2, 1.0, self.soft_t2_span)
            s_supp = self._soft_ramp(self.ratio_center_flair_low + 0.30 - center_flair,
                                     0.0, self.soft_supp_span)
            s_rim = self._soft_ramp(rim_flair, 1.0, self.soft_rim_span)
        else:
            s_t2 = self._soft_ramp(center_t2, self.center_t2_high_thresh - 0.10,
                                   self.soft_t2_span_nonratio)
            s_supp = self._soft_ramp(self.center_flair_low_thresh + 0.10 - center_flair,
                                     0.0, self.soft_supp_span_nonratio)
            s_rim = self._soft_ramp(rim_flair, self.rim_flair_high_thresh - 0.10,
                                    self.soft_rim_span_nonratio)
        return float((s_t2 + s_supp + s_rim) / 3.0)

    # -------------------------------------------------------------------------
    # Main extraction
    # -------------------------------------------------------------------------
    def __call__(
        self,
        tumorseg_ss: str,
        t2_path: str,
        flair_path: str,
        laterality: Optional[str],
        merged_seg: Optional[str],
        brain_mask_path: Optional[str],
    ) -> dict:
        t0 = time.time()
        flags = self.flags

        seg_img = NiftiImage(tumorseg_ss)
        seg = seg_img.array.astype(np.int16)
        spacing = tuple(float(s) for s in seg_img.spacing)
        voxel_mm3 = float(np.prod(spacing))

        t2 = NiftiImage(t2_path).array.astype(np.float32)
        flair = NiftiImage(flair_path).array.astype(np.float32)

        if brain_mask_path and os.path.exists(brain_mask_path):
            brain_mask = NiftiImage(brain_mask_path).array.astype(bool)
        else:
            brain_mask = np.isfinite(t2) & (t2 != 0)

        wt = self.whole_tumor_mask(seg)
        tc = self.tumor_core_mask(seg)
        wt_ml = float(wt.sum() * voxel_mm3 / 1000.0)
        tc_ml = float(tc.sum() * voxel_mm3 / 1000.0)

        # --- Step 1: choose lesion mask ---------------------------------------
        lesion = tc
        lesion_label = "TC"
        if flags.step1_wt_fallback:
            tc_wt_ratio = (tc_ml / wt_ml) if wt_ml > 0 else 0.0
            if (tc_ml < self.min_core_volume_ml) or (tc_wt_ratio < self.tc_wt_ratio_floor):
                if wt.sum() >= self.min_wt_voxels:
                    lesion = wt
                    lesion_label = "WT_fallback"
                    if self.verbose:
                        logger.info(
                            "step1: TC=%.2f mL, TC/WT=%.3f -> falling back to WT (%.2f mL)",
                            tc_ml, tc_wt_ratio, wt_ml,
                        )

        # Size guard --- same semantics as the original, but parameterised by
        # the chosen lesion (TC -> min_core_voxels; WT fallback -> min_wt_voxels).
        if (lesion_label == "TC" and lesion.sum() < self.min_core_voxels) or \
           (lesion_label == "WT_fallback" and lesion.sum() < self.min_wt_voxels):
            return self._nan_result(
                tc_ml, wt_ml, lesion_label,
                reason=f"lesion too small ({int(lesion.sum())} vox)",
            )

        # --- Step 2: center/rim split -----------------------------------------
        if flags.step2_center_rim_geometry:
            center_mask, rim_mask = self._center_rim_v2(lesion, spacing)
            min_center = self.min_center_voxels
            min_rim = self.min_rim_voxels
        else:
            center_mask, rim_mask = self._center_rim_legacy(lesion)
            min_center = self.legacy_min_center_voxels
            min_rim = self.legacy_min_rim_voxels

        if self.verbose:
            logger.debug("center=%d rim=%d (geometry=%s)",
                         int(center_mask.sum()), int(rim_mask.sum()),
                         "v2" if flags.step2_center_rim_geometry else "legacy")

        if center_mask.sum() < min_center or rim_mask.sum() < min_rim:
            if flags.step6_robust_fallbacks and lesion_label != "WT_fallback" and wt.sum() >= self.min_wt_voxels:
                # Step 6: retry with WT lesion mask
                lesion = wt
                lesion_label = "WT_step6_retry"
                if flags.step2_center_rim_geometry:
                    center_mask, rim_mask = self._center_rim_v2(lesion, spacing)
                else:
                    center_mask, rim_mask = self._center_rim_legacy(lesion)
            if center_mask.sum() < min_center or rim_mask.sum() < min_rim:
                return self._nan_result(
                    tc_ml, wt_ml, lesion_label,
                    reason=f"center/rim too small (c={int(center_mask.sum())}, r={int(rim_mask.sum())})",
                )

        # --- Step 3: normalisation reference ----------------------------------
        ratio_mode = flags.step3_cnwm_normalization
        cnwm_t2_mean = cnwm_flair_mean = None
        cnwm_source = None

        if ratio_mode:
            if merged_seg is None or not os.path.exists(merged_seg):
                if flags.step6_robust_fallbacks:
                    ratio_mode = False  # silently drop back to robust min-max
                    cnwm_source = "missing_merged_seg->robust_minmax"
                else:
                    return self._nan_result(
                        tc_ml, wt_ml, lesion_label,
                        reason="step3 requires merged_seg",
                    )
            else:
                merged = NiftiImage(merged_seg).array.astype(np.int16)
                cnwm = self._cnwm_mask(merged, laterality)
                if cnwm.sum() == 0:
                    if flags.step6_robust_fallbacks:
                        cnwm = self._wm_fallback_mask(merged)
                        cnwm_source = "wm_both_hemispheres"
                    else:
                        return self._nan_result(
                            tc_ml, wt_ml, lesion_label,
                            reason="empty CNWM mask",
                        )
                else:
                    cnwm_source = "contralateral_wm" if laterality in ("left", "right") else "bilateral_wm"

                if cnwm.sum() == 0 and flags.step6_robust_fallbacks:
                    # Final fallback: brain median (Step 6)
                    cnwm = brain_mask
                    cnwm_source = "brain_median_fallback"

                if cnwm.sum() == 0:
                    return self._nan_result(tc_ml, wt_ml, lesion_label,
                                            reason="no WM fallback available")

                cnwm_t2_mean = float(np.mean(t2[cnwm]))
                cnwm_flair_mean = float(np.mean(flair[cnwm]))
                if cnwm_t2_mean <= 0 or cnwm_flair_mean <= 0:
                    if flags.step6_robust_fallbacks:
                        ratio_mode = False
                        cnwm_source = (cnwm_source or "") + "->degenerate->robust_minmax"
                    else:
                        return self._nan_result(
                            tc_ml, wt_ml, lesion_label,
                            reason="degenerate CNWM mean (<=0)",
                        )

        if ratio_mode:
            center_t2_val = float(np.mean(t2[center_mask])) / (cnwm_t2_mean + 1e-8)
            center_flair_val = float(np.mean(flair[center_mask])) / (cnwm_flair_mean + 1e-8)
            rim_flair_val = float(np.mean(flair[rim_mask])) / (cnwm_flair_mean + 1e-8)
        else:
            t2_norm = self.robust_normalize(t2, brain_mask)
            flair_norm = self.robust_normalize(flair, brain_mask)
            center_t2_val = float(np.mean(t2_norm[center_mask]))
            center_flair_val = float(np.mean(flair_norm[center_mask]))
            rim_flair_val = float(np.mean(flair_norm[rim_mask]))

        # --- Mismatch degree (kept from original, in raw intensities) ----------
        if merged_seg and os.path.exists(merged_seg):
            merged = NiftiImage(merged_seg).array.astype(np.int16)
            cnwm_full = self._cnwm_mask(merged, laterality)
            if cnwm_full.sum() == 0:
                cnwm_full = self._wm_fallback_mask(merged)
            if cnwm_full.sum() > 0 and tc.sum() > 0:
                t2_tum = float(np.mean(t2[tc]))
                fl_tum = float(np.mean(flair[tc]))
                t2_wm = float(np.mean(t2[cnwm_full]))
                fl_wm = float(np.mean(flair[cnwm_full]))
                mismatch_degree = (t2_tum / (t2_wm + 1e-8)) - (fl_tum / (fl_wm + 1e-8))
            else:
                mismatch_degree = float("nan")
        else:
            mismatch_degree = float("nan")

        # --- Step 4 / Step 5: decision ----------------------------------------
        if flags.step4_ratio_thresholds and not ratio_mode:
            logger.warning("Step 4 (ratio thresholds) is on but Step 3 (CNWM normalisation) is off "
                           "— ratio thresholds will be applied to robust min-max [0,1] values "
                           "and will not behave as documented.")

        # Per-component score (used both as legacy mismatch_score and in soft branch)
        if flags.step4_ratio_thresholds:
            c_t2_excess = max(0.0, center_t2_val - self.ratio_center_t2_high)
            c_supp = max(0.0, self.ratio_center_flair_low - center_flair_val)
            r_excess = max(0.0, rim_flair_val - self.ratio_rim_flair_high)
            score_thresh = self.ratio_score_thresh
            t2_hi, fl_lo, rim_hi = (self.ratio_center_t2_high,
                                    self.ratio_center_flair_low,
                                    self.ratio_rim_flair_high)
        else:
            c_t2_excess = max(0.0, center_t2_val - self.center_t2_high_thresh)
            c_supp = max(0.0, self.center_flair_low_thresh - center_flair_val)
            r_excess = max(0.0, rim_flair_val - self.rim_flair_high_thresh)
            score_thresh = self.mismatch_score_thresh
            t2_hi, fl_lo, rim_hi = (self.center_t2_high_thresh,
                                    self.center_flair_low_thresh,
                                    self.rim_flair_high_thresh)

        mismatch_score = (c_t2_excess + c_supp + r_excess) / 3.0

        if flags.step5_soft_score:
            soft = self._soft_score(center_t2_val, center_flair_val, rim_flair_val,
                                    ratio_mode=ratio_mode)
            mismatch_score = soft  # surface the soft value in the JSON
            mismatch_present = int(soft >= self.soft_score_thresh)
            reason = f"soft={soft:.3f} thr={self.soft_score_thresh}"
        else:
            mismatch_present = int(
                (center_t2_val >= t2_hi)
                and (center_flair_val <= fl_lo)
                and (rim_flair_val >= rim_hi)
                and (mismatch_score >= score_thresh)
            )
            reason = (
                f"cT2={center_t2_val:.3f}(>={t2_hi}) "
                f"cFL={center_flair_val:.3f}(<={fl_lo}) "
                f"rFL={rim_flair_val:.3f}(>={rim_hi}) "
                f"score={mismatch_score:.3f}(>={score_thresh})"
            )

        if self.verbose:
            logger.info("[%s] %s -> present=%d", self.flags.label(), reason, mismatch_present)
            logger.info("elapsed %.2fs", time.time() - t0)

        return {
            "T2-FLAIR Mismatch Present": mismatch_present,
            "T2-FLAIR Mismatch Score": float(mismatch_score),
            "T2-FLAIR Mismatch Degree": float(mismatch_degree) if mismatch_degree == mismatch_degree else float("nan"),
            "Central Core T2 Mean": center_t2_val,
            "Central Core FLAIR Mean": center_flair_val,
            "Peripheral Rim FLAIR Mean": rim_flair_val,
            "Tumor Core Volume (mL)": tc_ml,
            "Whole Tumor Volume (mL)": wt_ml,
            "Lesion Mask Used": lesion_label,
            "Decision Reason": reason + (f" [cnwm={cnwm_source}]" if cnwm_source else ""),
        }

    def _nan_result(self, tc_ml: float, wt_ml: float, lesion_label: str, reason: str) -> dict:
        return {
            "T2-FLAIR Mismatch Present": float("nan"),
            "T2-FLAIR Mismatch Score": float("nan"),
            "T2-FLAIR Mismatch Degree": float("nan"),
            "Central Core T2 Mean": float("nan"),
            "Central Core FLAIR Mean": float("nan"),
            "Peripheral Rim FLAIR Mean": float("nan"),
            "Tumor Core Volume (mL)": tc_ml,
            "Whole Tumor Volume (mL)": wt_ml,
            "Lesion Mask Used": lesion_label,
            "Decision Reason": f"NaN: {reason}",
        }


# =============================================================================
# Step 7 — validation harness
# =============================================================================
DEFAULT_SUBTYPES = (
    "Astrocytoma_IDH-mutant",
    "Astrocytoma_IDH-mutant_Grade_4",
    "Glioblastoma_IDH-wildtype",
    "Oligodendroglioma_IDH-mutant_1p-19q-codeleted",
)


def _normalise_subject_id(s: str) -> str:
    return str(s).strip()


def load_ground_truth(excel_path: str) -> pd.DataFrame:
    """Return a DataFrame with columns: Subject ID, Center ID, mismatch (0/1)."""
    df = pd.read_excel(excel_path)
    df = df.copy()
    df["FLAIR Findings"] = df["FLAIR Findings"].fillna("")
    df["mismatch"] = df["FLAIR Findings"].str.contains("mismatch", case=False, na=False).astype(int)
    # Reduce duplicates: a subject is positive if ANY annotation says present.
    grp = df.groupby("Center ID", dropna=True).agg(
        Subject_ID=("Subject ID", "first"),
        mismatch=("mismatch", "max"),
    ).reset_index()
    return grp.rename(columns={"Center_ID": "Center ID"})


def index_dataset(dataset_root: str, subtypes=DEFAULT_SUBTYPES) -> dict[str, str]:
    """Map Center ID -> subtype subfolder (skipping macOS resource forks)."""
    idx: dict[str, str] = {}
    for sub in subtypes:
        sub_path = os.path.join(dataset_root, sub)
        if not os.path.isdir(sub_path):
            continue
        for f in os.listdir(sub_path):
            if f.startswith("._"):
                continue
            full = os.path.join(sub_path, f)
            if os.path.isdir(full):
                idx[f] = sub
    return idx


def discover_subject_files(subject_folder: str) -> dict:
    """Return a dict of NIfTI paths needed by the extractor (best-effort)."""
    def first(pat: str) -> Optional[str]:
        hits = sorted(p for p in glob.glob(os.path.join(subject_folder, pat))
                      if not os.path.basename(p).startswith("._"))
        return hits[0] if hits else None

    t2 = first("*_t2.nii.gz") or first("*-t2*.nii.gz")
    flair = first("*_flair.nii.gz") or first("*-flair*.nii.gz") or first("*-fla*.nii.gz")
    seg = (first("*_seg_pred.nii.gz") or first("*_seg.nii.gz")
           or first("*-seg.nii.gz") or first("*-seg_pred.nii.gz"))

    brain_mask = os.path.join(subject_folder, "tmp", "brain_mask.nii.gz")
    if not os.path.exists(brain_mask):
        brain_mask = None
    merged_seg = os.path.join(subject_folder, "tmp", "MNI152_in_subject_space_merged_seg.nii.gz")
    if not os.path.exists(merged_seg):
        merged_seg = None

    return {"t2": t2, "flair": flair, "seg": seg,
            "brain_mask": brain_mask, "merged_seg": merged_seg}


def infer_laterality(seg_path: str) -> Optional[str]:
    """Crude laterality heuristic from tumor centroid x-coordinate."""
    try:
        arr = NiftiImage(seg_path).array
        mask = arr > 0
        if not mask.any():
            return None
        cx = float(np.mean(np.where(mask)[0]))
        nx = arr.shape[0]
        return "left" if cx < nx / 2 else "right"
    except Exception:
        return None


def _run_one(extractor: T2FlairMismatchV2, files: dict, laterality: Optional[str]) -> dict:
    return extractor(
        tumorseg_ss=files["seg"],
        t2_path=files["t2"],
        flair_path=files["flair"],
        laterality=laterality,
        merged_seg=files["merged_seg"],
        brain_mask_path=files["brain_mask"],
    )


def validate(
    excel_path: str,
    dataset_root: str,
    presets: list[str],
    n_negatives: int = 62,
    seed: int = 0,
    output_csv: Optional[str] = None,
    laterality_overrides: Optional[dict[str, str]] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Step 7 validation harness."""
    gt = load_ground_truth(excel_path)
    folder_map = index_dataset(dataset_root)

    # Keep only subjects whose folders exist in the dataset
    gt = gt[gt["Center ID"].isin(folder_map)].reset_index(drop=True)
    pos = gt[gt["mismatch"] == 1]["Center ID"].tolist()
    neg_pool = gt[gt["mismatch"] == 0]["Center ID"].tolist()
    rng = random.Random(seed)
    neg = rng.sample(neg_pool, min(n_negatives, len(neg_pool)))
    subjects = [(c, 1) for c in pos] + [(c, 0) for c in neg]

    if verbose:
        logger.info("Step-7 cohort: %d positives + %d negatives", len(pos), len(neg))

    # Pre-discover file paths once
    file_index: dict[str, tuple[dict, Optional[str]]] = {}
    missing = []
    for cid, _ in subjects:
        folder = os.path.join(dataset_root, folder_map[cid], cid)
        files = discover_subject_files(folder)
        if not (files["t2"] and files["flair"] and files["seg"]):
            missing.append((cid, files))
            continue
        lat = None
        if laterality_overrides:
            lat = laterality_overrides.get(cid)
        if lat is None:
            lat = infer_laterality(files["seg"])
        file_index[cid] = (files, lat)

    if missing and verbose:
        logger.warning("Skipping %d subject(s) with missing NIfTI inputs", len(missing))
        for cid, f in missing[:5]:
            logger.warning("  %s: t2=%s flair=%s seg=%s", cid, f["t2"], f["flair"], f["seg"])

    # Run each preset
    summary_rows = []
    per_subject_rows = []
    for preset in presets:
        flags = StepFlags.from_string(preset)
        extractor = T2FlairMismatchV2(flags=flags, verbose=False)
        label = flags.label()
        logger.info("=== Running preset %s (%s) ===", preset, label)
        tp = tn = fp = fn = ns_pos = ns_neg = 0
        # Track central-FLAIR values per truth class for the per-preset median.
        # We accumulate every numeric Central Core FLAIR Mean returned by the
        # extractor (including for FN / FP cases) so the median reflects what
        # the rule was actually looking at — not just the cases it scored "1".
        cfl_pos: list[float] = []
        cfl_neg: list[float] = []
        for cid, truth in subjects:
            if cid not in file_index:
                if truth == 1: ns_pos += 1
                else: ns_neg += 1
                per_subject_rows.append({
                    "preset": label, "Center ID": cid, "truth": truth,
                    "prediction": float("nan"),
                    "Decision Reason": "MISSING_FILES",
                })
                continue
            files, lat = file_index[cid]
            try:
                out = _run_one(extractor, files, lat)
            except Exception as e:
                logger.warning("%s [%s] failed: %s", cid, label, e)
                if truth == 1: ns_pos += 1
                else: ns_neg += 1
                per_subject_rows.append({
                    "preset": label, "Center ID": cid, "truth": truth,
                    "prediction": float("nan"),
                    "Decision Reason": f"EXC: {e}",
                })
                continue
            pred = out["T2-FLAIR Mismatch Present"]
            is_nan = isinstance(pred, float) and math.isnan(pred)
            if is_nan:
                if truth == 1: ns_pos += 1
                else: ns_neg += 1
            elif truth == 1 and int(pred) == 1: tp += 1
            elif truth == 1 and int(pred) == 0: fn += 1
            elif truth == 0 and int(pred) == 1: fp += 1
            else: tn += 1

            # Accumulate central FLAIR values (regardless of decision) so the
            # per-preset median reports what the extractor measured, not only
            # what it labelled positive.
            cfl = out.get("Central Core FLAIR Mean")
            if isinstance(cfl, (int, float)) and not math.isnan(float(cfl)):
                (cfl_pos if truth == 1 else cfl_neg).append(float(cfl))

            row = {"preset": label, "Center ID": cid, "truth": truth,
                   "prediction": float("nan") if is_nan else int(pred)}
            row.update({k: v for k, v in out.items() if k != "T2-FLAIR Mismatch Present"})
            per_subject_rows.append(row)

        n_pos = tp + fn + ns_pos
        n_neg = tn + fp + ns_neg
        sens_strict = tp / n_pos if n_pos else float("nan")           # NaN counts as miss
        sens_excl   = tp / (tp + fn) if (tp + fn) else float("nan")   # NaN excluded
        spec_strict = tn / n_neg if n_neg else float("nan")
        spec_excl   = tn / (tn + fp) if (tn + fp) else float("nan")
        med_cfl_pos = float(np.median(cfl_pos)) if cfl_pos else float("nan")
        med_cfl_neg = float(np.median(cfl_neg)) if cfl_neg else float("nan")
        summary_rows.append({
            "preset": label,
            "TP": tp, "FN": fn, "NS_pos": ns_pos,
            "TN": tn, "FP": fp, "NS_neg": ns_neg,
            "sensitivity_strict": sens_strict,
            "sensitivity_excl_NS": sens_excl,
            "specificity_strict": spec_strict,
            "specificity_excl_NS": spec_excl,
            "agreement_pos": tp,
            "disagreement_pos": fn,
            "not_specified_pos": ns_pos,
            # Per-preset median of Central Core FLAIR Mean (over scored
            # subjects only — NaNs are excluded from the median). Watching
            # this column move toward / below the central-FLAIR threshold
            # is the most direct signal that the binding constraint is
            # being addressed.
            "median_cFL_pos": med_cfl_pos,
            "median_cFL_neg": med_cfl_neg,
            "n_cFL_pos": len(cfl_pos),
            "n_cFL_neg": len(cfl_neg),
        })

    summary = pd.DataFrame(summary_rows)
    per_subject = pd.DataFrame(per_subject_rows)

    print()
    print("=" * 78)
    print("Step-7 Validation Summary")
    print("=" * 78)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()

    if output_csv:
        summary_path = output_csv
        details_path = output_csv.replace(".csv", "_per_subject.csv")
        summary.to_csv(summary_path, index=False)
        per_subject.to_csv(details_path, index=False)
        logger.info("Wrote %s and %s", summary_path, details_path)

    return summary


# =============================================================================
# CLI
# =============================================================================
def _cli_single(args) -> None:
    flags = StepFlags.from_string(args.steps)
    extractor = T2FlairMismatchV2(flags=flags, verbose=args.verbose)
    files = discover_subject_files(args.subject_folder)
    lat = args.laterality or infer_laterality(files["seg"])
    out = _run_one(extractor, files, lat)
    print(json.dumps(out, indent=2, default=str))


def _cli_validate(args) -> None:
    presets = [p.strip() for p in args.presets.split(",") if p.strip()]
    validate(
        excel_path=args.excel,
        dataset_root=args.dataset,
        presets=presets,
        n_negatives=args.n_negatives,
        seed=args.seed,
        output_csv=args.output,
        verbose=args.verbose,
    )


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="T2-FLAIR mismatch v2 with step-toggle and validation harness."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_single = sub.add_parser("single", help="Run extractor on a single subject folder.")
    p_single.add_argument("--subject_folder", required=True)
    p_single.add_argument("--steps", default="none",
                          help="Step toggle string, e.g. 'none', 'S12', '1,2,3', 'all'.")
    p_single.add_argument("--laterality", default=None, choices=[None, "left", "right"])
    p_single.add_argument("--verbose", action="store_true")
    p_single.set_defaults(func=_cli_single)

    p_val = sub.add_parser("validate", help="Step-7 validation harness.")
    p_val.add_argument("--excel", required=True,
                       help="Path to aku_annotations_with_duplicates.xlsx")
    p_val.add_argument("--dataset", required=True,
                       help="Path to Dataset_AKU_WHO root")
    p_val.add_argument("--presets",
                       default="S0,S1,S12,S123,S1234,S12345,S123456",
                       help="Comma-separated preset list, each like 'S0', 'S12', or 'all'.")
    p_val.add_argument("--n_negatives", type=int, default=62)
    p_val.add_argument("--seed", type=int, default=0)
    p_val.add_argument("--output", default=None,
                       help="CSV path for summary; per-subject CSV is written alongside.")
    p_val.add_argument("--verbose", action="store_true")
    p_val.set_defaults(func=_cli_validate)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    args.func(args)


if __name__ == "__main__":
    main()
