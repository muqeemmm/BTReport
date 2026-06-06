"""
ml_classification_btreportpp.py

SVC (linear + RBF) and RandomForest baselines for IDH, 1p/19q and CNS WHO grade
trained on the *extended* BTReport++ metadata (one record per case, sourced
from each subject's `<BraTS_id>_metadata_final.json`).

The "++" variant adds shape / boundary / mismatch / multi-component
descriptors that only appear in `*_metadata_final.json`. Everything else --
data discovery, label derivation, preprocessing strategy, model bank and CV
protocol -- is identical to `ml_classification_btreport.py`.

Extra features added on top of the base BTReport schema
-------------------------------------------------------
Numeric:
    Central Core FLAIR Mean, Central Core T2 Mean, Peripheral Rim FLAIR Mean,
    T2-FLAIR Mismatch Score,
    Tumor Core Volume (mL), Whole Tumor Volume (mL),
    tumor_core_sphericity, whole_tumor_sphericity,
    Transition Zone Thickness (FLAIR|T1CE).{transition_zone_thickness_mm, mean_mm, median_mm},
    boundary_sharpness.{boundary_sharpness, boundary_grad_median, boundary_grad_p90},
    n_components_enhancing.n_components, n_components_nonenhancing.n_components,
    rim_core_adjacency.{adjacency_fraction, core_boundary_voxels, contact_voxels}

Binary:
    T2-FLAIR Mismatch Present, rim_core_adjacency.rim_touches_core

Single-categorical:
    T2-FLAIR Mismatch Degree
"""

from pathlib import Path

from _ml_common import run_ml_pipeline


HERE          = Path(__file__).parent
DATA_ROOT     = Path("/Users/muqeemmmm/GitHub/BTReport_v2/data/Dataset_AKU_WHO")
LABELS_XLSX   = Path("/Users/muqeemmmm/GitHub/BTReport_v2/data/dataset_table/cohort_merged.xlsx")
OUTPUT_DIR    = HERE / "outputs_btreportpp"
JSON_SUFFIX   = "_metadata_final.json"


# --- Extra features available in *_metadata_final.json --------------------- #
EXTRA_NUMERIC = [
    "Central Core FLAIR Mean", "Central Core T2 Mean", "Peripheral Rim FLAIR Mean",
    "T2-FLAIR Mismatch Score",
    "Tumor Core Volume (mL)", "Whole Tumor Volume (mL)",
    "tumor_core_sphericity", "whole_tumor_sphericity",
    "Transition Zone Thickness (FLAIR).transition_zone_thickness_mm",
    "Transition Zone Thickness (FLAIR).mean_mm",
    "Transition Zone Thickness (FLAIR).median_mm",
    "Transition Zone Thickness (T1CE).transition_zone_thickness_mm",
    "Transition Zone Thickness (T1CE).mean_mm",
    "Transition Zone Thickness (T1CE).median_mm",
    "boundary_sharpness.boundary_sharpness",
    "boundary_sharpness.boundary_grad_median",
    "boundary_sharpness.boundary_grad_p90",
    "n_components_enhancing.n_components",
    "n_components_nonenhancing.n_components",
    "rim_core_adjacency.adjacency_fraction",
    "rim_core_adjacency.core_boundary_voxels",
    "rim_core_adjacency.contact_voxels",
]

EXTRA_BINARY = [
    "T2-FLAIR Mismatch Present",
    "rim_core_adjacency.rim_touches_core",
]

EXTRA_SINGLE_CAT = [
    "T2-FLAIR Mismatch Degree",
]


if __name__ == "__main__":
    run_ml_pipeline(
        data_root        = DATA_ROOT,
        labels_xlsx      = LABELS_XLSX,
        output_dir       = OUTPUT_DIR,
        json_suffix      = JSON_SUFFIX,
        extra_numeric    = EXTRA_NUMERIC,
        extra_binary     = EXTRA_BINARY,
        extra_single_cat = EXTRA_SINGLE_CAT,
        cv_splits        = 4,
        cv_repeats       = 10,
    )
