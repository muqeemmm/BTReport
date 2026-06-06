"""
ml_classification_btreport.py

SVC (linear + RBF) and RandomForest baselines for IDH, 1p/19q and CNS WHO grade
trained on the *base* BTReport metadata (one record per case, sourced from each
subject's `<BraTS_id>_metadata_no_clinical.json`).

Data layout consumed:
    /Users/muqeemmmm/GitHub/BTReport_v2/data/Dataset_AKU_WHO/
        Astrocytoma_IDH-mutant/<BraTS_id>/<BraTS_id>_metadata_no_clinical.json
        Astrocytoma_IDH-mutant_Grade_4/<BraTS_id>/...
        Glioblastoma_IDH-wildtype/<BraTS_id>/...
        Oligodendroglioma_IDH-mutant_1p-19q-codeleted/<BraTS_id>/...
        (the "JSONs BTReport" folder is intentionally ignored)

Pipeline:
    1) Discover every <BraTS_id>_metadata_no_clinical.json under the four
       WHO2021 folders.
    2) Flatten the JSONs into a single CSV via json_csv_converter.json_to_csv.
    3) Read all labels (IDH, 1p/19q, Grade) EXCLUSIVELY from
       data/dataset_table/cohort_merged.xlsx, joined to each subject by a
       composite key: BraTS2021 if not 'x', else Center ID (which together
       cover all 646 subjects). Folder names are not used for labels.
    4) Build the BTReport preprocessor (numeric / binary / single-cat /
       multilabel groups), then run RepeatedStratifiedKFold over LinearSVC,
       RBF SVC, and RandomForest -- one task at a time.
    5) Write `per_split_metrics.csv` and `summary_metrics.csv` next to the
       flattened CSV.
"""

from pathlib import Path

from _ml_common import run_ml_pipeline


HERE          = Path(__file__).parent
DATA_ROOT     = Path("/Users/muqeemmmm/GitHub/BTReport_v2/data/Dataset_AKU_WHO")
LABELS_XLSX   = Path("/Users/muqeemmmm/GitHub/BTReport_v2/data/dataset_table/cohort_merged.xlsx")
OUTPUT_DIR    = HERE / "outputs_btreport"
JSON_SUFFIX   = "_metadata_no_clinical.json"


if __name__ == "__main__":
    run_ml_pipeline(
        data_root   = DATA_ROOT,
        labels_xlsx = LABELS_XLSX,
        output_dir  = OUTPUT_DIR,
        json_suffix = JSON_SUFFIX,
        # *_no_clinical.json carries only the base BTReport schema, so no
        # extra columns are added on top of the defaults in _ml_common.
        cv_splits   = 4,
        cv_repeats  = 10,
    )
