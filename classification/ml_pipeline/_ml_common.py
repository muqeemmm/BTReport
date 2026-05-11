# --------------------------------------------------------------------------- #
# _ml_common.py
#
# Shared helpers for the BTReport / BTReport++ ML baselines.
#
# Each per-suffix entry point (`ml_classification_btreport.py`,
# `ml_classification_btreportpp.py`) only needs to declare:
#   - the JSON filename suffix it consumes (e.g. "_metadata_no_clinical.json")
#   - any *extra* numeric / binary / single-categorical columns produced by
#     that flavour of JSON (after flattening through json_csv_converter)
# and then call `run_ml_pipeline(...)`.
#
# Pipeline:
#   1. Walk Dataset_AKU_WHO/*/<subject>/<subject>{suffix}, dump each subject's
#      JSON path; flatten all of them into a single CSV via json_to_csv.
#   2. Derive (IDH, 1p/19q, Grade) labels by combining the WHO2021 folder name
#      (which uniquely fixes IDH and 1p/19q, and pins Grade=4 for Glioblastoma /
#      "Grade_4" folders) with master_file.xlsx for Astrocytoma / Oligo grade.
#   3. Build a ColumnTransformer (numeric -> impute+scale, categorical ->
#      impute+one-hot, multilabel -> MultiLabelBinarizer).
#   4. Train LinearSVC, RBF SVC, and RandomForest under RepeatedStratifiedKFold
#      and report per-task summary metrics.
# --------------------------------------------------------------------------- #
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing  import Iterable, List, Optional, Sequence

import numpy  as np
import pandas as pd

from sklearn.base            import BaseEstimator, TransformerMixin
from sklearn.compose         import ColumnTransformer
from sklearn.ensemble        import RandomForestClassifier
from sklearn.impute          import SimpleImputer
from sklearn.metrics         import (accuracy_score, balanced_accuracy_score,
                                     f1_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import (FunctionTransformer, MultiLabelBinarizer,
                                     OneHotEncoder, StandardScaler)
from sklearn.svm             import SVC

from json_csv_converter import json_to_csv

ID_COLUMN    = "source_file"
TARGET_COLS  = ["IDH", "1p_19q", "Grade"]


# --------------------------------------------------------------------------- #
# Base feature schema (columns that appear in *_no_clinical.json).
# The "pp" script adds to these lists in its CONFIG.
# --------------------------------------------------------------------------- #
BASE_NUMERIC_COLS = [
    "age_years",
    "n_slices_with_shift", "mean_shift_mm", "median_shift_mm",
    "max_shift_mm", "p95_shift_mm",
    "volumes_ideal_midline.ncr.left",    "volumes_ideal_midline.ncr.right",
    "volumes_ideal_midline.ed.left",     "volumes_ideal_midline.ed.right",
    "volumes_ideal_midline.et.left",     "volumes_ideal_midline.et.right",
    "volumes_ideal_midline.ncr_et.left", "volumes_ideal_midline.ncr_et.right",
    "volumes_patient_midline.ncr.left",    "volumes_patient_midline.ncr.right",
    "volumes_patient_midline.ed.left",     "volumes_patient_midline.ed.right",
    "volumes_patient_midline.et.left",     "volumes_patient_midline.et.right",
    "volumes_patient_midline.ncr_et.left", "volumes_patient_midline.ncr_et.right",
    "ED Volume (mL)", "ET Volume (mL)", "NCR Volume (mL)",
    "Left Ventricle Volume (mm^3)", "Right Ventricle Volume (mm^3)",
    "Number of lesions", "Total tumor volume (mL)",
    "Proportion Enhancing", "Proportion Necrosis", "Proportion of Oedema",
]

BASE_BINARY_COLS = [
    "gender",
    "crosses_ideal_midline.ncr",   "crosses_ideal_midline.ed",
    "crosses_ideal_midline.et",    "crosses_ideal_midline.ncr_et",
    "crosses_patient_midline.ncr", "crosses_patient_midline.ed",
    "crosses_patient_midline.et",  "crosses_patient_midline.ncr_et",
    "Asymmetrical Ventricles", "CET Crosses midline",
    "Cortical involvement",    "Deep WM invasion",
    "Edema crosses midline",   "Ependymal (ventricular) Invasion",
    "Multiple satellites present", "Multifocal or Multicentric",
]

BASE_SINGLE_CAT_COLS = [
    "primary_side_ideal_midline.ncr",   "primary_side_ideal_midline.ed",
    "primary_side_ideal_midline.et",    "primary_side_ideal_midline.ncr_et",
    "primary_side_patient_midline.ncr", "primary_side_patient_midline.ed",
    "primary_side_patient_midline.et",  "primary_side_patient_midline.ncr_et",
    "midline_shift_present", "level_max_shift",
    "Eloquent Brain Involvement", "Enhancement Quality",
    "Effaced Ventricle", "Enlarged Ventricles",
    "Side of Tumor Epicenter", "Thickness of enhancing margin",
]

BASE_MULTI_LABEL_COLS = ["Anatomical Overlap Regions"]

# Columns flattened from JSON but not modelled.
BASE_DROPPED_COLS = [
    "Text Report", "Lesion Sizes APxTVxCC (cm)",
    "Region Proportions", "Tumor Location",
    # Final-only nested fields that flatten to non-numeric lists / metadata:
    "spacing_mm",
    "n_components_enhancing.component_sizes_mm3",
    "n_components_nonenhancing.component_sizes_mm3",
]


# --------------------------------------------------------------------------- #
# Multilabel transformer (JSON-aware: "Anatomical Overlap Regions" comes
# through json_to_csv as a JSON string like '["left putamen", ...]').
# --------------------------------------------------------------------------- #
def _split_multilabel(series: pd.Series):
    out = []
    for x in series.fillna("").astype(str):
        x = x.strip()
        if x == "" or x.lower() in ("nan", "null", "none") or x == "[]":
            out.append([])
            continue
        parsed = None
        if x[0] in "[\"":
            try:
                parsed = json.loads(x)
            except (json.JSONDecodeError, ValueError):
                parsed = None
        if isinstance(parsed, list):
            out.append([str(t).strip() for t in parsed if str(t).strip() != ""])
        else:
            out.append([t.strip() for t in x.split(",") if t.strip() != ""])
    return out


class MLBTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mlb = MultiLabelBinarizer(sparse_output=True)

    def fit(self, X, y=None):
        self.mlb.fit(_split_multilabel(pd.Series(np.asarray(X).ravel())))
        return self

    def transform(self, X):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                                    message=r"unknown class\(es\) .* will be ignored",
                                    category=UserWarning)
            return self.mlb.transform(_split_multilabel(pd.Series(np.asarray(X).ravel())))


# --------------------------------------------------------------------------- #
# JSON discovery + label derivation
# --------------------------------------------------------------------------- #
def discover_subject_jsons(data_root: Path, suffix: str) -> List[tuple]:
    """
    Walk Dataset_AKU_WHO/<WHO_FOLDER>/<subject>/<subject><suffix>, returning
    (subject_id, who_folder, path) for each subject that has the requested
    JSON file. The "JSONs BTReport" folder is excluded by design.
    """
    rows = []
    for who_folder in sorted(p for p in data_root.iterdir() if p.is_dir()):
        if who_folder.name == "JSONs BTReport":
            continue
        for subj in sorted(p for p in who_folder.iterdir() if p.is_dir()):
            cand = subj / f"{subj.name}{suffix}"
            if cand.exists():
                rows.append((subj.name, who_folder.name, cand))
    return rows


def build_labels(subjects: pd.DataFrame, labels_xlsx: Path) -> pd.DataFrame:
    """
    Attach (IDH, 1p_19q, Grade) labels read EXCLUSIVELY from cohort_merged.xlsx
    (no folder-name inference). Folder names are used only for traceability.

    cohort_merged.xlsx layout (relevant columns):
        BraTS2021, Center ID, IDH ('mutated' | 'wild type'),
        1p/19q ('intact' | 'co-deleted' | 'rel. co-deleted' | 'x'),
        Grade  (2 | 3 | 4)

    Join key
    --------
    Subject folders are named either:
      - 'BraTS2021_XXXXX' (302 / 646 cases — match cohort_merged.BraTS2021)
      - '<Center ID>'     (344 / 646 cases — match cohort_merged.'Center ID')
    so we build a composite key: BraTS2021 if not 'x', else Center ID.
    With this key every one of the 646 folder subjects is linkable.

    Label normalisation (keeps the rest of the project's vocabulary):
      IDH    : 'mutated'        -> 'mutant'
               'wild type'      -> 'wildtype'
      1p/19q : 'intact'         -> 'non-codeleted'
               'co-deleted'     -> 'codeleted'
               'rel. co-deleted'-> NaN (ambiguous, excluded from 1p/19q task)
               'x'              -> NaN (missing)
      Grade  : numeric int (NaN if 'x' / not parseable)
    """
    if not labels_xlsx.exists():
        raise FileNotFoundError(
            f"Required labels file not found: {labels_xlsx}")

    labels = pd.read_excel(labels_xlsx)
    needed = {"BraTS2021", "Center ID", "IDH", "1p/19q", "Grade"}
    missing = needed - set(labels.columns)
    if missing:
        raise ValueError(
            f"cohort_merged.xlsx is missing required column(s): {missing}")

    # Composite key: prefer BraTS2021, fall back to Center ID where BraTS=='x'.
    labels = labels.copy()
    labels["_key"] = labels["BraTS2021"].astype(str).where(
        labels["BraTS2021"].astype(str) != "x",
        labels["Center ID"].astype(str),
    )

    idh_map    = {"mutated": "mutant", "wild type": "wildtype"}
    onep_map   = {"intact": "non-codeleted", "co-deleted": "codeleted"}

    labels["IDH"]    = labels["IDH"].map(idh_map)
    labels["1p_19q"] = labels["1p/19q"].map(onep_map)   # drops 'x' / 'rel. co-deleted'
    labels["Grade"]  = pd.to_numeric(labels["Grade"], errors="coerce")

    keep = ["_key", "IDH", "1p_19q", "Grade"]
    return subjects.merge(labels[keep], left_on="subject_id",
                          right_on="_key", how="left").drop(columns="_key")


# --------------------------------------------------------------------------- #
# Preprocessor
# --------------------------------------------------------------------------- #
def build_preprocessor(df_cols, numeric_cols, binary_cols, single_cat_cols,
                       multi_label_cols):
    cols_set = set(df_cols)
    keep = lambda lst: [c for c in lst if c in cols_set]

    numeric_cols     = keep(numeric_cols)
    binary_cols      = keep(binary_cols)
    single_cat_cols  = keep(single_cat_cols)
    multi_label_cols = keep(multi_label_cols)

    def _to_numeric_block(X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return pd.DataFrame(X).apply(pd.to_numeric, errors="coerce").to_numpy()

    numeric_pipe = Pipeline([
        ("to_numeric", FunctionTransformer(_to_numeric_block, feature_names_out="one-to-one")),
        ("imputer",    SimpleImputer(strategy="median")),
        ("scaler",     StandardScaler()),
    ])
    def _to_str_block(X):
        # OneHotEncoder requires uniform dtype per column; some columns mix
        # nan (float) with strings after JSON->CSV flattening, so cast all
        # cells to string first and let the imputer replace "nan"/"" later.
        X = np.asarray(X, dtype=object)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        # Keep dtype=object (not <U..>) so SimpleImputer accepts it.
        out = np.empty(X.shape, dtype=object)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                v = X[i, j]
                out[i, j] = "__MISSING__" if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v)
        return out

    cat_pipe = Pipeline([
        ("to_str",  FunctionTransformer(_to_str_block, feature_names_out="one-to-one")),
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore")),
        ("scaler",  StandardScaler(with_mean=False)),
    ])
    mlb_pipes = [
        (col, Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            ("mlb",     MLBTransformer()),
            ("scaler",  StandardScaler(with_mean=False)),
        ]), [col])
        for col in multi_label_cols
    ]

    transformers = []
    if numeric_cols:    transformers.append(("numeric",    numeric_pipe, numeric_cols))
    if binary_cols:     transformers.append(("binary",     cat_pipe,     binary_cols))
    if single_cat_cols: transformers.append(("single_cat", cat_pipe,     single_cat_cols))
    transformers.extend(mlb_pipes)

    return ColumnTransformer(transformers=transformers, remainder="drop")


# --------------------------------------------------------------------------- #
# Models + CV evaluation
# --------------------------------------------------------------------------- #
def make_models(preprocess):
    to_dense = FunctionTransformer(lambda X: X.toarray() if hasattr(X, "toarray") else X)
    return [
        ("LinearSVC", Pipeline([
            ("preprocess", preprocess), ("dense", to_dense),
            ("clf", SVC(kernel="linear", probability=True, class_weight="balanced",
                        random_state=42, decision_function_shape="ovr")),
        ])),
        ("RBF_SVC", Pipeline([
            ("preprocess", preprocess), ("dense", to_dense),
            ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced",
                        random_state=42, decision_function_shape="ovr")),
        ])),
        ("RandomForest", Pipeline([
            ("preprocess", preprocess), ("dense", to_dense),
            ("clf", RandomForestClassifier(n_estimators=500, random_state=42,
                                           class_weight="balanced",
                                           min_samples_leaf=3, max_features="sqrt")),
        ])),
    ]


def _score_split(y_true, y_pred, proba, is_multiclass):
    out = {
        "accuracy":          accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro":          f1_score(y_true, y_pred, average="macro"),
    }
    try:
        if is_multiclass:
            out["roc_auc_ovr_macro"] = roc_auc_score(y_true, proba, multi_class="ovr",
                                                    average="macro")
        else:
            out["roc_auc"] = roc_auc_score(y_true, proba[:, 1])
    except Exception:
        pass
    return out


def evaluate_task(task_name, X, y, *, models, cv, is_multiclass):
    rows = []
    for split_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
        for name, model in models:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(Xtr, ytr)
                ypred = model.predict(Xte)
                yproba = model.predict_proba(Xte)
            scores = _score_split(yte, ypred, yproba, is_multiclass)
            rows.append({"task": task_name, "model": name, "split": split_idx, **scores})
    per_split = pd.DataFrame(rows)
    summary = (per_split
               .drop(columns=["split"])
               .groupby(["task", "model"])
               .agg(["mean", "std"])
               .round(4)
               .reset_index())
    return per_split, summary


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_ml_pipeline(
    *,
    data_root      : Path,
    labels_xlsx    : Path,
    output_dir     : Path,
    json_suffix    : str,
    extra_numeric  : Iterable[str] = (),
    extra_binary   : Iterable[str] = (),
    extra_single_cat: Iterable[str] = (),
    extra_multi_label: Iterable[str] = (),
    cv_splits      : int = 4,
    cv_repeats     : int = 10,
    rebuild_csv    : bool = False,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"features_{json_suffix.strip('.json').lstrip('_')}.csv"

    # 1) Discover and flatten JSONs.
    discovered = discover_subject_jsons(data_root, json_suffix)
    if not discovered:
        raise FileNotFoundError(
            f"No '*{json_suffix}' files found under {data_root}.")
    print(f"[1/5] Found {len(discovered)} subjects with suffix '{json_suffix}'.")

    if rebuild_csv or not csv_path.exists():
        json_paths = [p for _, _, p in discovered]
        json_to_csv(json_paths, csv_path)
        print(f"      Flattened -> {csv_path}")
    df = pd.read_csv(csv_path)

    # 2) Attach folder-derived metadata + labels.
    folder_meta = pd.DataFrame(
        [{"subject_id": s, "who": w} for s, w, _ in discovered]
    )
    # source_file in CSV is the JSON stem -> drop the suffix to recover BraTS ID.
    df["subject_id"] = df[ID_COLUMN].astype(str).str.replace(
        json_suffix.replace(".json", "") + "$", "", regex=True)
    df = df.merge(folder_meta, on="subject_id", how="left")
    df = build_labels(df, labels_xlsx)

    n_idh    = df["IDH"].notna().sum()
    n_1p19q  = df["1p_19q"].notna().sum()
    n_grade  = df["Grade"].notna().sum()
    print(f"[2/5] Labels from cohort_merged.xlsx: "
          f"IDH={n_idh}, 1p/19q={n_1p19q}, Grade={n_grade} (of {len(df)} subjects)")

    # 3) Build preprocessor.
    numeric_cols     = list(BASE_NUMERIC_COLS)     + list(extra_numeric)
    binary_cols      = list(BASE_BINARY_COLS)      + list(extra_binary)
    single_cat_cols  = list(BASE_SINGLE_CAT_COLS)  + list(extra_single_cat)
    multi_label_cols = list(BASE_MULTI_LABEL_COLS) + list(extra_multi_label)
    preprocess = build_preprocessor(df.columns, numeric_cols, binary_cols,
                                    single_cat_cols, multi_label_cols)
    feature_cols = [c for c in (numeric_cols + binary_cols + single_cat_cols
                                + multi_label_cols) if c in df.columns]
    print(f"[3/5] Using {len(feature_cols)} feature columns.")

    # 4) Tasks (drop rows missing the target).
    tasks = [
        ("IDH",    "IDH",    False),
        ("1p_19q", "1p_19q", False),
        ("Grade",  "Grade",  True),
    ]
    models = make_models(preprocess)
    cv = RepeatedStratifiedKFold(n_splits=cv_splits, n_repeats=cv_repeats,
                                 random_state=42)

    # 5) Run CV.
    all_per_split, all_summary = [], []
    for task_name, target, is_mc in tasks:
        sub = df[df[target].notna()].copy()
        if sub[target].nunique() < 2:
            print(f"[skip] task '{task_name}': only one class present.")
            continue
        X = sub[feature_cols]
        y = sub[target].astype(int if is_mc else "object")
        print(f"[4/5] Evaluating {task_name}: n={len(sub)}, "
              f"classes={sorted(y.unique())}")
        ps, sm = evaluate_task(task_name, X, y, models=models, cv=cv,
                               is_multiclass=is_mc)
        all_per_split.append(ps)
        all_summary.append(sm)

    if not all_summary:
        print("No runnable tasks.")
        return

    per_split_df = pd.concat(all_per_split, ignore_index=True)
    summary_df   = pd.concat(all_summary,   ignore_index=True)
    per_split_df.to_csv(output_dir / "per_split_metrics.csv", index=False)
    summary_df.to_csv(  output_dir / "summary_metrics.csv",   index=False)

    print("\n[5/5] SUMMARY (mean ± std across CV splits):")
    print(summary_df.to_string(index=False))
    return summary_df
