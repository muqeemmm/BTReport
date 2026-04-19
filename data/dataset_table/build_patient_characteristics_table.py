"""
Build the Patient Characteristics Table.

Inputs (all expected in the same folder as this script):
    - aku_annotations_with_duplicates.xlsx
        Source of the final cohort. The unique values of the "Subject ID"
        column give the set of patients to include.
    - master_file.xlsx
        Source of demographic / molecular / imaging metadata for each
        patient. Patients are joined using the "Center ID" column of the
        AKU file, which matches either "Local ID" or "BraTS2021" in the
        master file.
    - subjects_to_discard.txt
        One Subject ID per line. Any Subject ID listed here is excluded
        from the final cohort.

Output:
    - patient_characteristics_table.xlsx
        A formatted Patient Characteristics Table, with datasets as
        columns and each characteristic (Age, Sex, IDH, 1p/19q, Molecular
        Subtype, Tumor Grade, Multi-class Segmentation) as row groups.
    - cohort_merged.xlsx
        The merged per-patient data used to compute the table (handy for
        auditing / downstream use).

Usage:
    python build_patient_characteristics_table.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

AKU_FILE = BASE_DIR / "aku_annotations_with_duplicates.xlsx"
MASTER_FILE = BASE_DIR / "master_file.xlsx"
DISCARD_FILE = BASE_DIR / "subjects_to_discard.txt"

OUT_TABLE = BASE_DIR / "patient_characteristics_table.xlsx"
OUT_MERGED = BASE_DIR / "cohort_merged.xlsx"

# Column order in the final table (left to right).
DATASET_ORDER = [
    "UCSF-PDGM",
    "UPENN-GBM",
    "TCGA-LGG",
    "TCGA-GBM",
    "IvyGAP",
    "EGD",
]


# ---------------------------------------------------------------------------
# Data loading / filtering / merging
# ---------------------------------------------------------------------------
def load_discard_list(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            ids.add(line)
    return ids


def build_cohort() -> pd.DataFrame:
    """Load inputs, apply the Subject ID filter, and join to the master file."""
    aku = pd.read_excel(AKU_FILE)
    master = pd.read_excel(MASTER_FILE)

    # Keep one row per unique Subject ID (first occurrence) and record its Center ID.
    aku_unique = (
        aku.dropna(subset=["Subject ID"])
        .drop_duplicates(subset=["Subject ID"], keep="first")
        .loc[:, ["Subject ID", "Center ID"]]
        .copy()
    )

    # Drop permanently-discarded subjects.
    discard = load_discard_list(DISCARD_FILE)
    before = len(aku_unique)
    aku_unique = aku_unique[~aku_unique["Subject ID"].isin(discard)].copy()
    print(
        f"[cohort] unique Subject IDs: {before} -> {len(aku_unique)} "
        f"after dropping {before - len(aku_unique)} discarded subject(s)."
    )

    # Build lookup tables from the master file. A Center ID can match either
    # a Local ID or a BraTS2021 ID, so try both.
    master_by_local = (
        master.dropna(subset=["Local ID"])
        .drop_duplicates(subset=["Local ID"], keep="first")
        .set_index("Local ID")
    )
    master_by_brats = (
        master.dropna(subset=["BraTS2021"])
        .drop_duplicates(subset=["BraTS2021"], keep="first")
        .set_index("BraTS2021")
    )

    merged_rows = []
    unmatched = []
    for _, row in aku_unique.iterrows():
        cid = row["Center ID"]
        mrow = None
        if pd.notna(cid) and cid in master_by_local.index:
            mrow = master_by_local.loc[cid].to_dict()
        elif pd.notna(cid) and cid in master_by_brats.index:
            mrow = master_by_brats.loc[cid].to_dict()
        if mrow is None:
            unmatched.append((row["Subject ID"], cid))
            continue
        mrow["Subject ID"] = row["Subject ID"]
        mrow["Center ID"] = cid
        merged_rows.append(mrow)

    if unmatched:
        print(f"[cohort] WARNING: {len(unmatched)} subjects could not be matched to master_file.")
        for s, c in unmatched[:10]:
            print(f"    {s} (Center ID: {c})")

    merged = pd.DataFrame(merged_rows)
    print(f"[cohort] merged rows: {len(merged)} / master columns: {merged.shape[1]}")
    return merged


# ---------------------------------------------------------------------------
# Value normalisation helpers
# ---------------------------------------------------------------------------
def _is_missing(x) -> bool:
    """Missing values in master_file are NaN or the literal string 'x'."""
    if x is None:
        return True
    try:
        if pd.isna(x):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(x, str) and x.strip().lower() in {"", "x", "nan", "none", "unknown"}:
        return True
    return False


def _coalesce(row: pd.Series, primary: str, fallback: str):
    """Return primary value if present, else fallback, else None."""
    v = row.get(primary)
    if not _is_missing(v):
        return v
    v = row.get(fallback)
    if not _is_missing(v):
        return v
    return None


def normalise_sex(v):
    if _is_missing(v):
        return "Unknown"
    s = str(v).strip().lower()
    if s.startswith("m"):
        return "Male"
    if s.startswith("f"):
        return "Female"
    return "Unknown"


def normalise_idh(v):
    if _is_missing(v):
        return "Unknown"
    s = str(v).strip().lower()
    if "mut" in s:
        return "Mutated"
    if "wild" in s:
        return "Wildtype"
    return "Unknown"


def normalise_1p19q(v):
    if _is_missing(v):
        return "Unknown"
    s = str(v).strip().lower()
    if "co" in s and "del" in s:  # co-deleted / rel. co-deleted
        return "Co-deleted"
    if "intact" in s or "non" in s or "not" in s:
        return "Intact"
    return "Unknown"


def normalise_subtype(v, grade=None):
    """Map WHO-2021 diagnosis string to a molecular subtype.

    Grade 4 IDH-mutant astrocytomas are reported as their own subtype,
    distinct from WHO grade 2/3 IDH-mutant astrocytomas.
    """
    if _is_missing(v):
        return "Unknown"
    s = str(v).strip().lower()

    # Resolve an integer grade if possible (used to split astrocytomas).
    g = None
    if not _is_missing(grade):
        try:
            g = int(float(grade))
        except (TypeError, ValueError):
            g = None

    if "oligodendroglioma" in s:
        return "Oligodendroglioma"
    if "astrocytoma" in s and ("mutant" in s or "idh-mut" in s.replace(" ", "-")):
        return "Grade 4 IDH-mutant astrocytoma" if g == 4 else "IDH-mutant astrocytoma"
    if "glioblastoma" in s:
        return "IDH-wildtype glioblastoma"
    if "astrocytoma" in s:
        return "Grade 4 IDH-mutant astrocytoma" if g == 4 else "IDH-mutant astrocytoma"
    return "Unknown"


def normalise_grade(v):
    if _is_missing(v):
        return "Unknown"
    try:
        g = int(float(v))
    except (TypeError, ValueError):
        return "Unknown"
    if g in (2, 3, 4):
        return f"WHO grade {g}"
    return "Unknown"


def normalise_segmentation(v):
    """Multiclass Tumor Segmentation: 1 = Manual, 0 = Automatic."""
    if _is_missing(v):
        return "Unknown"
    try:
        k = int(float(v))
    except (TypeError, ValueError):
        return "Unknown"
    if k == 1:
        return "Manual"
    if k == 0:
        return "Automatic"
    return "Unknown"


# ---------------------------------------------------------------------------
# Table construction
# ---------------------------------------------------------------------------
def prepare_table_dataframe(merged: pd.DataFrame) -> pd.DataFrame:
    df = merged.copy()
    # Pull best-available molecular values (original first, then generated).
    df["Subtype_raw"] = df.apply(
        lambda r: _coalesce(r, "WHO 2021 (original)", "WHO 2021 (generated)"),
        axis=1,
    )
    # Normalised columns used for counting.
    df["_Sex"] = df["Gender"].map(normalise_sex)
    df["_IDH"] = df["IDH (original)"].map(normalise_idh)
    df["_1p19q"] = df["1p/19q (original)"].map(normalise_1p19q)
    df["_Subtype"] = df.apply(
        lambda r: normalise_subtype(r["Subtype_raw"], r.get("Grade")), axis=1
    )
    df["_Grade"] = df["Grade"].map(normalise_grade)
    df["_Seg"] = df["Multiclass Tumor Segmentation"].map(normalise_segmentation)
    df["_Age"] = pd.to_numeric(df["Age (years)"], errors="coerce")
    return df


def _fmt_n_pct(n: int, total: int) -> str:
    if total <= 0:
        return "0 (0%)"
    pct = 100.0 * n / total
    return f"{n} ({pct:.0f}%)"


def _fmt_mean_sd(series: pd.Series) -> str:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return "--"
    mean = series.mean()
    sd = series.std(ddof=1) if len(series) > 1 else 0.0
    return f"{mean:.1f} ± {sd:.1f}"


def _counts_by_dataset(df: pd.DataFrame, col: str, datasets: list[str]) -> dict:
    """Return {dataset: Series(value_counts)} including a TOTAL key."""
    out = {}
    for ds in datasets:
        sub = df[df["Dataset"] == ds]
        out[ds] = sub[col].value_counts(dropna=False)
    out["TOTAL"] = df[col].value_counts(dropna=False)
    return out


def _cell(counts_by_ds: dict, ds: str, value: str, totals: dict) -> str:
    n = int(counts_by_ds[ds].get(value, 0))
    return _fmt_n_pct(n, totals[ds])


def build_table_rows(df: pd.DataFrame, datasets: list[str]) -> tuple[list[list], dict]:
    """Return a list of rows ready to be written into the worksheet.

    Sub-rows labelled ``Unknown`` whose counts are zero in every column
    (including TOTAL) are omitted to keep the table tight.
    """
    # Dataset-level totals (including TOTAL column).
    totals = {ds: int((df["Dataset"] == ds).sum()) for ds in datasets}
    totals["TOTAL"] = int(len(df))

    all_ds = datasets + ["TOTAL"]
    header = ["Characteristic"] + [f"{ds} (n={totals[ds]})" for ds in all_ds]

    rows: list[list] = [header]

    def _count_row(label: str, counts: dict) -> tuple[list, int]:
        """Build one sub-row and return (row, total-count-across-columns)."""
        row = [f"  {label}"]
        total_n = 0
        for ds in all_ds:
            n = int(counts[ds].get(label, 0)) if ds != "TOTAL" else int(counts["TOTAL"].get(label, 0))
            total_n += n
            row.append(_fmt_n_pct(n, totals[ds]))
        return row, total_n

    def _append(label: str, counts: dict, row_list: list) -> None:
        """Append a sub-row, skipping Unknown rows that are zero everywhere."""
        row, total_n = _count_row(label, counts)
        if label == "Unknown" and total_n == 0:
            return
        row_list.append(row)

    # ---- Age (years) ----
    rows.append(["Age (years)"] + [""] * len(all_ds))

    known_row = ["  Known (mean ± SD)"]
    for ds in all_ds:
        sub = df if ds == "TOTAL" else df[df["Dataset"] == ds]
        known_row.append(_fmt_mean_sd(sub["_Age"]))
    rows.append(known_row)

    # Age Unknown – only keep if any dataset has missing ages.
    unknown_row = ["  Unknown"]
    total_unknown = 0
    for ds in all_ds:
        sub = df if ds == "TOTAL" else df[df["Dataset"] == ds]
        n_unknown = int(sub["_Age"].isna().sum())
        total_unknown += n_unknown
        unknown_row.append(_fmt_n_pct(n_unknown, totals[ds]))
    if total_unknown > 0:
        rows.append(unknown_row)

    # ---- Sex ----
    rows.append(["Sex"] + [""] * len(all_ds))
    counts = _counts_by_dataset(df, "_Sex", datasets)
    for label in ["Female", "Male", "Unknown"]:
        _append(label, counts, rows)

    # ---- IDH ----
    rows.append(["IDH"] + [""] * len(all_ds))
    counts = _counts_by_dataset(df, "_IDH", datasets)
    for label in ["Mutated", "Wildtype", "Unknown"]:
        _append(label, counts, rows)

    # ---- 1p/19q ----
    rows.append(["1p/19q"] + [""] * len(all_ds))
    counts = _counts_by_dataset(df, "_1p19q", datasets)
    for label in ["Co-deleted", "Intact", "Unknown"]:
        _append(label, counts, rows)

    # ---- Molecular Subtype ----
    rows.append(["Molecular Subtype"] + [""] * len(all_ds))
    counts = _counts_by_dataset(df, "_Subtype", datasets)
    for label in [
        "Oligodendroglioma",
        "IDH-mutant astrocytoma",
        "Grade 4 IDH-mutant astrocytoma",
        "IDH-wildtype glioblastoma",
        "Unknown",
    ]:
        _append(label, counts, rows)

    # ---- Tumor Grade ----
    rows.append(["Tumor Grade"] + [""] * len(all_ds))
    counts = _counts_by_dataset(df, "_Grade", datasets)
    for label in ["WHO grade 2", "WHO grade 3", "WHO grade 4", "Unknown"]:
        _append(label, counts, rows)

    # ---- Multi-class Segmentation ----
    rows.append(["Multi-class Segmentation"] + [""] * len(all_ds))
    counts = _counts_by_dataset(df, "_Seg", datasets)
    for label in ["Manual", "Automatic", "Unknown"]:
        _append(label, counts, rows)

    return rows, totals


# ---------------------------------------------------------------------------
# Excel writer
# ---------------------------------------------------------------------------
def write_table_xlsx(rows: list[list], totals: dict, datasets: list[str], out_path: Path) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "Patient Characteristics"

    # Title row.
    n_cols = len(rows[0])
    title = f"Table 1. Patient Characteristics Table (N = {totals['TOTAL']})"
    ws.cell(row=1, column=1, value=title)
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=n_cols)
    title_cell = ws.cell(row=1, column=1)
    title_cell.font = Font(name="Times New Roman", size=13, bold=True)
    title_cell.alignment = Alignment(horizontal="center", vertical="center")

    # Header + body.
    thin = Side(style="thin", color="000000")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    header_fill = PatternFill("solid", start_color="D9E1F2")
    group_fill = PatternFill("solid", start_color="F2F2F2")

    group_labels = {
        "Age (years)",
        "Sex",
        "IDH",
        "1p/19q",
        "Molecular Subtype",
        "Tumor Grade",
        "Multi-class Segmentation",
    }

    start_row = 2  # header row in worksheet
    for r_idx, row in enumerate(rows):
        ws_row = start_row + r_idx
        for c_idx, value in enumerate(row, start=1):
            cell = ws.cell(row=ws_row, column=c_idx, value=value)
            cell.border = border
            cell.font = Font(name="Times New Roman", size=11)
            if c_idx == 1:
                cell.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
            else:
                cell.alignment = Alignment(horizontal="center", vertical="center")

            if r_idx == 0:
                cell.fill = header_fill
                cell.font = Font(name="Times New Roman", size=11, bold=True)
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            elif row[0] in group_labels and c_idx == 1:
                cell.fill = group_fill
                cell.font = Font(name="Times New Roman", size=11, bold=True)
            elif row[0] in group_labels:
                cell.fill = group_fill

    # Column widths.
    ws.column_dimensions["A"].width = 34
    for c in range(2, n_cols + 1):
        ws.column_dimensions[get_column_letter(c)].width = 18

    ws.row_dimensions[1].height = 22

    # Footnote.
    foot_row = start_row + len(rows) + 1
    footnote = (
        "Values are N (%), except Age which is reported as mean ± SD. "
        "Cohort defined by the unique Subject IDs in aku_annotations_with_duplicates.xlsx; "
        "subjects listed in subjects_to_discard.txt are excluded. "
        "Metadata is sourced from master_file.xlsx, joined via the AKU Center ID."
    )
    ws.cell(row=foot_row, column=1, value=footnote)
    ws.merge_cells(start_row=foot_row, start_column=1, end_row=foot_row, end_column=n_cols)
    foot_cell = ws.cell(row=foot_row, column=1)
    foot_cell.font = Font(name="Times New Roman", size=9, italic=True)
    foot_cell.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
    ws.row_dimensions[foot_row].height = 40

    wb.save(out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    merged = build_cohort()

    # Save the merged per-patient file for auditing.
    merged.to_excel(OUT_MERGED, index=False)
    print(f"[output] wrote merged cohort to {OUT_MERGED.name}")

    df = prepare_table_dataframe(merged)

    # Keep only datasets that actually appear in the cohort, in the
    # preferred order, then append any stragglers alphabetically.
    present = [ds for ds in DATASET_ORDER if (df["Dataset"] == ds).any()]
    extras = sorted(set(df["Dataset"].dropna().unique()) - set(present))
    datasets = present + extras

    rows, totals = build_table_rows(df, datasets)
    write_table_xlsx(rows, totals, datasets, OUT_TABLE)
    print(f"[output] wrote Patient Characteristics Table to {OUT_TABLE.name}")
    print(f"[output] total patients in table: {totals['TOTAL']}")
    for ds in datasets:
        print(f"    {ds}: {totals[ds]}")


if __name__ == "__main__":
    main()
