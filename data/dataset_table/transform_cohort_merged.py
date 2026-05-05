"""
Transform cohort_merged.xlsx into a tidy patient-level table.

Steps (per user spec):
  1. Add WHO2021 = WHO 2021 (original); fill missing values from IDH/Grade/
     1p/19q rules.
  2. Add WHO2016 = Tumor Subtype (original) modified by IDH/Grade rules.
  3. Drop columns ending with "(generated)".
  4. Strip the "(original)" suffix from the remaining columns.
  5. Drop the Tumor Subtype columns.
  6. Reorder to: Center ID, Subject ID, Dataset, Hospital, BraTS2021, Gender,
     Age (years), IDH, 1p/19q, Grade, WHO2016, WHO2021.

Input  : cohort_merged.xlsx (in this folder)
Output : cohort_merged.xlsx (overwritten in place)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
IN_FILE = BASE_DIR / "cohort_merged.xlsx"
OUT_FILE = BASE_DIR / "cohort_merged.xlsx"

FINAL_COLUMNS = [
    "Center ID",
    "Subject ID",
    "Dataset",
    "Hospital",
    "BraTS2021",
    "Gender",
    "Age (years)",
    "IDH",
    "1p/19q",
    "Grade",
    "WHO2016",
    "WHO2021",
]

WHO2021_ASTRO = "Astrocytoma, IDH-mutant"
WHO2021_OLIGO = "Oligodendroglioma, IDH-mutant, 1p/19q-codeleted"
WHO2021_GBM = "Glioblastoma, IDH-wildtype"

WHO2016_GBM_MUT = "Glioblastoma, IDH-mutant"
WHO2016_GBM_WT = "Glioblastoma, IDH-wildtype"
WHO2016_ASTRO_MUT = "Astrocytoma, IDH-mutant"
WHO2016_OLIGO_MUT = "Oligodendroglioma, IDH-mutant, 1p/19q-codeleted"


def _is_missing(v) -> bool:
    """Treat NaN, None, blank, 'x', 'nan' as missing."""
    if v is None:
        return True
    try:
        if pd.isna(v):
            return True
    except (TypeError, ValueError):
        pass
    return isinstance(v, str) and v.strip().lower() in {"", "x", "nan", "none"}


def _to_grade(v):
    if _is_missing(v):
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _norm(v):
    return None if _is_missing(v) else str(v).strip().lower()


def fill_who2021(row) -> object:
    """Step 1: copy WHO 2021 (original); back-fill missing using IDH/Grade/1p19q."""
    raw = row.get("WHO 2021 (original)")
    if not _is_missing(raw):
        return raw

    idh = _norm(row.get("IDH (original)"))
    grade = _to_grade(row.get("Grade"))
    p19q = _norm(row.get("1p/19q (original)"))

    is_mut = idh is not None and "mut" in idh
    is_wt = idh is not None and ("wild" in idh)

    if is_mut and grade == 4:
        return WHO2021_ASTRO
    if is_mut and grade in (2, 3):
        if p19q is not None and "co" in p19q and "del" in p19q:
            return WHO2021_OLIGO
        if p19q is not None and "intact" in p19q:
            return WHO2021_ASTRO
    if is_wt and grade == 4:
        return WHO2021_GBM

    return None  # Leaves the cell empty if no rule applies.


def derive_who2016(row) -> object:
    """Step 2: copy Tumor Subtype (original); apply IDH/Grade/1p19q rules.

    Astrocytoma and Oligodendroglioma are only renamed when their 1p/19q
    status is consistent with the target diagnosis:
        - Astrocytoma + Grade 2/3 + 1p/19q intact     -> Astrocytoma, IDH-mutant
        - Oligodendroglioma + Grade 2/3 + 1p/19q
          co-deleted                                  -> Oligodendroglioma, IDH-mutant, 1p/19q-codeleted
    Rows that do not satisfy the 1p/19q requirement keep the original
    Tumor Subtype value.
    """
    raw = row.get("Tumor Subtype (original)")
    if _is_missing(raw):
        return None

    subtype = str(raw).strip().lower()
    idh = _norm(row.get("IDH (original)"))
    grade = _to_grade(row.get("Grade"))
    p19q = _norm(row.get("1p/19q (original)"))

    is_mut = idh is not None and "mut" in idh
    is_wt = idh is not None and "wild" in idh
    is_codel = p19q is not None and "co" in p19q and "del" in p19q
    is_intact = p19q is not None and "intact" in p19q

    if subtype == "glioblastoma" and is_mut:
        return WHO2016_GBM_MUT
    if subtype == "glioblastoma" and is_wt:
        return WHO2016_GBM_WT
    if subtype == "astrocytoma" and grade in (2, 3) and is_intact:
        return WHO2016_ASTRO_MUT
    if subtype == "oligodendroglioma" and grade in (2, 3) and is_codel:
        return WHO2016_OLIGO_MUT
    # Histologic oligodendrogliomas without confirmed 1p/19q co-deletion
    # cannot be called oligodendroglioma under WHO 2021, but for WHO2016
    # we keep the histologic label and just normalise its capitalisation.
    if subtype == "oligodendroglioma":
        return "Oligodendroglioma"
    if subtype == "oligoastrocytoma":
        return "Oligoastrocytoma"

    # No rule matched — keep the original value as-is.
    return raw


def main() -> None:
    df = pd.read_excel(IN_FILE)
    print(f"[in ] {IN_FILE.name}: {df.shape}")

    # Step 1 & 2: derive new columns from the original/IDH/Grade fields.
    df["WHO2021"] = df.apply(fill_who2021, axis=1)
    df["WHO2016"] = df.apply(derive_who2016, axis=1)

    # Step 3: drop "(generated)" columns.
    drop_generated = [c for c in df.columns if c.strip().endswith("(generated)")]
    df = df.drop(columns=drop_generated)

    # Step 4: strip the "(original)" suffix from the remaining columns.
    rename_map = {}
    for c in df.columns:
        if c.strip().endswith("(original)"):
            new_name = c.rsplit("(original)", 1)[0].strip()
            rename_map[c] = new_name
    df = df.rename(columns=rename_map)

    # Step 5: drop the Tumor Subtype columns (the (generated) variant was
    # already removed above; this also handles the renamed (original) one).
    df = df.drop(
        columns=[c for c in ("Tumor Subtype", "Tumor Subtype (generated)") if c in df.columns]
    )

    # Step 6: enforce the final column order (and drop everything else).
    missing = [c for c in FINAL_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Expected columns missing after transform: {missing}")
    df = df.loc[:, FINAL_COLUMNS]

    df.to_excel(OUT_FILE, index=False)
    print(f"[out] {OUT_FILE.name}: {df.shape}")
    print(f"[out] columns: {list(df.columns)}")
    print("\nWHO2021 value counts:")
    print(df["WHO2021"].value_counts(dropna=False))
    print("\nWHO2016 value counts:")
    print(df["WHO2016"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
