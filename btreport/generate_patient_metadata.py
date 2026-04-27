"""
Generate per-subject metadata JSONs containing age and gender,
looked up from data/dataset_table/master_file.xlsx.

Writes <subject_folder>/<subject_id>_metadata.json with fields:
  - age_years
  - gender

Usage:
    python3 -m btreport.generate_patient_metadata \
        --subjects_dir <path/to/JSONs BTReport>  \
        --master_file  <path/to/master_file.xlsx>
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd


def build_lookup(master_file: str) -> dict:
    """Return a dict mapping subject_id -> {age_years, gender}."""
    df = pd.read_excel(master_file)

    lookup = {}

    for _, row in df.iterrows():
        gender = str(row.get("Gender", "")).strip()
        age_raw = row.get("Age (years)", "")
        try:
            age = int(float(age_raw))
        except (ValueError, TypeError):
            age = None

        entry = {"age_years": age, "gender": gender if gender else None}

        # BraTS2021 column: e.g. "BraTS2021_00005"
        brats_id = str(row.get("BraTS2021", "")).strip()
        if brats_id and brats_id.lower() != "x" and brats_id != "nan":
            lookup[brats_id] = entry

        # Local ID column: e.g. "UCSF-PDGM-460", "EGD-0004"
        local_id = str(row.get("Local ID", "")).strip()
        if local_id and local_id.lower() != "x" and local_id != "nan":
            lookup[local_id] = entry

    return lookup


def main(args: argparse.Namespace):
    subjects_dir = Path(args.subjects_dir)
    lookup = build_lookup(args.master_file)

    subject_dirs = sorted(
        p for p in subjects_dir.iterdir() if p.is_dir()
    )

    found = 0
    missing = []

    for subject_path in subject_dirs:
        subject_id = subject_path.name
        entry = lookup.get(subject_id)

        out_path = subject_path / f"{subject_id}_metadata_clinical.json"

        if entry is None:
            print(f"[WARN] No match in master_file for: {subject_id}")
            missing.append(subject_id)
            # Still write a stub so downstream code doesn't break
            entry = {"age_years": None, "gender": None}

        with open(out_path, "w") as f:
            json.dump(entry, f, indent=2)

        found += 1
        print(f"[OK] {subject_id}: age={entry['age_years']}, gender={entry['gender']}  -> {out_path.name}")

    print(f"\nDone. Wrote {found} metadata files ({len(missing)} subjects had no match).")
    if missing:
        print("Subjects with no match:", missing)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate per-subject metadata JSONs from master_file.xlsx."
    )
    parser.add_argument(
        "--subjects_dir",
        type=str,
        required=True,
        help="Path to the directory containing subject folders (e.g. 'JSONs BTReport').",
    )
    parser.add_argument(
        "--master_file",
        type=str,
        default="data/dataset_table/master_file.xlsx",
        help="Path to master_file.xlsx. Default: data/dataset_table/master_file.xlsx",
    )
    args = parser.parse_args()
    main(args)
