"""
Collect per-subject JSONs from all 4 WHO subtype folders into a single
'JSONs' directory under Dataset_AKU_WHO/, each subject in its own subfolder.

Source files (inside each subject folder):
  <subject>_metadata_no_clinical.json           → JSONs/<subject>/<subject>_metadata_btreport.json
  <subject>_metadata_final_t2_flair_update.json → JSONs/<subject>/<subject>_metadata_btreport_pp.json
"""

import os
import shutil
from pathlib import Path

DATASET_ROOT = Path("/media/ist/data/Muqeem/Projects/Brain_Project/Classification Code/BTReport/data/Dataset_AKU_WHO")
OUTPUT_DIR   = DATASET_ROOT / "JSONs"
SKIP_DIRS    = {"JSONs BTReport", "JSONs"}

SUFFIX_MAP = {
    "_metadata_no_clinical.json":            "_metadata_btreport.json",
    "_metadata_final_t2_flair_update.json":  "_metadata_btreport_pp.json",
}

OUTPUT_DIR.mkdir(exist_ok=True)

copied = 0
missing = []

tumor_type_dirs = sorted(
    d for d in DATASET_ROOT.iterdir()
    if d.is_dir() and not d.name.startswith(".") and d.name not in SKIP_DIRS
)

for tumor_type_dir in tumor_type_dirs:
    print(f"\n--- {tumor_type_dir.name} ---")
    for subject_dir in sorted(tumor_type_dir.iterdir()):
        if not subject_dir.is_dir() or subject_dir.name.startswith("."):
            continue
        subject = subject_dir.name
        subject_out_dir = OUTPUT_DIR / subject
        subject_out_dir.mkdir(exist_ok=True)
        for src_suffix, dst_suffix in SUFFIX_MAP.items():
            src = subject_dir / f"{subject}{src_suffix}"
            dst = subject_out_dir / f"{subject}{dst_suffix}"
            if src.exists():
                shutil.copy2(src, dst)
                print(f"  Copied {src.name}  →  {subject}/{dst.name}")
                copied += 1
            else:
                print(f"  MISSING: {src.name}")
                missing.append(str(src))

print(f"\nDone. {copied} files copied to {OUTPUT_DIR}")
if missing:
    print(f"{len(missing)} missing source files:")
    for m in missing:
        print(f"  {m}")
