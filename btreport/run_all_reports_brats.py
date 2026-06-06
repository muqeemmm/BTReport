import argparse
import glob
import subprocess
import os
import json
from .utils.log import get_logger
from os.path import join
from pathlib import Path

# logger = get_logger("btreport.run_all_reports", subject="run_all")
logger = get_logger("batchgen")


def main():
    parser = argparse.ArgumentParser(description="Run generate_report.py on ALL subject folders inside a directory.")

    # parser.add_argument("--save_dir", type=str, required=True, help="Path to save results.")

    dataset_root = "/media/ist/data/Muqeem/Projects/Brain_Project/Classification Code/BTReport/data/Dataset_AKU_WHO"
    SKIP_DIRS = {"JSONs BTReport"}

    parser.add_argument("--clear_tmp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ncr_label", type=int, default=1)
    parser.add_argument("--ed_label", type=int, default=2)
    parser.add_argument("--et_label", type=int, default=4)
    parser.add_argument("--llm", type=str, default="gpt-oss:120b")
    parser.add_argument("--eval", action="store_true", help="Running evaluation after generation.")
    parser.add_argument("--update_t2_flair", action="store_true",
                        help="Recompute T2-FLAIR mismatch from existing *_metadata_final.json files only.")
    parser.add_argument(
        "--tz_method",
        type=str,
        default="method_b",
        choices=["method_a", "method_b"],
        help=(
            "Transition zone thickness method passed to generate_report. "
            "'method_a': distance-transform (recommended). "
            "'method_b': ray-casting with FLAIR and T1CE."
        ),
    )

    args = parser.parse_args()

    # save_dir = os.path.realpath(args.save_dir)
    llm = args.llm

    generate_script = "btreport.generate_report"

    # merged_path = os.path.join(save_dir, "merged_reports_btreport.json")
    # merged = {}
    # if os.path.exists(merged_path):
    #     with open(merged_path, "r") as f:
    #         merged = json.load(f)

    not_processed_list = []

    tumor_type_dirs = sorted(
        d for d in os.listdir(dataset_root)
        if not d.startswith(".") and d not in SKIP_DIRS
        and os.path.isdir(os.path.join(dataset_root, d))
    )

    for tumor_type in tumor_type_dirs:
        tumor_type_dir = os.path.join(dataset_root, tumor_type)
        logger.info(f"\n{'='*60}\nTumor type: {tumor_type}\n{'='*60}")

        sorted_entries = sorted(
            e for e in os.listdir(tumor_type_dir) if not e.startswith(".")
        )

        for entry in sorted_entries:
            subject_dir = os.path.join(tumor_type_dir, entry)

            if not os.path.isdir(subject_dir):
                continue

            final_json      = join(subject_dir, f"{entry}_metadata_final.json")
            t2_updated_json = join(subject_dir, f"{entry}_metadata_final_t2_flair_update.json")

            if args.update_t2_flair:
                # Prerequisite: full pipeline must have already run
                if not os.path.exists(final_json):
                    logger.info(f"Skipping {entry} — {entry}_metadata_final.json not found.")
                    not_processed_list.append(entry)
                    continue
                # Skip if update output already exists
                if not args.overwrite and os.path.exists(t2_updated_json):
                    logger.info(f"Skipping {entry} — {entry}_metadata_final_t2_flair_update.json already exists.")
                    continue
            else:
                # Normal mode: skip if full pipeline already completed
                if not args.overwrite and os.path.exists(final_json):
                    logger.info(f"Skipping {entry} — {entry}_metadata_final.json already exists.")
                    continue

            if not (os.path.exists(join(subject_dir, f"{entry}_seg.nii.gz"))
                    or os.path.exists(join(subject_dir, f"{entry}_seg_pred.nii.gz"))):
                logger.info(f"Skipping {entry} — no segmentation found.")
                not_processed_list.append(entry)
                continue

            paths = build_subject_paths(entry)

            logger.info(f"\n=== Processing {entry} ({tumor_type}) ===")

            cmd = [
                "python3",
                "-m",
                generate_script,
                "--subject_folder",
                subject_dir,
                "--ncr_label",
                str(args.ncr_label),
                "--ed_label",
                str(args.ed_label),
                "--et_label",
                str(args.et_label),
            ]

            if args.update_t2_flair:
                cmd.append("--update_t2_flair")
            else:
                cmd.extend(["--llm", llm, "--tz_method", args.tz_method])

            if args.clear_tmp:
                cmd.append("--clear_tmp")
            if args.overwrite:
                cmd.append("--overwrite")

            subprocess.run(cmd, check=True)

    logger.info("\nFinished processing all subjects.")
    if not_processed_list:
        not_processed_path = os.path.join("/media/ist/data/Muqeem/Projects/Brain_Project/Classification Code/BTReport/docs", "not_processed.txt")
        with open(not_processed_path, "w") as f:
            for entry in not_processed_list:
                f.write(entry + "\n")
        logger.info(f"Saved not-processed list to {not_processed_path}")


def build_subject_paths(subject_id):
    """
    Build external artifact paths for a subject, assuming a shared directory layout.
    """
    base_brats = "/gscratch/kurtlab/brats2023/data"
    base_mni_subj = "/gscratch/kurtlab/brats2023/data/MNI152_IN_SUBJECT_SPACE"
    base_mni = "/gscratch/kurtlab/brats2023/data/MNI152"
    base_midline = "/gscratch/kurtlab/brats2023/data/midline"
    base_synthseg = "/gscratch/scrubbed/juampablo/MNI152_IN_SUBJECT_SPACE/synthseg/brats-gli"

    subj_rel = f"brats-gli/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/{subject_id}"

    return {
        "t1_path": os.path.join(
            base_brats, subj_rel, f"{subject_id}-t1n.nii.gz"
        ),
        "tumor_path": os.path.join(
            base_brats, subj_rel, f"{subject_id}-seg.nii.gz"
        ),
        "MNI_in_subject": os.path.join(
            base_mni_subj, "moved", subj_rel, f"{subject_id}-t1n.nii.gz"
        ),
        "MNI_in_subject_transform": os.path.join(
            base_mni_subj, "transforms", subj_rel, f"{subject_id}-t1n.nii.gz"
        ),
        "subject_in_MNI": os.path.join(
            base_mni, subj_rel, f"{subject_id}-t1n.nii.gz"
        ),
        "subject_in_MNI_transform": os.path.join(
            base_mni, "transforms", subj_rel, f"{subject_id}-t1n.nii.gz"
        ),
        "tumorseg_in_MNI": os.path.join(
            base_mni, subj_rel, f"{subject_id}-seg.nii.gz"
        ),
        "patient_midline": os.path.join(
            base_midline, subj_rel, "patient_midline.nii.gz"
        ),
        "anatseg": os.path.join(
            base_synthseg, f"{subject_id}.nii.gz"
        ),
    }


def update_merged_reports(root_dir, subject_id, llm, real_report_key="Clinical Report", output_filename="merged_reports_btreport.json", subject_report_filename="patient_metadata_btreport.json"):
    merged_path = os.path.join(root_dir, output_filename)
    if os.path.exists(merged_path):
        with open(merged_path, "r") as f:
            merged = json.load(f)
    else:
        merged = {}

    subject_report_path = os.path.join(root_dir, subject_id, subject_report_filename)
    if not os.path.exists(subject_report_path):
        logger.info(f"Missing {subject_report_path}, skipping merge.")
        return

    try:
        with open(subject_report_path, "r") as f:
            subject_btreport = json.load(f)
    except Exception as e:
        logger.info(f"Failed to read {subject_report_path}: {e}")
        return

    bt_generated_key = f"BTReport Generated Report ({llm})"
    predicted_key = f"Predicted Report ({llm})"

    # if real_report_key not in subject_btreport:
    #     logger.info(f"Merge error: '{real_report_key}' missing in {subject_report_path}")
    #     return

    if bt_generated_key not in subject_btreport:
        logger.info(f"Merge error: '{bt_generated_key}' missing in {subject_report_path}")
        return

    if subject_id not in merged:
        merged[subject_id] = {}

    # Store ground truth
    # merged[subject_id][real_report_key] = subject_btreport[real_report_key]

    # Store prediction
    merged[subject_id][predicted_key] = subject_btreport[bt_generated_key]

    # Safe write
    tmp_path = merged_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(merged, f, indent=2)
    os.replace(tmp_path, merged_path)

    logger.info(f"Merged {subject_id} [{llm}]")


if __name__ == "__main__":
    main()
