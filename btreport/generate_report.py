from pathlib import Path
from .utils import register, plotting, anat_segmentation
from .utils.log import get_logger
from .llm_report_generation.ollama_report_gen import generate_llm_report
from .midline_shift.midline_shift3d import midline_shift_3d
from .vasari_features import ExtractVASARI
from .additional_features.additional_features_3d import (compute_sphericity,
                                                         compute_vasari_style_morphometrics,
                                                         compute_transition_zone_thickness,
                                                         ExtractT2FLAIRMismatch)
import ants
from antspynet.utilities import brain_extraction as be

# from .vasari_features.extract_vasari_features import vasari_features

import os, shutil, glob, json
import argparse
from os.path import join
import nibabel as nib
import numpy as np



def main(args: argparse.Namespace):
    t1_path    = glob.glob(os.path.join(args.subject_folder, "*_t1.nii.gz"))[0]
    t2_path    = glob.glob(os.path.join(args.subject_folder, "*_t2.nii.gz"))[0]
    t1ce_path  = glob.glob(os.path.join(args.subject_folder, "*_t1ce.nii.gz"))[0]
    flair_path = glob.glob(os.path.join(args.subject_folder, "*_flair.nii.gz"))[0]
    try:
        tumor_path = glob.glob(os.path.join(args.subject_folder, "*_seg_pred.nii.gz"))[0]
    except:
        tumor_path = glob.glob(os.path.join(args.subject_folder, "*_seg.nii.gz"))[0]

    # Determine unique values in tumor segmentation
    tumor_img = nib.load(tumor_path)
    tumor_data = tumor_img.get_fdata()
    unique_values = np.unique(tumor_data)

    # Only setting the default ET label. If not available, will be handled by the code itself.
    if 4 in unique_values:
        et_label = 4
    else:
        et_label = 3

    subject_name = Path(args.subject_folder).name
    tmp_dir = join(args.subject_folder, "tmp")

    # Intermediate file paths (all live inside tmp/)
    mni_in_subj       = join(tmp_dir, "MNI152_in_subject_space.nii.gz")
    mni_tfm           = join(tmp_dir, "MNI152_in_subject_space_transform.nii.gz")
    sub_in_mni        = join(tmp_dir, "subject_in_MNI152_space.nii.gz")
    sub_tfm           = join(tmp_dir, "subject_in_MNI152_space_transform.nii.gz")
    tum_in_mni        = join(tmp_dir, "tumor_seg_in_MNI152_space.nii.gz")
    patient_midline   = join(tmp_dir, "patient_midline.nii.gz")
    ideal_midline     = join(tmp_dir, "ideal_midline.nii.gz")
    midline_distances = join(tmp_dir, "midline_distances.nii.gz")
    anatseg           = mni_in_subj.replace(".nii.gz", "_synthseg.nii.gz")
    merged_seg        = mni_in_subj.replace(".nii.gz", "_merged_seg.nii.gz")

    # Output paths
    metadata_clinical_path    = join(args.subject_folder, f"{subject_name}_metadata_clinical.json")
    metadata_no_clinical_path = join(args.subject_folder, f"{subject_name}_metadata_no_clinical.json")
    metadata_final_path       = join(args.subject_folder, f"{subject_name}_metadata_final.json")

    # Determine whether to resume from step [4/5] or run from scratch.
    # Resume condition: tmp/ exists AND both checkpoint JSONs are present.
    resume_from_step4 = (
        os.path.isdir(tmp_dir)
        and bool(os.listdir(tmp_dir))
        and os.path.exists(metadata_no_clinical_path)
        and os.path.exists(metadata_clinical_path)
    )

    if resume_from_step4:
        logger.info(
            f"non-empty tmp/ and checkpoint JSONs found — skipping steps [0-3/5], "
            f"resuming from step [4/5]."
        )
        # Load steps [0-3] results from checkpoint
        with open(metadata_no_clinical_path, "r") as f:
            metadata = json.load(f)
        # Merge / overwrite with patient-level metadata (age, gender)
        with open(metadata_clinical_path, "r") as f:
            metadata.update(json.load(f))

    else:
        logger.info("No resume checkpoint found — running pipeline from step [0/5].")
        os.makedirs(tmp_dir, exist_ok=True)

        # Start from patient-level metadata (age, gender) if available
        metadata = {}
        if os.path.exists(metadata_clinical_path):
            with open(metadata_clinical_path, "r") as f:
                metadata.update(json.load(f))

        # ------------------------------------------------------------------
        # Step [0/5]: Registration
        # ------------------------------------------------------------------
        logger.info(f"** [0/5] Starting registration steps...")
        register.register_mni_to_subject(fixed=t1_path, moved=mni_in_subj, transform=mni_tfm, overwrite=args.overwrite)  # register MNI152 to subject space
        register.register_to_mni(moving=t1_path, moved=sub_in_mni, transform=sub_tfm, overwrite=args.overwrite)  # register T1 to MNI152 space
        register.register_midline_to_subject(moved=patient_midline, transform=mni_tfm, overwrite=args.overwrite)  # register MNI152 midline to subject space using mni_tfm
        register.apply_transform(moving=tumor_path, moved=tum_in_mni, transform=sub_tfm, is_seg=True)  # register tumor mask to MNI152 space using sub_tfm
        logger.info(f"* Finished registration steps!")

        # ------------------------------------------------------------------
        # Step [1/5]: Anatomical segmentation
        # ------------------------------------------------------------------
        # SynthSeg is unreliable on images with tumors, so we run it on the (healthy) MNI atlas registered to the subject space, then overlay the tumor mask.
        logger.info(f"** [1/5] Starting anatomical segmentation steps...")
        anat_segmentation.synthseg(input_path=mni_in_subj, output_path=anatseg)

        # Merge tumor, midline, and anatomical segmentation masks
        overlap_regions = anat_segmentation.merge_tumor_midline_and_anat_masks(
            synthseg_path=anatseg,
            tumor_path=tumor_path,
            midline_path=patient_midline,
            save_path=merged_seg,
            ncr_label=args.ncr_label,
            ed_label=args.ed_label,
            et_label=et_label,
            tumor_type=metadata.get("tumor-type", "glioma"),
            overwrite=args.overwrite,
        )
        metadata.update({"Anatomical Overlap Regions": overlap_regions})
        logger.info(f"* Finished segmentation steps! Merged mask can be found in {merged_seg}")

        # ------------------------------------------------------------------
        # Step [2/5]: Midline shift
        # ------------------------------------------------------------------
        logger.info(f"** [2/5] Starting midline shift processing...")
        midline_summary = midline_shift_3d(tumor = tumor_path,
                                           deformed_midline_path = patient_midline,
                                           ideal_midline_path = ideal_midline,
                                           midline_distances_path = midline_distances,
                                           anat_seg_path = anatseg,
                                           ncr_label = args.ncr_label, ed_label = args.ed_label, et_label = et_label, overwrite = args.overwrite)
        metadata.update(midline_summary)

        # ------------------------------------------------------------------
        # Step [3/5]: VASARI features
        # ------------------------------------------------------------------
        # vasari_summary = vasari_features(tumor=tumor_path, tumor_mni=tum_in_mni, metadata=metadata, merged=merged_seg, verbose=False, ncr_label=args.ncr_label, ed_label=args.ed_label, et_label=args.et_label)
        logger.info(f"** [3/5] Starting VASARI feature extraction steps...")
        extractor = ExtractVASARI(enhancing_label = et_label, nonenhancing_label = args.ncr_label, oedema_label = args.ed_label, verbose = False)
        vasari_summary = extractor(tumorseg_mni = tum_in_mni, tumorseg_ss = tumor_path, merged = merged_seg, metadata = metadata)
        metadata.update(vasari_summary)

        # Save checkpoint after steps [0-3] so that a future interrupted run
        # can resume from step [4/5] without redoing the expensive registration
        # and segmentation steps.
        metadata_no_clinical = {k: v for k, v in metadata.items() if k != "Clinical Report"}
        with open(metadata_no_clinical_path, "w") as f:
            json.dump(metadata_no_clinical, f, indent=2)
        logger.info(f"Saved steps [0-3] checkpoint to {metadata_no_clinical_path}")

    # ----------------------------------------------------------------------
    # Step [4/5]: Additional features  (always executed)
    # ----------------------------------------------------------------------
    logger.info(f"** [4/5] Starting Additional features extraction steps...")

    # Get laterality of tumor epicenter for use in T2-FLAIR mismatch feature extraction
    laterality = metadata.get("Side of Tumor Epicenter")

    # Get brain mask for use in additional feature extraction. 
    t1_img            = ants.image_read(mni_in_subj)
    brain_mask_pth    = join(args.subject_folder, "tmp", "brain_mask.nii.gz")
    brain_mask        = be(t1_img, modality = "t1", verbose=False)
    ants.image_write(brain_mask, brain_mask_pth)

    # Compute sphericity features
    sphericity = compute_sphericity(seg_path = tumor_path)
    metadata.update(sphericity)

    # Compute transition zone thickness features
    transition_zone_metrics = compute_transition_zone_thickness(
        seg_path   = tumor_path,
        flair_path = flair_path,   # used only by method_b; ignored by method_a
        t1ce_path  = t1ce_path,    # used only by method_b for T1CE-based thickness
        method     = "method_b",
    )
    metadata.update(transition_zone_metrics)

    # Compute morphometrics
    morphometrics = compute_vasari_style_morphometrics(tumorseg_ss_path   = tumor_path,
                                                       intensity_path     = flair_path,
                                                       brain_mask_path    = brain_mask_pth,
                                                       enhancing_label    = et_label,
                                                       nonenhancing_label = args.ncr_label,
                                                       oedema_label       = args.ed_label)
    metadata.update(morphometrics)

    # Compute T2-FLAIR mismatch features
    t2_flair_mismatch_extractor = ExtractT2FLAIRMismatch(enhancing_label = et_label, nonenhancing_label = args.ncr_label, oedema_label = args.ed_label)
    t2_flair_mismatch_metrics = t2_flair_mismatch_extractor(tumorseg_ss = tumor_path,
                                                            flair_path = flair_path, t2_path = t2_path,
                                                            laterality = laterality, merged_seg = merged_seg, brain_mask_path = None)
    metadata.update(t2_flair_mismatch_metrics)

    # ----------------------------------------------------------------------
    # Step [5/5]: LLM report generation (currently disabled)
    # ----------------------------------------------------------------------
    logger.info(f"** [5/5] Starting report generation with LLM ({args.llm})...")

    # json_save_key = f"BTReport Generated Report ({args.llm}, run_name={args.run_name})"
    # if json_save_key not in metadata:
    #     args.image_path = join(args.subject_folder, "tumor_maxslice.png") if args.image else None
    #     report = generate_llm_report(args.subject_folder.split("/")[-1], metadata, model=args.llm, image_path=args.image_path)
    #     logger.info(f"* Finished LLM report generation using extracted metadata!")
    #     metadata[json_save_key] = report
    # else:
    #     logger.info(f'Key {json_save_key} found in metadata, skipping LLM report')

    # Save final combined metadata
    with open(metadata_final_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved final metadata to {metadata_final_path}")

    if args.clear_tmp:  # Delete intermediate files after processing, useful for memory reduction but you lose interpretability of results.
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate a brain tumor report for one subject.")
    parser.add_argument("--subject_folder", type=str, help="Path to the subject folder containing the MRI data.")

    parser.add_argument("--clear_tmp", action="store_true", help="Delete the temporary directory after processing.")
    parser.add_argument("--overwrite", action="store_true", help="Redo this step, overwriting previous results.")
    parser.add_argument("--ncr_label", type=int, default=1)
    parser.add_argument("--ed_label", type=int, default=2)
    parser.add_argument("--et_label", type=int, default=4)
    parser.add_argument("--devices", type=str, default="0", help="String with cuda device IDs for use by synthseg and SynthMorph. E.g. '0,1' or '0'.")
    parser.add_argument("--run_name", type=str, default='v0')

    parser.add_argument(
        "--image",
        action="store_true",
        help="Indicator as to whther the model will use images for generation. Will look for tumor_maxslice.png in subject_folder",
    )
    parser.add_argument("--llm", type=str, default="gpt-oss:120b")
    parser.add_argument(
        "--tz_method",
        type=str,
        default="method_b",
        choices=["method_a", "method_b"],
        help=(
            "Transition zone thickness method. "
            "'method_a': distance-transform local thickness (purely geometric, recommended). "
            "'method_b': ray-casting along outward normals with FLAIR intensity band (research/exploratory)."
        ),
    )

    args = parser.parse_args()

    subject = os.path.basename(os.path.normpath(args.subject_folder))
    logger = get_logger(subject)
    # logger = get_logger("btreport.subject", subject=subject)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices)
    logger.info(f"Using GPUs: CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    main(args)
