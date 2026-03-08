import csv, sys, os, math, subprocess, datetime, argparse, logging
from pathlib import Path

logger = logging.getLogger(__name__)


MNI152 = Path(__file__).resolve().parent / "MNI152_T1_1mm_Brain-filled.nii.gz"
MIDLINE = Path(__file__).resolve().parent / "midline_plane_regressed.nii.gz"

SUBJECTS_DIR = os.path.realpath(os.environ.get("SUBJECTS_DIR"))
if SUBJECTS_DIR is None:
    logger.info("Environment variable SUBJECTS_DIR working dir is not set, defaulting to curent working dir")
    SUBJECTS_DIR = os.path.realpath(os.getcwd())
    os.environ["SUBJECTS_DIR"] = SUBJECTS_DIR

# raise ValueError(Path(__file__).resolve().parent.parent)

WRAPPER = os.environ.get("SYNTHMORPH_SIF")
if WRAPPER is None:
    raise RuntimeError("Environment variable SYNTHMORPH_SIF pointing to SynthMorph .sif is not set!")


def run_registration(moving, fixed, moved, transform, wrapper=WRAPPER, subjects_dir=SUBJECTS_DIR):
    """Run SynthMorph registration for one subject."""
    env = os.environ.copy()
    env["SUBJECTS_DIR"] = subjects_dir

    logger.info(f"========================================")
    logger.info(f"    SynthMorph Registration")
    logger.info(f"----------------------------------------")
    logger.info(f"  Moving    : {os.path.join(subjects_dir,moving)}")
    logger.info(f"  Fixed     : {os.path.join(subjects_dir,fixed)}")
    logger.info(f"  Output    : {os.path.join(subjects_dir,moved)}")
    logger.info(f"  Transform : {os.path.join(subjects_dir,transform)}")
    logger.info(f"========================================")

    cmd = [wrapper, "register", "-g", "-o", moved, "-t", transform, moving, fixed]
    subprocess.run(cmd, check=True, env=env)


def run_apply(transform, moving, moved, wrapper, subjects_dir, is_seg=False):
    """Run SynthMorph apply for one image or segmentation."""
    logger.info(f"========================================")
    logger.info(f"    SynthMorph Apply Registration")
    logger.info(f"----------------------------------------")
    logger.info(f"  Moving    : {os.path.join(subjects_dir,moving)}")
    logger.info(f"  Output    : {os.path.join(subjects_dir,moved)}")
    logger.info(f"  Transform : {os.path.join(subjects_dir,transform)}")
    logger.info(f"========================================")
    cmd = [wrapper, "apply"]
    if is_seg:
        cmd += ["-m", "nearest"]
    cmd += [transform, moving, moved]
    subprocess.run(cmd, check=True)


def register_to_mni(moving, moved, transform, subjects_dir=SUBJECTS_DIR, fixed=MNI152, overwrite=False):
    if os.path.exists(moved) and not overwrite:
        logger.info(f"Path {moved} exists, skipping registration..")
        return
    run_registration(moving=moving, fixed=fixed, moved=moved, transform=transform, wrapper=WRAPPER, subjects_dir=subjects_dir)


def register_mni_to_subject(fixed, moved, transform, subjects_dir=SUBJECTS_DIR, moving=MNI152, overwrite=False):
    if os.path.exists(moved) and not overwrite:
        logger.info(f"Path {moved} exists, skipping registration..")
        return
    run_registration(moving=moving, fixed=fixed, moved=moved, transform=transform, wrapper=WRAPPER, subjects_dir=subjects_dir)


def apply_transform(moving, moved, transform, is_seg=False, wrapper=WRAPPER, subjects_dir=SUBJECTS_DIR, overwrite=False):
    if os.path.exists(moved) and not overwrite:
        logger.info(f"Path {moved} exists, skipping registration..")
        return
    run_apply(transform=transform, moving=moving, moved=moved, wrapper=wrapper, subjects_dir=subjects_dir, is_seg=is_seg)


def register_midline_to_subject(moved, transform, moving=MIDLINE, is_seg=True, wrapper=WRAPPER, subjects_dir=SUBJECTS_DIR, overwrite=False):
    if os.path.exists(moved) and not overwrite:
        logger.info(f"Path {moved} exists, skipping registration..")
        return
    run_apply(transform=transform, moving=moving, moved=moved, wrapper=wrapper, subjects_dir=subjects_dir, is_seg=is_seg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser.add_argument("--moving", required=True)
    # parser.add_argument("--fixed", required=True)
    # parser.add_argument("--moved", required=True)
    # parser.add_argument("--transform", required=True)
    # parser.add_argument("--wrapper", default=WRAPPER, required=False)
    # parser.add_argument("--subjects_dir", default=SUBJECTS_DIR, required=False)

    parser.add_argument("--moving", required=True, type=os.path.realpath)
    parser.add_argument("--fixed", required=True, type=os.path.realpath)
    parser.add_argument("--moved", required=True, type=os.path.realpath)
    parser.add_argument("--transform", required=True, type=os.path.realpath)
    parser.add_argument("--wrapper", default=WRAPPER, type=os.path.realpath)
    parser.add_argument("--subjects_dir", default=SUBJECTS_DIR, type=os.path.realpath)


    parser.add_argument("--apply", action="store_true", required=False)
    parser.add_argument("--is_seg", action="store_true", required=False)
    

    args = parser.parse_args()

    if args.apply:
        run_apply(args.transform, args.moving, args.moved, args.wrapper, args.subjects_dir, is_seg=args.is_seg)
    else:
        run_registration(moving=args.moving, fixed=args.fixed, moved=args.moved, transform=args.transform, wrapper=args.wrapper, subjects_dir=args.subjects_dir)
