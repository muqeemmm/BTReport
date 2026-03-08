import os, glob, shutil, json, argparse, re, logging
from .evaluation.utils import parse_radiology_report
from .evaluation.eval_tbfact import TBFactEval
from tqdm import tqdm
from .utils.log import get_logger
from RadEval import RadEval, compare_systems
from .utils import load_json
from pprint import pprint

import fcntl
import tempfile


def eval_json(
    args,
    include_keys=[
        "BRAIN",
        "MASS EFFECT & VENTRICLES",
    ],
):
    with open(args.json, "r") as f:
        data = json.load(f)

    all_subjects = sorted(list(data.keys()))

    split_subjects = all_subjects[args.split_no :: args.num_splits]

    print(
        f"Total subjects: {len(all_subjects)} | "
        f"Split {args.split_no}/{args.num_splits} -> "
        f"{len(split_subjects)} subjects"
    )

    split_data = {k:v for k, v in data.items() if k in set(split_subjects)}

    for idx, (subject_id, metadata) in tqdm(enumerate(split_data.items()), total=len(split_data), colour="red"):
        print("*=" * 80)
        print("=*" * 80)
        print(f"[{idx}/{len(split_data)}] -- Evaluating: {subject_id}")
        print("=*" * 80)
        print("*=" * 80)

        save_path = args.json.replace(".json", f"__run_name_{args.run_name}.json")
        save_path = save_path.replace(".json", "_eval_results.json")
        if args.do_details:
            save_path = save_path.replace(".json", "_details.json")

        existing_lookup = {}
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                existing_results = json.load(f)
            for d in existing_results:
                for sid, metrics in d.items():
                    existing_lookup[str(sid)] = metrics

        # # check that keys exist
        # assert args.real_report_key in metadata, f"Missing real report key '{args.real_report_key}' in {args.json}"
        # assert args.synthetic_report_key in metadata,  f"Missing synthetic report key '{args.synthetic_report_key}' in {args.json}"
        # check that keys exist
        if args.real_report_key not in metadata:
            logger.warning(
                f"Missing real report key '{args.real_report_key}' in {args.json}; skipping."
            )
            continue

        if args.synthetic_report_key not in metadata:
            logger.warning(
                f"Missing synthetic report key '{args.synthetic_report_key}' in {args.json}; skipping."
            )
            continue

        # # separate both real and generated reports into sections (e.g. FINDINGS, BRAIN, MASS EFFECT & VENTRICLES, etc.)
        real_reports = metadata[args.real_report_key]
        synthetic_reports = metadata[args.synthetic_report_key]
        refs = [real_reports]
        hyps = [synthetic_reports]
        if args.parse_real:
            real_reports = parse_radiology_report(real_reports)
            refs = ["\n\n".join(f"{k}:\n{real_reports[k]}" for k in include_keys if k in real_reports)]

        if args.parse_synthetic:
            synthetic_reports = parse_radiology_report(synthetic_reports)
            hyps = ["\n\n".join(f"{k}:\n{synthetic_reports[k]}" for k in include_keys if k in synthetic_reports)]

        pprint(refs)
        print("-*-*" * 30)
        print("-*-*" * 30)
        pprint(hyps)

        refs_missing = any(s.strip() == "" for s in refs)
        hyps_missing = any(s.strip() == "" for s in hyps)

        if refs_missing and hyps_missing:
            logger.info(f"Both reference and generated reports do not have {include_keys} sections")
            continue
            # return False, f"Both reference and generated reports do not have {include_keys} sections"
        elif refs_missing:
            logger.info(f"Reference report does not have {include_keys} sections")
            continue
            # return False, f"Reference report does not have {include_keys} sections"
        elif hyps_missing:
            logger.info(f"Generated report does not have {include_keys} sections")
            continue
            # return False, f"Generated report does not have {include_keys} sections"

        rouge_evaluator = RadEval(do_rouge=True, do_details=args.do_details)
        bleu_evaluator = RadEval(do_bleu=True, do_details=args.do_details)
        bertscore_evaluator = RadEval(do_bertscore=True, do_details=args.do_details)
        srr_bert_evaluator = RadEval(do_radgraph=True, do_details=args.do_details)
        ratescore_evaluator = RadEval(do_ratescore=True, do_details=args.do_details)
        tbf_evaluator = TBFactEval(llm=args.llm, do_details=args.do_details)

        if not args.do_details:
            eval_functions = {
                "bleu": lambda hyps, refs: bleu_evaluator(refs, hyps)["bleu"],
                "rouge1": lambda hyps, refs: rouge_evaluator(refs, hyps)["rouge1"],
                "rouge2": lambda hyps, refs: rouge_evaluator(refs, hyps)["rouge2"],
                "rougeL": lambda hyps, refs: rouge_evaluator(refs, hyps)["rougeL"],
                "bertscore": lambda hyps, refs: bertscore_evaluator(refs, hyps)["bertscore"],
                "ratescore": lambda hyps, refs: ratescore_evaluator(refs, hyps)["ratescore"],
                f"tbfact ({args.llm})": lambda hyps, refs: tbf_evaluator(refs=refs, hyps=hyps),
            }
        else:
            eval_functions = {
                "bleu": lambda hyps, refs: bleu_evaluator(refs, hyps),
                "rouge1": lambda hyps, refs: rouge_evaluator(refs, hyps),
                "rouge2": lambda hyps, refs: rouge_evaluator(refs, hyps),
                "rougeL": lambda hyps, refs: rouge_evaluator(refs, hyps),
                "bertscore": lambda hyps, refs: bertscore_evaluator(refs, hyps),
                # 'radgraph': lambda hyps, refs: srr_bert_evaluator(refs, hyps),
                "ratescore": lambda hyps, refs: ratescore_evaluator(refs, hyps),
                f"tbfact ({args.llm})": lambda hyps, refs: tbf_evaluator(refs=refs, hyps=hyps),
            }

        # results = {str(subject_id): {k: fn(hyps, refs) for k, fn in eval_functions.items()}}
        subject_key = str(subject_id)
        prev = existing_lookup.get(subject_key, {})  

        current = {}

        for metric_name, fn in eval_functions.items():
            if args.skip_processed and metric_name in prev and prev[metric_name] is not None:
                logger.info(f"Skipping {metric_name} for {subject_key} (already computed)")
                current[metric_name] = prev[metric_name]
            else:
                logger.info(f" [{idx}/{len(data)}]-- Computing {metric_name} for {subject_key}")
                current[metric_name] = fn(hyps, refs)

        results = {subject_key: current}

        append_to_json_list(save_path, results)

        logger.info(f"Saved evaluation results to {save_path}")
        # return True, 'Success'


# def append_to_json_list(save_path, new_item):
#     """
#     new_item must be a dict with exactly one key, e.g.:
#         {"10064774423088": {...metrics...}}

#     If the ID already exists in the JSON list:
#         merge the nested dicts (existing[id].update(new_data))
#     Else:
#         append new_item to the list
#     """

#     # Load existing data
#     if os.path.exists(save_path):
#         with open(save_path, "r") as f:
#             data = json.load(f)
#         if not isinstance(data, list):
#             raise ValueError("JSON file must contain a list of dicts.")
#     else:
#         data = []

#     # Extract the patient ID from the new item
#     if len(new_item) != 1:
#         raise ValueError("new_item must contain exactly one top-level key.")
#     new_id = list(new_item.keys())[0]
#     new_value = new_item[new_id]

#     # Check whether ID already exists and merge
#     merged = False
#     for entry in data:
#         if new_id in entry:
#             # Merge dictionaries (update existing metrics with new ones)
#             entry[new_id].update(new_value)
#             merged = True
#             break

#     # If not merged, append as new entry
#     if not merged:
#         data.append(new_item)

#     # Save updated file
#     with open(save_path, "w") as f:
#         json.dump(data, f, indent=2)

#     return data



def append_to_json_list(save_path, new_item):
    """
    new_item must be a dict with exactly one key, e.g.:
        {"10064774423088": {...metrics...}}

    If the ID already exists in the JSON list:
        merge the nested dicts (existing[id].update(new_data))
    Else:
        append new_item to the list

    This implementation is safe under concurrent writers.
    """

    if len(new_item) != 1:
        raise ValueError("new_item must contain exactly one top-level key.")

    new_id = next(iter(new_item))
    new_value = new_item[new_id]

    # Ensure parent directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Open file (create if missing)
    with open(save_path, "a+") as f:
        # Acquire exclusive lock (blocks until available)
        fcntl.flock(f, fcntl.LOCK_EX)

        try:
            # Move to beginning and load existing data
            f.seek(0)
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError("JSON file must contain a list of dicts.")
            except json.JSONDecodeError:
                # Empty or partially written file
                data = []

            # Merge or append
            merged = False
            for entry in data:
                if new_id in entry:
                    entry[new_id].update(new_value)
                    merged = True
                    break

            if not merged:
                data.append(new_item)

            # Write atomically via temp file
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=os.path.dirname(save_path),
                prefix=".tmp_",
                suffix=".json",
            )

            try:
                with os.fdopen(tmp_fd, "w") as tmp:
                    json.dump(data, tmp, indent=2)
                    tmp.flush()
                    os.fsync(tmp.fileno())

                # Atomic replace
                os.replace(tmp_path, save_path)

            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        finally:
            # Always release the lock
            fcntl.flock(f, fcntl.LOCK_UN)

    return data



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation across all subjects.")
    parser.add_argument(
        "--json",
        type=str,
        required=True,
        help="Path to .json file, expected structure {'subject0_id':{'Clinical Report':.., 'Predicted Report':...}, ...}",
    )
    parser.add_argument("--real_report_key", type=str, default="Clinical Report")
    parser.add_argument("--synthetic_report_key", type=str, default="Predicted Report")
    parser.add_argument("--devices", type=str, default="0,1,2,3")
    parser.add_argument("--do_details", action="store_true")
    parser.add_argument("--llm", type=str, default="deepseek-r1:70b")
    parser.add_argument("--run_name", type=str, default="v0")

    parser.add_argument("--skip_processed", action="store_true")

    parser.add_argument(
        "--parse-real",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Parse the real clinical reports, use --no-parse-real for False",
    )
    parser.add_argument(
        "--parse-synthetic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Parse the synthetic generated reports, use --no-parse-synthetic for False",
    )

    parser.add_argument("--num_splits", type=int, default=1)
    parser.add_argument("--split_no", type=int, default=0)

    args = parser.parse_args()


    if not (0 <= args.split_no < args.num_splits):
        raise ValueError("--split_no must satisfy 0 <= split_no < num_splits")


    logger = get_logger(args.json)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices)
    logger.info(f"Using GPUs: CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    eval_json(args)
