"""
Glioma classification via Groq (openai/gpt-oss-120b).

For each subject metadata JSON in the JSONs folder (btreport or btreport_pp
variant, selected by --json-type), predicts IDH status, 1p/19q co-deletion,
and CNS WHO grade following WHO CNS5 (2021).

Output is constrained to `glioma_classification_schema.json` (same folder)
and the prompt mirrors `glioma_classification_prompt_base.pdf`.

After inference, computes Accuracy, Sensitivity, Specificity, and F1 per task
against ground-truth labels in cohort_merged.xlsx — reported for the whole
cohort and per dataset.

Confidence elicitation follows the Chain-of-Thought Confidence Elicitation
(CoT CE) strategy of Tian et al. (2023), as benchmarked in Ren et al.,
"Towards Reliable Medical LLMs" (arXiv:2601.15645): the model must produce
its clinical reasoning BEFORE committing to a prediction and a confidence
score, so the chain of thought informs the score rather than rationalising
a number already chosen. The schema field order (reasoning -> features ->
prediction -> confidence) enforces this generation order under strict
structured output.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import openpyxl
from dotenv import dotenv_values
from groq import Groq
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


# --- Prompt (transcribed from glioma_classification_prompt_base.pdf,
# ---          adapted for Chain-of-Thought Confidence Elicitation) -----------

SYSTEM_PROMPT = (
    "You are an expert neuroradiologist performing pre-operative molecular "
    "subtyping of adult-type diffuse glioma from structured MRI metadata, "
    "following the WHO CNS5 (2021) classification. For every task you reason "
    "step by step BEFORE stating any prediction or confidence score: first lay "
    "out the clinically grounded explanation, then commit to the prediction, "
    "then assign the confidence. Output MUST be a single valid JSON object that "
    "conforms exactly to the GliomaClassificationOutput schema. No markdown, no "
    "prose outside the JSON, no code fences -- JSON only."
)

USER_PROMPT_TEMPLATE = """# ROLE
You are an expert neuroradiologist.

# TASK
You are given structured pre-operative brain MRI metadata for a single patient
with a suspected **adult-type diffuse glioma**, defined according to the WHO
CNS5 molecular classification. Based on this metadata, predict the following:

  (a) IDH mutation status        -> {{"mutant", "wildtype"}}
  (b) 1p/19q co-deletion status  -> {{"codeleted", "non-codeleted"}}
  (c) CNS WHO grade              -> {{2, 3, 4}}

# CHAIN-OF-THOUGHT CONFIDENCE ELICITATION
For each of the three tasks you MUST work in the following order. Do not skip,
reorder, or shortcut these steps:

  1. **Reasoning first.** Write a concise, clinically grounded explanation that
     weighs the available metadata. Explicitly contrast the evidence that
     supports each candidate answer against the evidence that argues against
     it. Reach the prediction as the *conclusion* of this reasoning.
  2. **Supporting metadata fields.** List the exact field names (verbatim) that
     support the prediction you reasoned toward.
  3. **Contradicting features.** List the fields that argue against it.
  4. **Prediction.** State the prediction that follows from steps 1-3.
  5. **Confidence score** (between 0.0 and 1.0). Derive the score *from* the
     reasoning above -- it must reflect the balance of supporting vs.
     contradicting evidence you already articulated, NOT a number chosen before
     reasoning. Assign high confidence only when key features are well
     supported and few contradict; assign low confidence when evidence is
     thin, conflicting, or absent.

The JSON schema lists `reasoning`, `supporting_features` and
`contradicting_features` before `prediction` and `confidence` for exactly this
reason: generate them in that order.

# INPUT CONSTRAINTS
- Input consists **only of structured MRI-derived metadata** (no images).
- Do not assume access to data beyond what is provided.
- Treat null values as **"not assessed"** (do not infer).

# AVAILABLE MRI SEQUENCES
T1w, T2w, T2-FLAIR, T1-Gd.
**Not available:** DWI/ADC, perfusion (rCBV/DSC), MRS (2HG), SWI/GRE.

# RULES
1. Use ONLY the metadata provided; no hallucinated features or sequences.
2. Every claim in `reasoning` must be traceable to a field named verbatim in
   `supporting_features`.
3. Do NOT invoke diffusion, perfusion, spectroscopy, calcifications, or SWI.
4. If evidence is insufficient: give a best estimate, set confidence in
   0.50-0.60, and state the uncertainty in reasoning.
5. Confidence must be >= 0.50 for binary tasks (otherwise flip the prediction).
6. The confidence score must be the conclusion of the reasoning, not its
   premise -- never write the reasoning to justify a pre-chosen score.
7. Output must be **valid JSON only** (no markdown, no commentary).

# INPUT
Patient subject_id: {subject_id}
MRI metadata:
{metadata_json}
"""


# --- Config ------------------------------------------------------------------

MODEL_ID    = "openai/gpt-oss-120b"
HERE        = Path(__file__).parent
SCHEMA_PATH = HERE / "glioma_classification_schema.json"
ENV_PATH    = Path("/Users/muqeemmmm/GitHub/Report-Generation/classification/.env")
JSONS_DIR   = Path("/Users/muqeemmmm/GitHub/BTReport_v2/data/Dataset_AKU_WHO/JSONs")
COHORT_XLSX = Path("/Users/muqeemmmm/GitHub/BTReport_v2/data/dataset_table/cohort_merged.xlsx")


# --- Ground-truth helpers ----------------------------------------------------

def load_ground_truth(xlsx_path: Path) -> dict:
    """
    Returns a dict keyed by subject_id with:
      {"idh": str|None, "codeletion": str|None, "grade": int|None, "dataset": str}

    GT label normalisation:
      IDH:    "mutated"               -> "mutant"
              "wild type"             -> "wildtype"
      1p/19q: "co-deleted" /
              "rel. co-deleted"       -> "codeleted"
              "intact"                -> "non-codeleted"
              "x"                     -> None  (not assessed)
      Grade:  int as-is
    """
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    # headers: Center ID(0), Subject ID(1), Dataset(2), Hospital(3),
    #          BraTS2021(4), Gender(5), Age(6), IDH(7), 1p/19q(8), Grade(9)

    def _idh(v):
        if v == "mutated":   return "mutant"
        if v == "wild type": return "wildtype"
        return None

    def _codeletion(v):
        if v in ("co-deleted", "rel. co-deleted"): return "codeleted"
        if v == "intact":                          return "non-codeleted"
        return None  # "x" or unknown

    gt: dict = {}
    for row in rows[1:]:
        center_id = row[0]
        brats_id  = row[4] if (row[4] and row[4] != "x") else None
        entry = {
            "idh":        _idh(row[7]),
            "codeletion": _codeletion(row[8]),
            "grade":      row[9] if isinstance(row[9], int) else None,
            "dataset":    row[2],
        }
        if center_id:
            gt[center_id] = entry
        if brats_id:
            gt[brats_id] = entry

    return gt


# --- Metrics -----------------------------------------------------------------

def _binary_metrics(y_true, y_pred, pos_label) -> dict:
    """Accuracy, Sensitivity, Specificity, F1 for a binary task."""
    if not y_true:
        return {"n": 0, "accuracy": None, "sensitivity": None,
                "specificity": None, "f1": None}
    all_labels = sorted(set(y_true) | set(y_pred))
    neg_candidates = [l for l in all_labels if l != pos_label]
    # Fallback: derive neg_label from the known label space when only one class present
    neg_label = neg_candidates[0] if neg_candidates else (
        "wildtype" if pos_label == "mutant" else "non-codeleted"
    )
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, pos_label=pos_label, average="binary",
                   zero_division=0)
    cm  = confusion_matrix(y_true, y_pred, labels=[pos_label, neg_label])
    TP, FN = cm[0, 0], cm[0, 1]
    FP, TN = cm[1, 0], cm[1, 1]
    sens = TP / (TP + FN) if (TP + FN) > 0 else None
    spec = TN / (TN + FP) if (TN + FP) > 0 else None
    return {
        "n":           len(y_true),
        "accuracy":    round(acc, 4),
        "sensitivity": round(sens, 4) if sens is not None else None,
        "specificity": round(spec, 4) if spec is not None else None,
        "f1":          round(f1, 4),
    }


def _grade_metrics(y_true, y_pred) -> dict:
    """Macro-averaged Accuracy, Sensitivity, Specificity, F1 for grade (2/3/4)."""
    if not y_true:
        return {"n": 0, "accuracy": None, "sensitivity": None,
                "specificity": None, "f1": None}
    classes = [2, 3, 4]
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, labels=classes, average="macro",
                   zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    sens_list, spec_list = [], []
    for i in range(len(classes)):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - TP - FN - FP
        sens_list.append(TP / (TP + FN) if (TP + FN) > 0 else 0.0)
        spec_list.append(TN / (TN + FP) if (TN + FP) > 0 else 0.0)
    return {
        "n":           len(y_true),
        "accuracy":    round(acc, 4),
        "sensitivity": round(sum(sens_list) / len(sens_list), 4),
        "specificity": round(sum(spec_list) / len(spec_list), 4),
        "f1":          round(f1, 4),
    }


def compute_metrics(predictions: dict, gt: dict) -> dict:
    """
    Returns metrics per task broken down by whole cohort and per dataset.
      {
        "whole_cohort": {"idh": {...}, "codeletion": {...}, "grade": {...}},
        "<dataset>":    {"idh": {...}, ...},
        ...
      }
    """
    data = defaultdict(lambda: {
        "idh_t": [], "idh_p": [],
        "cod_t": [], "cod_p": [],
        "grd_t": [], "grd_p": [],
    })

    for sid, pred in predictions.items():
        row = gt.get(sid)
        if row is None:
            continue
        ds = row["dataset"] or "Unknown"

        for bucket in ("whole_cohort", ds):
            d = data[bucket]
            if row["idh"] is not None and pred.get("idh") is not None:
                d["idh_t"].append(row["idh"])
                d["idh_p"].append(pred["idh"])
            if row["codeletion"] is not None and pred.get("codeletion") is not None:
                d["cod_t"].append(row["codeletion"])
                d["cod_p"].append(pred["codeletion"])
            if row["grade"] is not None and pred.get("grade") is not None:
                d["grd_t"].append(row["grade"])
                d["grd_p"].append(pred["grade"])

    results = {}
    for bucket, d in data.items():
        results[bucket] = {
            "idh":        _binary_metrics(d["idh_t"], d["idh_p"], "mutant"),
            "codeletion": _binary_metrics(d["cod_t"], d["cod_p"], "codeleted"),
            "grade":      _grade_metrics(d["grd_t"], d["grd_p"]),
        }
    return results


def print_metrics(metrics: dict) -> None:
    col_w = 22
    header = (f"{'Subset':<{col_w}} {'Task':<12} {'N':>5}  "
              f"{'Acc':>6}  {'Sens':>6}  {'Spec':>6}  {'F1':>6}")
    sep = "=" * len(header)
    print(f"\n{sep}\nCLASSIFICATION METRICS\n{sep}")
    print(header)
    print("-" * len(header))

    task_labels = {"idh": "IDH", "codeletion": "1p/19q", "grade": "Grade"}
    buckets = ["whole_cohort"] + sorted(k for k in metrics if k != "whole_cohort")

    for bucket in buckets:
        for task_key, label in task_labels.items():
            m = metrics[bucket][task_key]
            def _fmt(v):
                return f"{v:.4f}" if v is not None else "  N/A"
            print(f"{bucket:<{col_w}} {label:<12} {m['n']:>5}  "
                  f"{_fmt(m['accuracy']):>6}  {_fmt(m['sensitivity']):>6}  "
                  f"{_fmt(m['specificity']):>6}  {_fmt(m['f1']):>6}")
        print()


# --- Run ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify gliomas via Groq on btreport or btreport_pp JSONs."
    )
    parser.add_argument(
        "--json-type",
        choices=["btreport", "btreport_pp"],
        required=True,
        help="Which metadata JSON variant to use (btreport or btreport_pp).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args      = parse_args()
    json_type = args.json_type  # "btreport" or "btreport_pp"

    OUTPUT_DIR = HERE / f"groq_outputs_{json_type}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    schema = json.load(open(SCHEMA_PATH))
    config = dotenv_values(ENV_PATH)
    client = Groq(api_key=config["GROQ_API_KEY"])
    gt     = load_ground_truth(COHORT_XLSX)

    subjects = sorted(d.name for d in JSONS_DIR.iterdir() if d.is_dir())

    for i, subject_id in enumerate(subjects, 1):
        print(f"Processing {subject_id} ({i}/{len(subjects)})")

        metadata_path = JSONS_DIR / subject_id / f"{subject_id}_metadata_{json_type}.json"
        if not metadata_path.exists():
            print(f"  skip: missing {metadata_path}")
            continue

        out_path = OUTPUT_DIR / f"{subject_id}_classification_gpt_oss_120b.json"
        if out_path.exists():
            print(f"  skip: {out_path} already exists")
            continue

        metadata    = json.load(open(metadata_path))
        user_prompt = USER_PROMPT_TEMPLATE.format(
            subject_id=subject_id,
            metadata_json=json.dumps(metadata, indent=2, default=str),
        )

        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.0,
            seed=42,
            max_completion_tokens=4096,
            reasoning_effort="medium",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "GliomaClassificationOutput",
                    "schema": schema,
                    "strict": True,
                },
            },
        )

        result = json.loads(response.choices[0].message.content)
        result["subject_id"] = subject_id

        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  saved -> {out_path}")

    # --- Metrics -----------------------------------------------------------------
    print("\nCollecting predictions for metric computation...")
    predictions: dict = {}
    for out_file in sorted(OUTPUT_DIR.glob("*_classification_gpt_oss_120b.json")):
        result = json.load(open(out_file))
        sid = result.get("subject_id",
                         out_file.stem.replace("_classification_gpt_oss_120b", ""))
        predictions[sid] = {
            "idh":        result.get("idh", {}).get("prediction"),
            "codeletion": result.get("one_p_nineteen_q", {}).get("prediction"),
            "grade":      result.get("who_grade", {}).get("prediction"),
        }

    metrics = compute_metrics(predictions, gt)
    print_metrics(metrics)

    metrics_path = OUTPUT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved -> {metrics_path}")
