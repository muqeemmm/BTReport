"""
Glioma classification via Groq (openai/gpt-oss-120b).

For each subject metadata JSON in `INPUT_FOLDER`, predicts IDH status,
1p/19q co-deletion, and CNS WHO grade following WHO CNS5 (2021).
Output is constrained to `glioma_classification_schema.json` (same folder)
and the prompt mirrors `glioma_classification_prompt_base.pdf`.

Confidence elicitation follows the Chain-of-Thought Confidence Elicitation
(CoT CE) strategy of Tian et al. (2023), as benchmarked in Ren et al.,
"Towards Reliable Medical LLMs" (arXiv:2601.15645): the model must produce
its clinical reasoning BEFORE committing to a prediction and a confidence
score, so the chain of thought informs the score rather than rationalising
a number already chosen. The schema field order (reasoning -> features ->
prediction -> confidence) enforces this generation order under strict
structured output.
"""

import json
from datetime import date
from pathlib import Path

from dotenv import dotenv_values
from groq import Groq


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
INPUT_FOLDER = Path("/Users/muqeemmmm/GitHub/BTReport_v2/data/dummy_dataset")
OUTPUT_DIR   = HERE / f"groq_outputs_{date.today().isoformat()}"


# --- Run ---------------------------------------------------------------------

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents = True, exist_ok = True)

    schema = json.load(open(SCHEMA_PATH))
    config = dotenv_values(ENV_PATH)
    client = Groq(api_key = config["GROQ_API_KEY"])

    # Discover subjects (BTReport layout: <id>/<id>_detailed.json)
    subjects = sorted(d.name for d in INPUT_FOLDER.iterdir() if d.is_dir())

    for i, subject_id in enumerate(subjects, 1):
        print(f"Processing {subject_id} ({i}/{len(subjects)})")

        metadata_path = INPUT_FOLDER / subject_id / f"{subject_id}_metadata_no_clinical.json"
        if not metadata_path.exists():
            print(f"  skip: missing {metadata_path}")
            continue

        out_path = OUTPUT_DIR / f"{subject_id}_classification_gpt_oss_120b.json"
        if out_path.exists():
            print(f"  skip: {out_path} already exists")
            continue

        metadata    = json.load(open(metadata_path))
        user_prompt = USER_PROMPT_TEMPLATE.format(
            subject_id = subject_id,
            metadata_json = json.dumps(metadata, indent=2, default=str),
        )

        response = client.chat.completions.create(
            model = MODEL_ID,
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature = 0.0,
            seed = 42,
            max_completion_tokens = 4096,
            reasoning_effort = "medium",
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "GliomaClassificationOutput",
                    "schema": schema,
                    "strict": True,
                },
            },
        )

        result = json.loads(response.choices[0].message.content)
        result["subject_id"] = subject_id  # enforce consistency

        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  saved -> {out_path}")
