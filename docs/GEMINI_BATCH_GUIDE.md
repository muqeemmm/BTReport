# Gemini Batch Classification — Implementation Guide

This document explains how `classify_glioma_gemini25.py` works and how to run it.
It is the Gemini counterpart of `classify_glioma_groq.py`: same task, same
Chain-of-Thought Confidence Elicitation (CoT CE) schema, same metrics — but
inference goes through Gemini's **asynchronous Batch API** to get the documented
**50% cost discount** versus synchronous calls.

---

## 1. Why batch changes the architecture

The Groq script is synchronous: one request → one response, in a per-subject
loop, with key rotation and inline retries. Gemini's Batch API is **fire-and-forget
asynchronous**. You hand Google a bundle of requests, it processes them within a
target of 24 hours (often far less), and you collect results afterward. That
forces a three-phase design instead of one loop:

| Phase | What happens | Groq equivalent |
|-------|--------------|-----------------|
| **submit** | Bundle all subjects into batch job(s), send to Google, persist job names | (none — Groq calls inline) |
| **fetch** | Poll job state until terminal, download results, write one JSON per subject | the body of the Groq loop |
| **metrics** | Score saved outputs against ground truth | identical to Groq |

Because submit and fetch can be minutes-to-hours apart, the job names are saved
to `gemini25_outputs_<json_type>/batch_state.json` so you can submit now and fetch
later from a separate process.

Key trade-off: batch is **half price and higher throughput**, but **not
interactive**. Use it for the whole-cohort evaluation runs, not for debugging a
single subject (use a normal `generate_content` call for that).

---

## 2. Two ways to submit, and why this script uses inline + chunking

The Batch API accepts requests in two forms:

1. **Inline requests** — a Python list of request dicts passed straight to
   `client.batches.create(src=[...])`. Capped at ~20 MB total per job. Results
   come back **in input order** as `batch_job.dest.inlined_responses`.
2. **JSONL input file** — a file uploaded via the File API where each line is
   `{"key": "...", "request": {...}}`. Up to 2 GB. Results come back as a
   downloadable JSONL keyed by your `key`.

This script uses **inline requests, split into chunks of 40 subjects per job**.
Reasoning:

- **Field order is load-bearing.** CoT CE only works if the model writes
  `reasoning` → `supporting_features` → `contradicting_features` *before*
  `prediction` and `confidence`. The google-genai SDK converts a **Pydantic
  class** passed as `response_schema` into a Gemini schema with an explicit
  `property_ordering` array (verified — see §4). Hand-writing a JSON schema in a
  JSONL file does **not** guarantee key order unless you set `propertyOrdering`
  yourself.
- **No File API round-trip** (upload, mime-type handling, download, JSONL parse)
  to get wrong.
- **Chunking keeps each payload under the 20 MB inline limit** and lets a job
  fail/expire without taking the whole cohort with it.

When you'd switch to the JSONL-file path instead: cohorts in the thousands, or
per-subject payloads large enough that even 40 fit poorly under 20 MB. The doc's
§6 sketches that variant.

---

## 3. What is identical to the Groq script (and why)

To keep the two models directly comparable, these are copied verbatim:

- **`SYSTEM_PROMPT` and `USER_PROMPT_TEMPLATE`** — same clinical instructions and
  CoT-CE step ordering.
- **Ground-truth and ID loading** (`load_ground_truth`, `load_id_map`) and label
  normalisation (`mutated→mutant`, `co-deleted→codeleted`, etc.).
- **Metrics** (`_binary_metrics`, `_grade_metrics`, `compute_metrics`,
  `print_metrics`) — accuracy/sensitivity/specificity/F1, per-cohort and
  per-dataset, joined on the folder name stored back into each result as
  `subject_id`.
- **Output file naming** — `<folder>_classification_<model_slug>.json`, e.g.
  `EGD-0008_classification_gemini_2_5_pro.json`. The metrics reader you already
  have works unchanged once you point it at `gemini25_outputs_<json_type>/`.

What changed: the inference layer, the schema representation (JSON Schema →
Pydantic), and the phased CLI.

---

## 4. Schema: JSON Schema → Pydantic, and the ordering proof

`glioma_classification_schema.json` is mirrored as Pydantic models
(`TaskIDH`, `TaskCodeletion`, `TaskGrade`, `GliomaClassificationOutput`). Pydantic
preserves field **declaration order**, and the SDK turns that into Gemini's
`property_ordering`. A check on `google-genai 2.7.0` produced:

```json
"idh": {
  "type": "OBJECT",
  "property_ordering": [
    "reasoning", "supporting_features", "contradicting_features",
    "prediction", "confidence"
  ],
  "properties": { "...": "..." }
}
```

That `property_ordering` is exactly what forces the model to reason before it
commits — the whole point of CoT CE.

Two Gemini-specific schema notes:

- **Enums are for STRING types.** `Literal["mutant","wildtype"]` and
  `Literal["codeleted","non-codeleted"]` map cleanly to string enums. WHO grade,
  however, is left as a plain `int` (constrained to 2/3/4 in the prompt and
  validated downstream) because integer enums are not reliably supported. The
  ground-truth grades are ints, so the metrics join is unaffected.
- **`additionalProperties: false` / `$schema` / `minLength`** from the original
  JSON Schema have no direct equivalent in Gemini's schema subset; they are
  simply dropped by the SDK. Constraints that matter for the task are restated in
  the prompt rules.

---

## 5. Gemini gotchas worth knowing

- **Auth.** The client reads `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) from `.env`,
  falling back to the environment. Unlike the Groq script there is **no key
  rotation** — batch jobs run under one project with much higher batch rate
  limits, so a pool of keys isn't needed.
- **"Thinking" tokens.** Gemini 2.5 Pro reasons internally by default; those
  thinking tokens are billed (still at the 50% batch rate). Our schema's
  `reasoning` field is separate visible output. If cost matters more than depth,
  `gemini-2.5-flash` is dramatically cheaper — the script takes `--model`.
- **Determinism.** `temperature=0.0` and `seed=42` are set per request, matching
  the Groq run. Gemini is still less strictly reproducible than a seeded Groq
  call; expect minor wording drift in `reasoning`.
- **`max_output_tokens=8192`.** CoT CE produces three reasoning blocks plus an
  integrated diagnosis; 8 K leaves headroom. If you see truncated/invalid JSON,
  raise it.
- **Job states.** Terminal states are `SUCCEEDED`, `FAILED`, `CANCELLED`,
  `EXPIRED`. A job that sits pending/running beyond **48 h** expires with no
  results — resubmit or shrink the chunk. `failedRequestCount` in `batchStats`
  flags partial failures; per-subject errors surface as `inline_response.error`
  and are logged and skipped (that subject simply has no output file, so a re-run
  of `submit` picks it up again).
- **Idempotency.** `batches.create` is **not** idempotent — calling `submit`
  twice creates two jobs. The script guards against duplicate *work* by skipping
  subjects that already have an output file (override with `--force`), but it
  won't dedupe jobs you submitted manually.

---

## 6. Optional: the JSONL-file variant

If you outgrow inline (thousands of subjects), switch the submit phase to a file:

```python
# build one JSONL line per subject; `key` lets you map results back by name
with open("batch_in.jsonl", "w") as f:
    for s in pending:
        f.write(json.dumps({
            "key": s["folder_name"],
            "request": {
                "contents": [{"parts": [{"text": s["prompt"]}], "role": "user"}],
                "generation_config": {
                    "temperature": 0.0,
                    "response_mime_type": "application/json",
                    "response_schema": GEMINI_SCHEMA_DICT,  # must include propertyOrdering
                },
                "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            },
        }) + "\n")

uploaded = client.files.upload(file="batch_in.jsonl",
    config=types.UploadFileConfig(display_name="glioma", mime_type="jsonl"))
job = client.batches.create(model=MODEL, src=uploaded.name,
    config={"display_name": "glioma-file"})
```

Results download as JSONL; parse each line's `response` (or `error`) and key it
back by `key`. The catch is `GEMINI_SCHEMA_DICT`: you must serialise the schema
yourself and set `propertyOrdering` by hand to preserve CoT CE — which is the
exact pitfall the inline+Pydantic path avoids. You can generate that dict from
the Pydantic model with the SDK transformer
(`google.genai._transformers.t_schema(None, GliomaClassificationOutput)`),
dump it, and embed it.

---

## 7. Running it

Prereqs:

```bash
conda activate BTReport
pip install google-genai            # SDK used here (tested on 2.7.0)
echo 'GEMINI_API_KEY=your_key_here' >> .env
```

Run from the same working directory as the Groq script (paths
`data/Dataset_AKU_WHO/JSONs` and `data/dataset_table/cohort_merged.xlsx` are
relative to cwd).

One-shot (submit → poll → fetch → score). Blocks until the job finishes:

```bash
python3 classification/llm_pipeline/classify_glioma_gemini25.py run \
    --json-type btreport --model gemini-2.5-pro
```

Decoupled (recommended for large cohorts — submit, walk away, collect later):

```bash
# 1. fire off the jobs (returns immediately)
python3 .../classify_glioma_gemini25.py submit  --json-type btreport

# 2. check progress any time
python3 .../classify_glioma_gemini25.py status  --json-type btreport

# 3. once SUCCEEDED, download + save + score
python3 .../classify_glioma_gemini25.py fetch   --json-type btreport
python3 .../classify_glioma_gemini25.py metrics --json-type btreport
```

Outputs land in `classification/llm_pipeline/gemini25_outputs_btreport/`:
per-subject `*_classification_gemini_2_5_pro.json`, `batch_state.json`,
`timings.csv`, and `metrics.json` — mirroring the Groq output folder so any
downstream leaderboard code keeps working.

Flags: `--model` (default `gemini-2.5-pro`; pass `gemini-2.5-flash` to slash
cost), `--force` (re-run subjects that already have outputs), `--json-type`
(`btreport` or `btreport_pp`, required for everything except `status`).
