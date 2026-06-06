# Setting up Meta Llama 4 Maverick (Vertex AI) for glioma classification

This guide covers everything needed to run
`classification/llm_pipeline/classify_glioma_llama4.py`, which classifies gliomas
with **Meta Llama 4 Maverick** on **Google Vertex AI Model-as-a-Service (MaaS)**.
It mirrors the other `classify_glioma_*.py` scripts (same prompt, schema, and
metrics) and is laid out like the synchronous `classify_glioma_groq.py`.

## Why this isn't the `google.genai` / Gemini path

Llama 4 is **not** available on the Gemini Developer API (the AI-Studio
`google.genai` client your `gemini25`/`gemini31` scripts use with a
`GEMINI_API_KEY`). On Google Cloud it is served through Vertex AI via an
**OpenAI-compatible** ChatCompletions endpoint. So the new script uses the
`openai` client (like the Groq script), not `google.genai`. Two consequences:

- **Billing is 100% Google Cloud / Vertex AI.** No OpenAI account or key is used; "OpenAI-compatible" refers only to the request/response *shape*.
- **Auth is a short-lived Google Cloud OAuth token**, not a static API key — minted from Application Default Credentials (ADC) and refreshed automatically by the script.

---

## Part 1 — One-time Google Cloud prerequisites

You need these done once per Google Cloud project. (Console links assume you're
signed into the Google account that will own the billing.)

1. **A Google Cloud project with billing enabled.**
   Create or pick a project, then confirm a billing account is attached:
   [Verify billing is enabled](https://console.cloud.google.com/billing).

2. **Enable the Vertex AI API** (`aiplatform.googleapis.com`) on that project:
   [Enable the API](https://console.cloud.google.com/apis/enableflow?apiid=aiplatform.googleapis.com).
   Requires the *Service Usage Admin* role (`roles/serviceusage.serviceUsageAdmin`).

3. **Accept Meta's licence (EULA) on the model card.** MaaS endpoints are blocked
   until you do this once. Open the Llama 4 Maverick card and click *Agree*:
   [Llama 4 Maverick 17B-128E model card](https://console.cloud.google.com/vertex-ai/publishers/meta/model-garden/llama-4-maverick-17b-128e-instruct-maas).

4. **IAM role for your user/service account.** The identity that runs the script
   needs to call Vertex AI prediction. The simplest grant is
   **Vertex AI User** (`roles/aiplatform.user`) on the project. (For a tighter
   grant, the script only needs `aiplatform.endpoints.predict`.)

5. **Region.** Llama 4 Maverick MaaS is served in **`us-east5`**. Use that as
   `VERTEX_LOCATION` unless Google has since added more regions
   ([region availability & quotas](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/partner-models/llama/use-llama)).
   Default quota is large (hundreds of thousands QPM), so the ~650-subject cohort
   is well within limits; request an increase via Cloud Quotas only if needed.

---

## Part 2 — Local machine setup

1. **Install the gcloud CLI** (if not already):
   <https://cloud.google.com/sdk/docs/install>.

2. **Log in and create Application Default Credentials.** This is what the script
   reads to mint access tokens:

   ```bash
   gcloud auth login                       # your Google identity
   gcloud config set project YOUR_PROJECT_ID
   gcloud auth application-default login    # writes ADC for libraries to use
   ```

   On a headless server or CI, use a service account instead:

   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
   ```

3. **Install the two extra Python dependencies** into the `BTReport` conda env
   (the `openai` SDK is already used by the Groq/GPT scripts; `google-auth` is
   the only genuinely new one):

   ```bash
   conda activate BTReport
   pip install --upgrade "openai>=1.30" "google-auth>=2.29"
   ```

4. **Add project + region to `.env`** (same file the other scripts read). No API
   key is needed for Llama — auth comes from ADC above:

   ```dotenv
   # Vertex AI (Llama 4 Maverick MaaS)
   VERTEX_PROJECT_ID = your-gcp-project-id
   VERTEX_LOCATION   = us-east5
   ```

5. **Smoke-test auth + endpoint** before the full run:

   ```bash
   python3 - <<'PY'
   import google.auth, google.auth.transport.requests
   from openai import OpenAI
   from dotenv import dotenv_values
   cfg = dotenv_values(".env")
   project, location = cfg["VERTEX_PROJECT_ID"], cfg.get("VERTEX_LOCATION", "us-east5")
   creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
   creds.refresh(google.auth.transport.requests.Request())
   client = OpenAI(
       base_url=f"https://{location}-aiplatform.googleapis.com/v1beta1/"
                f"projects/{project}/locations/{location}/endpoints/openapi",
       api_key=creds.token,
   )
   r = client.chat.completions.create(
       model="meta/llama-4-maverick-17b-128e-instruct-maas",
       messages=[{"role": "user", "content": "Reply with the single word: ok"}],
       max_tokens=5, temperature=0.0,
   )
   print("RESPONSE:", r.choices[0].message.content)
   PY
   ```

   A one-word reply confirms the project, EULA, IAM role, and region are all good.

---

## Part 3 — Running the classification

Run from the **repo root** (the script uses repo-relative paths for `data/` and
`.env`, exactly like the Groq script):

```bash
conda activate BTReport

# Full run (inference for every subject, then metrics) on the btreport JSONs
python3 classification/llm_pipeline/classify_glioma_llama4.py --json-type btreport

# Post-processed metadata variant
python3 classification/llm_pipeline/classify_glioma_llama4.py --json-type btreport_pp

# Recompute metrics only, from already-saved outputs (no API calls / no cost)
python3 classification/llm_pipeline/classify_glioma_llama4.py --json-type btreport --metrics-only
```

Outputs are written to `classification/llm_pipeline/llama4_outputs_<json_type>/`:

- `<subject>_classification_llama_4_maverick.json` — one structured result per subject
- `timings.csv` — per-subject latency log
- `metrics.json` — Accuracy / Sensitivity / Specificity / F1 per task, whole-cohort and per-dataset

Re-running **skips** subjects that already have an output file, so an interrupted
run resumes cheaply — delete a subject's JSON to force a re-classification.

---

## How the script handles the two Vertex-specific differences

**Token refresh.** `VertexTokenAuth` mints a Google Cloud access token from ADC
and rebuilds the `openai` client. Tokens expire (~1 h); since a full cohort run
can exceed that, the script checks validity before each request and re-mints as
needed (and also refreshes on a 401/403).

**Structured output + Llama Guard.** Vertex MaaS open models
[support strict structured output](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/maas/capabilities/structured-output),
so the scripts default to `RESPONSE_FORMAT_MODE = "json_schema"` — constraining
generation to `glioma_classification_schema.json` (the same `strict: True` pattern
as the `gpt55`/`groq` scripts). This is what prevents the key-drift/collapsed-block
problems; the content-based extractor and structure-retry loop remain only as a
safety net. Set the constant to `"json_object"` or `"none"` to relax it. **Llama
Guard moderation is ON by default** and can flag clinical text; if you see
`BadRequestError`s that look like content blocks, set `ENABLE_LLAMA_GUARD = False`
near the top of the script (passed through as `extra_body`).

---

## Troubleshooting

| Symptom | Likely cause / fix |
| --- | --- |
| `403 PERMISSION_DENIED` | EULA not accepted on the model card, or the identity lacks `roles/aiplatform.user`. |
| `401 UNAUTHENTICATED` | ADC missing/expired — re-run `gcloud auth application-default login`. The script also auto-refreshes once. |
| `404` on the endpoint | Wrong `VERTEX_LOCATION` (must be a region serving Llama 4 Maverick, currently `us-east5`) or wrong model id. |
| `VERTEX_PROJECT_ID not set` | Add it to `.env` or `export GOOGLE_CLOUD_PROJECT=...`. |
| `BadRequestError` mentioning safety | Llama Guard flagged the content — set `ENABLE_LLAMA_GUARD = False`. |
| `429` | Rate/quota limit — the script backs off automatically; request a quota increase if persistent. |

---

## Batch processing (50% cheaper) — `classify_glioma_llama4_batch.py`

For the full cohort, the **batch** path costs half as much as the synchronous
endpoint. It's the Vertex sibling of your `classify_glioma_gpt55.py`: Vertex
consumes the **same OpenAI batch JSONL schema** you already build, so the
`submit` → `poll` → `fetch` shape and `custom_id` matching are reused. Only the
transport differs — JSONL is staged in Cloud Storage and inference runs as a
Vertex `batchPredictionJob` instead of an OpenAI Files/Batch call.

**Extra prerequisites** (on top of Parts 1–2):

1. **A Cloud Storage bucket** for batch I/O staging, ideally in the same region:

   ```bash
   gcloud storage buckets create gs://YOUR_BUCKET --location=us-east5
   ```

2. **Storage permissions** for the identity running the script —
   **Storage Object Admin** (`roles/storage.objectAdmin`) on that bucket.

3. **One extra dependency** and **one extra `.env` line**:

   ```bash
   pip install --upgrade google-cloud-storage
   ```

   ```dotenv
   VERTEX_GCS_BUCKET = your-bucket-name
   ```

**Running it** (same sub-command pattern as your `gpt55`/`gemini31` batch scripts):

```bash
# Submit, poll to completion, then score — end to end
python3 classification/llm_pipeline/classify_glioma_llama4_batch.py run --json-type btreport

# Or run the phases separately (submit returns immediately; fetch can resume later)
python3 classification/llm_pipeline/classify_glioma_llama4_batch.py submit  --json-type btreport
python3 classification/llm_pipeline/classify_glioma_llama4_batch.py status                      # live job states
python3 classification/llm_pipeline/classify_glioma_llama4_batch.py fetch   --json-type btreport
python3 classification/llm_pipeline/classify_glioma_llama4_batch.py metrics --json-type btreport
```

Outputs land in `llama4_batch_outputs_<json_type>/` (one JSON per subject, plus
`batch_state.json`, `timings.csv`, `metrics.json`). Submitted jobs are tracked in
`batch_state.json`, so `fetch` can be picked up later from a different process —
the cohort is chunked (`CHUNK_SIZE = 200`) into one job per chunk.

**Batch-specific caveats:**

- Jobs are **all-or-nothing per chunk**, so the per-subject resume of the sync script doesn't apply the same way; re-`submit` with `--force` to redo subjects.
- **Llama Guard is not per-request toggleable in batch mode** (unlike the sync script's `ENABLE_LLAMA_GUARD`).
- The output JSONL line shape can vary; `_extract_from_line()` parses the common Vertex/OpenAI variants defensively and matches by `custom_id`.

---

## DeepSeek on the same setup — `classify_glioma_deepseek.py` (+ `_batch.py`)

DeepSeek is served on Vertex AI MaaS through the **same OpenAI-compatible endpoint
and the same ADC auth** as Llama 4, so everything in Parts 1–3 applies. The
DeepSeek scripts are siblings of the Llama ones (same prompt, schema, metrics,
content-based extraction, structure-retry loop, and batch machinery). Only three
things differ.

**1. Model id and region.** DeepSeek uses the `deepseek-ai/` publisher prefix, and
the serving region differs from Llama's `us-east5`:

| Model | `MODEL_ID` | Region |
| --- | --- | --- |
| DeepSeek R1-0528 (default) | `deepseek-ai/deepseek-r1-0528-maas` | `us-central1` |
| DeepSeek V3.1 | `deepseek-ai/deepseek-v3.1-maas` | `us-central1` |
| DeepSeek V3.2 | `deepseek-ai/deepseek-v3.2-maas` | supports the `global` endpoint |

**Note on structured output for R1.** Vertex MaaS open models support strict
`json_schema` structured output, so the scripts use `RESPONSE_FORMAT_MODE`:

- **Sync** (`classify_glioma_deepseek.py`) defaults to `"json_schema"`. R1 is a
  reasoning model and *might* reject it; if so, the request loop **auto-disables
  structured output for the rest of the run** and falls back to the prompt +
  `<think>`-stripping parser. So you can just run it — it self-detects.
- **Batch** (`classify_glioma_deepseek_batch.py`) defaults to `"none"`, because a
  batch job can't fall back mid-flight and a rejected `response_format` would fail
  all 200 rows. **Validate with the sync script first;** if R1 (or whichever model
  you point at) accepts `json_schema`, set `RESPONSE_FORMAT_MODE = "json_schema"`
  in the batch script too. For Llama and DeepSeek V3.x, `"json_schema"` is safe.

Confirm the exact current MaaS id on the
[DeepSeek model card](https://console.cloud.google.com/vertex-ai/publishers/deepseek-ai/model-garden/deepseek-v3.1-maas)
and **accept its licence once**, exactly like the Llama card. Set the region in
`.env`:

```dotenv
VERTEX_LOCATION = us-central1
```

The sync script handles the **`global`** endpoint automatically (set
`VERTEX_LOCATION = global` for models that support it — it builds the
non-region-prefixed host). Batch, however, is a **regional** service: the batch
script rejects `global` and you must use a region such as `us-central1`.

**2. Reasoning output.** DeepSeek V3.1/R1 are reasoning models and may prepend a
`<think>…</think>` block (or return the answer in `reasoning_content`). The scripts
strip the think block before JSON parsing and fall back to `reasoning_content`, and
`max_tokens` is raised to 8192 so reasoning doesn't crowd out the JSON answer.

**3. No Llama Guard.** DeepSeek MaaS has no per-request safety toggle, so the
DeepSeek scripts send no `extra_body`.

**Running it** (sync and batch mirror the Llama commands):

```bash
# Synchronous (self-heals collapsed outputs via the structure-retry loop)
python3 classification/llm_pipeline/classify_glioma_deepseek.py --json-type btreport
python3 classification/llm_pipeline/classify_glioma_deepseek.py --json-type btreport --metrics-only

# Batch (50% cheaper); --json-type is required for every phase
python3 classification/llm_pipeline/classify_glioma_deepseek_batch.py submit  --json-type btreport
python3 classification/llm_pipeline/classify_glioma_deepseek_batch.py status  --json-type btreport
python3 classification/llm_pipeline/classify_glioma_deepseek_batch.py fetch   --json-type btreport
python3 classification/llm_pipeline/classify_glioma_deepseek_batch.py metrics --json-type btreport
```

Outputs land in `deepseek_outputs_<json_type>/` (sync) and
`deepseek_batch_outputs_<json_type>/` (batch) with the `deepseek_v3_1` slug, so
they never collide with the Llama results.

## Aside: the lowest-friction alternative

Your `.env` already has Groq keys, and **Groq also hosts Llama 4 Maverick**
(`meta-llama/llama-4-maverick-17b-128e-instruct`). If you don't specifically need
Google Cloud billing/compliance, you can get Llama 4 numbers by copying
`classify_glioma_groq.py` and changing `MODEL_ID` — no GCP project, EULA, or ADC
setup required. The Vertex path in this guide is the right choice when billing
must go through Google Cloud (as discussed) or when you need Vertex's enterprise
data-governance guarantees.
