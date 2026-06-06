# Running MedGemma-27B & OpenBioLLM-70B on DeepInfra

A step-by-step guide for the glioma-classification pipeline. It covers deploying the
two open-weight medical models as **DeepInfra Custom LLMs** and running them with the
existing `classify_glioma_medgemma.py` and `classify_glioma_openbiollm.py` scripts.

> **Naming note.** "BioMedLLM-70B" in the request refers to **OpenBioLLM-70B**
> (`aaditya/Llama3-OpenBioLLM-70B`), the Llama-3 70B biomedical model used elsewhere
> in this pipeline. (PubMedBERT's "BioMedLM" is a different, much smaller model.)

---

## 0. The one thing to understand first

These two models are **not** in DeepInfra's shared serverless catalog, so you cannot
just call them per-token. You must run each as a **Custom LLM**: a *dedicated* GPU
deployment that DeepInfra spins up for you.

- **Billing is per GPU-hour while the deployment is up — not per token.** You pay for
  uptime regardless of how many requests you send.
- **DeepInfra does not currently support quantization.** Weights run at full
  precision (bf16), which sets the GPU count:
  - **OpenBioLLM-70B** (~140 GB in bf16) → **2 × 80 GB GPU**.
  - **MedGemma-27B** (~54 GB in bf16) → **1 × 80 GB GPU**.
- **Delete the deployment the moment you finish.** A forgotten 2-GPU deployment over a
  weekend (64 h) costs ~$256. Set a spending limit too.

---

## 1. Prerequisites

| # | What | Where |
|---|------|-------|
| 1 | A DeepInfra account (you sign in with **GitHub** — your GitHub username becomes the model-name prefix) | https://deepinfra.com |
| 2 | A DeepInfra **API key** | https://deepinfra.com/dash/api_keys |
| 3 | A **spending limit** set (safety net) | https://deepinfra.com/dash/billing |
| 4 | A **Hugging Face account + token** (MedGemma only — it is a gated model) | https://huggingface.co/settings/tokens |
| 5 | MedGemma license accepted on HF | https://huggingface.co/google/medgemma-27b-text-it |
| 6 | The pipeline scripts + Python env already set up in this repo | `classification/llm_pipeline/` |

For MedGemma, step 5 is mandatory: open the model page while logged in, accept the
Health AI Developer Foundations terms, then generate a **read** token in step 4. You
will paste that token into the deployment so DeepInfra can pull the gated weights.

---

## 2. Deploy each model

You can deploy from the **web UI** (easiest) or the **HTTP API**. Both produce a
deployment whose full id is `YOUR_GITHUB_USERNAME/<model-name>` — that id is what you
later put in `MODEL_ID`.

### 2A. Web UI (recommended)

Go to **Dashboard → New Deployment → Custom LLM**
(https://deepinfra.com/dash/deployments?new=custom-llm) and create one deployment per
model with these settings:

**OpenBioLLM-70B**

| Field | Value |
|-------|-------|
| `model_name` | `openbiollm-70b` |
| `weights` (HF repo) | `aaditya/Llama3-OpenBioLLM-70B` |
| `gpu` | `A100-80GB` (cheapest) or `H100-80GB` (≈2× faster) |
| `num_gpus` | **2** (required — no quantization, bf16 ≈ 140 GB) |
| `max_batch_size` | `8` (our script is sequential; small is fine) |
| `min_instances` | `1` while running the batch, `0` afterwards |
| `max_instances` | `1` |
| HF token | not needed (public repo) |

**MedGemma-27B**

| Field | Value |
|-------|-------|
| `model_name` | `medgemma-27b` |
| `weights` (HF repo) | `google/medgemma-27b-text-it` |
| `gpu` | `A100-80GB` or `H100-80GB` |
| `num_gpus` | **1** (bf16 ≈ 54 GB) |
| `max_batch_size` | `8` |
| `min_instances` | `1` while running, `0` afterwards |
| `max_instances` | `1` |
| **HF token** | **required** — paste your HF read token (gated repo) |

> If the UI has no HF-token field for a gated repo, use the HTTP API form below
> (which takes `hf.token`), or deploy an **ungated mirror** instead, e.g.
> `MODEL_ID` pointing at `Muhammadidrees/Medgamma27B` (community re-upload). For a
> publishable benchmark, prefer the official gated `google/medgemma-27b-text-it`.

### 2B. HTTP API (scriptable alternative)

```bash
export DEEPINFRA_TOKEN=di-xxxxxxxx          # your DeepInfra API key
export HF_TOKEN=hf_xxxxxxxx                  # your Hugging Face read token (MedGemma)

# --- OpenBioLLM-70B (public, 2 GPUs) ---
curl -X POST https://api.deepinfra.com/deploy/llm \
  -H "Authorization: Bearer $DEEPINFRA_TOKEN" -H 'Content-Type: application/json' \
  -d '{
    "model_name": "openbiollm-70b",
    "gpu": "A100-80GB",
    "num_gpus": 2,
    "max_batch_size": 8,
    "hf": { "repo": "aaditya/Llama3-OpenBioLLM-70B" },
    "settings": { "min_instances": 1, "max_instances": 1 }
  }'

# --- MedGemma-27B (gated, 1 GPU, needs HF token) ---
curl -X POST https://api.deepinfra.com/deploy/llm \
  -H "Authorization: Bearer $DEEPINFRA_TOKEN" -H 'Content-Type: application/json' \
  -d '{
    "model_name": "medgemma-27b",
    "gpu": "A100-80GB",
    "num_gpus": 1,
    "max_batch_size": 8,
    "hf": { "repo": "google/medgemma-27b-text-it", "token": "'"$HF_TOKEN"'" },
    "settings": { "min_instances": 1, "max_instances": 1 }
  }'
```

> **4-GPU account limit.** OpenBioLLM (2) + MedGemma (1) = 3 GPUs, so you can run both
> at once. Contact DeepInfra if you need more.

---

## 3. Confirm a deployment is running

```bash
# List deployments and their state
curl https://api.deepinfra.com/deploy/list -H "Authorization: Bearer $DEEPINFRA_TOKEN"
```

Or watch the **Dashboard → Deployments** page until status is `running`. First start-up
pulls the weights and can take several minutes (70B longer).

Quick smoke test once it's up (replace `YOUR_USERNAME`):

```bash
curl "https://api.deepinfra.com/v1/openai/chat/completions" \
  -H "Authorization: Bearer $DEEPINFRA_TOKEN" -H "Content-Type: application/json" \
  -d '{ "model": "YOUR_USERNAME/openbiollm-70b",
        "messages": [{"role":"user","content":"Reply with the single word: ready"}] }'
```

> Before it's fully running you can also address it as `"model": "deploy_id:YOUR_DEPLOY_ID"`.

---

## 4. Point the pipeline at DeepInfra

The scripts already support DeepInfra. You select it per run with three env values:

| Variable | Value |
|----------|-------|
| `PROVIDER` | `deepinfra` |
| `DEEPINFRA_API_KEY` | your DeepInfra API key |
| `MODEL_ID` | the deployment id, `YOUR_USERNAME/<model-name>` (REQUIRED for deepinfra) |

`MODEL_ID` is mandatory for DeepInfra — the scripts deliberately refuse to start
without it (the bare HF repo id would not resolve). Because the id differs per model,
set it on the command line for each run, or keep it in `.env` and run one model at a
time.

Add the key once to `classification/llm_pipeline/.env`:

```dotenv
DEEPINFRA_API_KEY=di-xxxxxxxx
```

---

## 5. Run the classification

Run each model with its own `MODEL_ID`. (`--json-type` selects the metadata variant;
run both `btreport` and `btreport_pp` if your experiment needs both.)

```bash
cd /Users/muqeemmmm/GitHub/BTReport_v2     # repo root

# OpenBioLLM-70B
PROVIDER=deepinfra MODEL_ID=YOUR_USERNAME/openbiollm-70b \
  python3 classification/llm_pipeline/classify_glioma_openbiollm.py --json-type btreport

# MedGemma-27B
PROVIDER=deepinfra MODEL_ID=YOUR_USERNAME/medgemma-27b \
  python3 classification/llm_pipeline/classify_glioma_medgemma.py --json-type btreport
```

Notes:

- **Resumable.** Completed cases write a JSON file and are skipped on re-run; failed/
  skipped cases leave no file, so just re-run the same command to finish them.
- **Strict JSON output** is requested via `response_format` (sent through `extra_body`);
  if a deployment rejects it, the script auto-falls back to prompt-only + tolerant
  parsing, so you still get scored results.
- **MedGemma's no-system-role quirk is already handled** — the script folds the system
  prompt into the user turn (Gemma's chat template rejects a separate system message).
- Outputs land in `medgemma_outputs_<json-type>/` and `openbiollm_outputs_<json-type>/`,
  each with `metrics.json` and `timings.csv`. Re-score anytime with `--metrics-only`.

Recommended: do a tiny pilot first to confirm connectivity and latency, e.g. let it run
~20 subjects, Ctrl-C, check the output files, then launch the full cohort.

---

## 6. Stop billing when you're done  ⚠️

This is the step people forget. As soon as the runs finish:

```bash
# Option A: scale to zero (keeps the config, stops most billing)
curl -X PUT https://api.deepinfra.com/deploy/DEPLOY_ID \
  -H "Authorization: Bearer $DEEPINFRA_TOKEN" -H 'Content-Type: application/json' \
  -d '{"settings": {"min_instances": 0, "max_instances": 0}}'

# Option B: delete entirely (recommended once the benchmark is complete)
curl -X DELETE https://api.deepinfra.com/deploy/DEPLOY_ID \
  -H "Authorization: Bearer $DEEPINFRA_TOKEN"
```

Or click the trash icon in **Dashboard → Deployments**. Billing is invoiced weekly,
so verify the deployments are gone before you walk away.

---

## 7. Cost estimate (recap)

DeepInfra 2026 dedicated rates: **A100-80GB $0.89/hr**, **H100-80GB $1.79/hr**, billed
per GPU while running. Conservative single-stream estimate for one pass over 646 cases:

| Model | GPUs | A100 $/hr | H100 $/hr | ~GPU-hrs (×1.2) | ~Cost (A100 / H100) |
|-------|------|-----------|-----------|-----------------|---------------------|
| OpenBioLLM-70B | 2 | $1.78 | $3.58 | 9.7 | ~$17 / ~$35 |
| MedGemma-27B | 1 | $0.89 | $1.79 | 5.4 | ~$5 / ~$10 |
| **Both, one pass** | | | | ~15 | **~$22 / ~$45** |

A100s minimise $/hr; H100s finish sooner for similar total cost. Running requests
concurrently (raising `max_batch_size` and parallelising the client) would cut
wall-clock and cost further. Both metadata variants roughly double the figure.

---

## 8. Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| Script exits: *"PROVIDER=deepinfra requires MODEL_ID…"* | Set `MODEL_ID` to your `username/model-name` (or `deploy_id:ID`). |
| `403 ... gated` during deploy | Accept the MedGemma license on HF and pass a valid HF token in the deploy config. |
| `404 model does not exist` at inference | Deployment not running yet, or wrong `MODEL_ID`. Check `/deploy/list`; use `deploy_id:ID` while it starts. |
| OOM / won't start (70B) | `num_gpus` too low — OpenBioLLM needs **2** GPUs (no quantization). |
| Slow first request | Cold start while weights load; set `min_instances: 1` for the duration of the batch. |
| Structured-output 400 | Harmless — the script auto-disables strict schema and continues with tolerant parsing. |

---

## References

- DeepInfra — Custom LLMs (deploy, call, scale, delete): https://docs.deepinfra.com/private-models/custom-llms
- DeepInfra — Structured Outputs: https://docs.deepinfra.com/chat/structured-outputs
- DeepInfra — Pricing (dedicated GPU rates): https://deepinfra.com/pricing
- DeepInfra — Deployments dashboard: https://deepinfra.com/dash/deployments
- OpenBioLLM-70B (HF): https://huggingface.co/aaditya/Llama3-OpenBioLLM-70B
- MedGemma-27B text (HF, gated): https://huggingface.co/google/medgemma-27b-text-it
