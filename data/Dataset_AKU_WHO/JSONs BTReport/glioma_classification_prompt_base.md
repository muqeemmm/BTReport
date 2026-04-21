# Curated Prompt — **BASE** variant (BTReport schema, AKU/WHO dataset)

**Task:** Joint prediction of **IDH mutation status**, **1p/19q co-deletion status**, and **CNS WHO Grade (2 / 3 / 4)** from the structured BTReport-style MRI metadata used in the `JSONs BTReport` dataset (646 BraTS2021 cases, 42-field schema per case, pre-operative T1n / T2w / T2-FLAIR / T1-Gd only).

**Variant:** *Base* — minimal, knowledge-free prompt that mirrors **BTReport Appendix D (short version)** and the **"base" condition** from Kim et al. (*European Radiology* 2026). It contains only: a role, the task statement, the label space, a short anti-hallucination rule set, the exact metadata schema the dataset uses, and the output schema. **No WHO CNS5 rubric, no imaging-to-molecular feature guide, no consistency rules, no chain-of-thought scaffold.** The model must rely entirely on its pre-trained clinical knowledge.

**Companion file:** `glioma_classification_prompt_enhanced.md` — the *enhanced* variant with full WHO CNS5 (2021) rubric and an imaging-feature guide restricted to the same 42 available fields. Run both prompts against identical `{metadata_json}` inputs with the shared `glioma_classification_schema.json` output schema to measure the contribution of injected domain knowledge.

---

## 1. Prompt (paste into `system` or `user` role as your API supports)

```
# ROLE
You are a neuroradiologist.

# TASK
Given the structured pre-operative brain-MRI metadata for a single adult patient with a
suspected diffuse glioma, predict all three of the following labels:

  (a) IDH mutation status           -> {"mutant", "wildtype"}
  (b) 1p/19q co-deletion status     -> {"codeleted", "non-codeleted"}
  (c) CNS WHO grade                 -> {2, 3, 4}

For each task, provide a prediction, a confidence between 0.0 and 1.0, a brief reasoning,
and a list of the metadata fields (use their EXACT names, including spaces, parentheses,
and capitalisation) that supported your decision.

# AVAILABLE MRI SEQUENCES
The metadata is derived from pre-operative multiparametric BraTS-style MRI:
T1n (native T1), T2w, T2-FLAIR, and T1-Gd (post-contrast). No diffusion, perfusion, or
spectroscopy data are available. No patient demographics are available.

# INPUT METADATA SCHEMA (exact field names from the BTReport JSON)
The input is a JSON object with 42 fields. Field types are str, int, float, list, dict,
or null. Null means "not assessed" for that case. Allowed categorical values are shown.

  # Mass effect / midline shift
  "n_slices_with_shift"                  : int
  "mean_shift_mm"                        : float  (mm, signed)
  "median_shift_mm"                      : float  (mm, signed)
  "max_shift_mm"                         : float  (mm, signed; magnitude in mm)
  "p95_shift_mm"                         : float  (mm)
  "midline_shift_present"                : "No" | "Yes" | "Minimal"
  "level_max_shift"                      : "septum pellucidum" | "falx cerebri above" |
                                           "falx cerebri below" | "third ventricle" |
                                           "fourth ventricle"

  # Ventricular status
  "Asymmetrical Ventricles"              : "Present" | "Absent"
  "Effaced Ventricle"                    : "Left" | "Right" | null
  "Enlarged Ventricles"                  : "Left" | "Right" | "Left and Right" | null
  "Left Ventricle Volume (mm^3)"         : float
  "Right Ventricle Volume (mm^3)"        : float

  # Tumor compartments (BraTS labels: ED=edema, ET=enhancing tumor, NCR=necrotic core)
  "ED Volume (mL)"                       : float
  "ET Volume (mL)"                       : float
  "NCR Volume (mL)"                      : float
  "Total tumor volume (mL)"              : float
  "Proportion Enhancing"                 : float  (percent of total tumor volume)
  "Proportion Necrosis"                  : float  (percent)
  "Proportion of Oedema"                 : float  (percent)

  # Lesion geometry
  "Lesion Sizes APxTVxCC (cm)"           : list of [AP, TV, CC] triplets in cm
  "Number of lesions"                    : int
  "Multifocal or Multicentric"           : "Solitary" | "Multifocal"
  "Multiple satellites present"          : "Present" | "Absent"

  # Location
  "Tumor Location"                       : str   (free-text combination of region tokens)
  "Region Proportions"                   : list of [region_name, proportion] pairs
  "Side of Tumor Epicenter"              : "Left" | "Right" | "Midline" | "Bilateral" | "Un-defined"
  "Anatomical Overlap Regions"           : list[str]  (subcortical / ventricular structures)

  # Per-compartment midline behaviour (ideal vs. patient midline)
  "volumes_ideal_midline"                : { "ncr": {"left": num, "right": num},
                                             "ed":  {...}, "et": {...}, "ncr_et": {...} }
  "crosses_ideal_midline"                : { "ncr": bool, "ed": bool, "et": bool, "ncr_et": bool }
  "primary_side_ideal_midline"           : { "ncr": "left"|"right", ...  }
  "volumes_patient_midline"              : same shape as volumes_ideal_midline
  "crosses_patient_midline"              : same shape as crosses_ideal_midline
  "primary_side_patient_midline"         : same shape as primary_side_ideal_midline

  # Summary crossings and invasion
  "CET Crosses midline"                  : "True" | "False"   (string, not bool)
  "Edema crosses midline"                : "True" | "False"
  "Cortical involvement"                 : "Present" | "Absent"
  "Deep WM invasion"                     : "Present" | "Absent"
  "Ependymal (ventricular) Invasion"     : "Present" | "Absent"
  "Eloquent Brain Involvement"           : "No involvement" | "Motor" | "Vision" |
                                           "Speech motor" | "Motor and Vision" |
                                           "Speech motor and Motor" |
                                           "Speech motor and Vision" |
                                           "Speech motor, Motor and Vision"

  # Enhancement phenotype
  "Enhancement Quality"                  : "None" | "Mild" | "Marked"
  "Thickness of enhancing margin"        : "<3mm" | ">3mm" | "Solid"

  # Auto-generated summary string
  "Text Report"                          : str   (short natural-language summary)

# RULES
  1. Use ONLY the metadata provided. Do NOT hallucinate features, measurements, or
     sequences that are not present in the input. If a field is missing or null, treat it
     as "not assessed" and do not assume a value.
  2. Every claim in the `reasoning` field must be traceable to a specific metadata field,
     and that field must be named verbatim in `supporting_features` (e.g.,
     "Proportion Necrosis", "Enhancement Quality", "Thickness of enhancing margin").
  3. Do NOT invoke diffusion (ADC, DWI), perfusion (rCBV, DSC), spectroscopy (2HG, MRS),
     the T2-FLAIR mismatch sign, calcifications, SWI, or patient demographics (age, sex)
     -- none of these are present in this dataset.
  4. If evidence is insufficient for a confident call, output your best estimate but set
     `confidence` < 0.6 and explain the limitation in `reasoning`.
  5. Do not output any prose outside the JSON object. No preamble, no markdown fences, no
     trailing commentary. The response MUST be parseable by `json.loads`.

# INPUT
Patient subject_id: {subject_id}
MRI metadata:
{metadata_json}

# OUTPUT (return ONLY this JSON object -- no other text)
{
  "subject_id": "{subject_id}",
  "idh": {
    "prediction": "mutant" | "wildtype",
    "confidence": 0.0,
    "reasoning": "Brief justification referencing named metadata fields.",
    "supporting_features": ["Proportion Necrosis", "Enhancement Quality", ...],
    "contradicting_features": ["<exact field name>", ...]
  },
  "one_p_nineteen_q": {
    "prediction": "codeleted" | "non-codeleted",
    "confidence": 0.0,
    "reasoning": "Brief justification referencing named metadata fields.",
    "supporting_features": [...],
    "contradicting_features": [...]
  },
  "who_grade": {
    "prediction": 2 | 3 | 4,
    "confidence": 0.0,
    "reasoning": "Brief justification referencing named metadata fields.",
    "supporting_features": [...],
    "contradicting_features": [...]
  },
  "integrated_diagnosis": "Short integrated diagnosis string combining the three predictions.",
  "consistency_check_passed": true | false,
  "overall_comment": "Any residual uncertainty, conflicts, or limitations (e.g., absent diffusion/perfusion/SWI data) worth flagging."
}
```

---

## 2. What this prompt does *not* include (by design)

The following elements are **deliberately omitted** so that any performance delta between this prompt and the enhanced variant is attributable to injected domain knowledge, not to output format or task framing:

| Element | In base? | In enhanced? |
|---|:---:|:---:|
| Expert role elaboration (board certification, WHO CNS5 fluency) | ✗ | ✓ |
| WHO CNS5 (2021) integrated-diagnosis rubric (three families, allowed grade ranges) | ✗ | ✓ |
| Imaging-to-molecular feature guide mapped onto the 42 available fields | ✗ | ✓ |
| Hard consistency rules (1p/19q codel ⇒ IDH-mutant and grade ≤ 3; IDH-wildtype ⇒ 1p/19q non-codel; grade 4 forbidden in oligodendroglioma) | ✗ | ✓ |
| 7-step chain-of-thought reasoning protocol | ✗ | ✓ |
| Cardinal-feature checklist | ✗ | ✓ |
| Optional few-shot `<example_cases>` slot | ✗ | ✓ |
| Explicit schema of the 42 metadata fields | ✓ | ✓ |
| Metadata-only evidence rule | ✓ | ✓ |
| Explicit "no diffusion / perfusion / spectroscopy / demographics / mismatch-sign / calcifications" caveat | ✓ | ✓ |
| JSON output schema | ✓ | ✓ |

Kept in both: the sole-evidence rule, the traceability-to-fields rule (using exact field names), the "absent modalities" caveat, the confidence-calibration instruction, the JSON-only-output rule, and the output schema. These are the minimum scaffolding required to (a) prevent hallucination and (b) make the two variants directly comparable for evaluation.

---

## 3. Notes on the dataset

The BTReport JSONs in `JSONs BTReport/` contain **pre-operative BraTS segmentations + radiologist-style feature extraction**, not raw DICOM or any molecularly-informative sequences. Specifically absent from this schema (and therefore absent from both prompts):

- **T2-FLAIR mismatch sign** — the single most specific imaging marker of IDH-mutant astrocytoma; its absence is a material limitation for the IDH task.
- **Intratumoral calcifications (SWI/GRE)** — the most specific imaging marker of 1p/19q co-deletion; its absence is a material limitation for the 1p/19q task.
- **ADC / diffusion restriction** — useful for grade; absent.
- **rCBV / DSC perfusion** — useful for grade and IDH; absent.
- **MR spectroscopy / 2HG peak** — pathognomonic for IDH-mutant when available; absent.
- **Patient age and sex** — strong priors for IDH; absent (the files are `*_no_clinical.json`).

Both prompts have been rewritten to rely only on features that are actually present. The enhanced prompt's imaging-feature guide has been re-centered on enhancement phenotype, necrosis fraction, edema behaviour, invasion patterns (cortical / deep WM / ependymal), multifocality, mass effect, location priors (via `Region Proportions` / `Tumor Location`), and eloquent-brain involvement — the features that the schema actually provides.

---

## 4. Usage notes

**Temperature.** `temperature=0` for deterministic output. For reasoning-only models where temperature is not exposed (e.g., `o3-mini`, `o4-mini`), use the default.

**Structured output mode.** Bind `glioma_classification_schema.json` via OpenAI `response_format`, Anthropic tool use, or Gemini `responseSchema` to guarantee a parseable output.

**Pairing with the enhanced variant.** Run both prompts on identical `{metadata_json}` inputs with identical model, decoding parameters, and random seed (where supported). The only variable should be the prompt text. Report the accuracy delta as the contribution of domain-knowledge injection — this is the analogue of the Kim et al. base-vs-enhanced comparison applied to glioma classification over the BTReport / BraTS2021 / AKU-WHO dataset.

**Reasoning models.** Even without the 7-step reasoning protocol, reasoning LLMs (o-series, DeepSeek-R1, Claude extended thinking, Qwen-QwQ) will still produce internal chain-of-thought before emitting the JSON. The base prompt does not prevent this — it simply does not scaffold it. The delta between base and enhanced on reasoning vs. non-reasoning models is itself an interesting finding to report.
