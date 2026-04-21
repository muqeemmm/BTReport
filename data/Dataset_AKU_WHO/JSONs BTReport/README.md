# Brain Glioma Classification Prompts — Base vs. Enhanced (BTReport / AKU-WHO schema)

Two carefully curated LLM prompts for the joint classification of adult diffuse gliomas from **pre-operative brain-MRI structured metadata**, along with one shared structured-output schema. The pair is designed to replicate the **base vs. enhanced** evaluation pattern from Kim et al. (*European Radiology* 2026) and the **BTReport** short-vs-long prompt contrast from Heras Rivera et al. (Appendices C and D), applied to molecular and grade prediction rather than report generation.

Both prompts are tailored to the **exact 42-field metadata schema** emitted by the BTReport feature-extraction pipeline over the BraTS2021 cohort (646 cases in `JSONs BTReport/`, file pattern `BraTS2021_XXXXX_metadata_no_clinical.json`). Field names in each prompt are **verbatim** (including spaces, parentheses, and mixed case) so the raw JSON can be passed to the LLM without any key renaming.

## Tasks predicted jointly

- **IDH mutation status** — mutant vs. wildtype
- **1p/19q co-deletion status** — codeleted vs. non-codeleted
- **CNS WHO grade** — 2, 3, or 4

All three labels are emitted in a single JSON response with per-task reasoning, confidence, and supporting metadata fields, plus a final integrated diagnosis and a self-reported WHO-CNS5 consistency check.

## Files in this folder

| File | Purpose |
|---|---|
| `glioma_classification_prompt_base.md` | **Base** prompt. Role, task, label space, short anti-hallucination rules, the full 42-field BTReport schema, output schema. **No clinical knowledge injected.** Mirrors BTReport Appendix D. |
| `glioma_classification_prompt_enhanced.md` | **Enhanced** prompt. Everything in the base prompt, plus: expert role elaboration, full WHO CNS5 (2021) integrated-diagnosis rubric, imaging-to-molecular feature guide **scoped to the 42 available fields**, hard consistency rules, 7-step chain-of-thought reasoning protocol, cardinal-feature checklist, explicit unavailable-modality blacklist, 1p/19q dataset caveat, optional `<example_cases>` few-shot slot. Mirrors BTReport Appendix C + Kim et al.'s enhanced condition. |
| `glioma_classification_schema.json` | Shared **JSON Schema** for the output object. Drop-in for OpenAI `response_format`, Anthropic tool use, Gemini `responseSchema`, or local validation with `jsonschema`. |
| `glioma_prompt_design_record.docx` | Full design record of the prompt engineering process (literature review, design decisions, the two prompts verbatim, schema, and evaluation protocol). Suitable for defending the methodology to reviewers. |
| `README.md` | This file. |

## The 42-field BTReport input schema

Every case in `JSONs BTReport/` is a single JSON object with exactly 42 keys. Fields are grouped as follows:

| Group | Fields |
|---|---|
| Mass effect / midline shift | `n_slices_with_shift`, `mean_shift_mm`, `median_shift_mm`, `max_shift_mm`, `p95_shift_mm`, `midline_shift_present`, `level_max_shift` |
| Ventricular status | `Asymmetrical Ventricles`, `Effaced Ventricle`, `Enlarged Ventricles`, `Left Ventricle Volume (mm^3)`, `Right Ventricle Volume (mm^3)` |
| Tumor compartments (BraTS) | `ED Volume (mL)`, `ET Volume (mL)`, `NCR Volume (mL)`, `Total tumor volume (mL)`, `Proportion Enhancing`, `Proportion Necrosis`, `Proportion of Oedema` |
| Lesion geometry | `Lesion Sizes APxTVxCC (cm)`, `Number of lesions`, `Multifocal or Multicentric`, `Multiple satellites present` |
| Location | `Tumor Location`, `Region Proportions`, `Side of Tumor Epicenter`, `Anatomical Overlap Regions` |
| Per-compartment midline (ideal) | `volumes_ideal_midline`, `crosses_ideal_midline`, `primary_side_ideal_midline` |
| Per-compartment midline (patient) | `volumes_patient_midline`, `crosses_patient_midline`, `primary_side_patient_midline` |
| Summary crossings / invasion | `CET Crosses midline`, `Edema crosses midline`, `Cortical involvement`, `Deep WM invasion`, `Ependymal (ventricular) Invasion`, `Eloquent Brain Involvement` |
| Enhancement phenotype | `Enhancement Quality`, `Thickness of enhancing margin` |
| Summary string | `Text Report` |

Categorical vocabularies are fixed across the 646 cases (e.g., `Enhancement Quality` ∈ {`None`, `Mild`, `Marked`}; `Thickness of enhancing margin` ∈ {`<3mm`, `>3mm`, `Solid`}; `Multifocal or Multicentric` ∈ {`Solitary`, `Multifocal`}). The `Region Proportions` field uses 13 anatomical region tokens (`cortex`, `frontal_lobe`, `temporal`, `parietal`, `occipital`, `insula`, `ventricles`, `eloquent_grouped`, `thalamus`, `corpus_callosum`, `brainstem`, `internal_capsule`, `midline`). Both prompts document these exact vocabularies so the LLM never has to guess valid values.

## Modalities and markers that are NOT in this dataset

Both prompts explicitly instruct the LLM to avoid invoking the following, because they are absent from the 42-field schema (the files are `*_no_clinical.json` and contain only BraTS segmentation-derived features over T1n / T2w / T2-FLAIR / T1-Gd):

- **Patient age and sex** — strong priors for IDH; absent.
- **T2-FLAIR mismatch sign** — the single most specific imaging marker of IDH-mutant astrocytoma; not computed in the pipeline.
- **Intratumoral calcifications (SWI / GRE)** — the most specific imaging marker of 1p/19q co-deletion; SWI not performed.
- **ADC / diffusion restriction (DWI)** — useful for grade; absent.
- **rCBV / DSC perfusion** — useful for grade and IDH differentiation; absent.
- **MR spectroscopy / 2HG peak / choline-NAA ratio** — pathognomonic for IDH-mutant when available; absent.

The enhanced prompt flags this as a material limitation for the 1p/19q task (the strongest imaging markers — calcifications and mismatch — are both missing) and recommends capping 1p/19q confidences accordingly.

## Suggested evaluation design

Replicate the Kim et al. (2026) methodology, adapted to classification:

1. Identify the subset of the 646 BraTS2021 cases for which ground-truth IDH, 1p/19q, and WHO grade labels are available (via the AKU-WHO cohort or linked public releases such as UPenn-GBM, UCSF-PDGM, LGG-1p19q, or the original BraTS2021 clinical metadata).
2. For each case and each model under test (e.g., GPT-4o, o3 / o4-mini, Claude Sonnet / Opus, DeepSeek-R1, Qwen2.5-72B, Llama-3.1-70B, Gemini 2.x), run **both** prompts with identical decoding parameters (`temperature=0` where supported). The only variable should be the prompt text.
3. Parse each JSON output against `glioma_classification_schema.json` and compute:
   - Per-task accuracy, balanced accuracy, and macro-F1 (IDH, 1p/19q, WHO grade).
   - **Joint accuracy**: fraction of cases where all three predictions are correct.
   - **Consistency rate**: fraction of outputs with `consistency_check_passed == true`.
   - **Contradiction rate**: cases where the model's own `consistency_check_passed` is false.
   - **Calibration**: reliability diagrams of `confidence` vs. correctness per task.
4. Use paired statistical tests (e.g., McNemar, paired *t*-test on per-case error counts, or Benjamini–Hochberg-adjusted *p*-values as in Kim et al.) to assess the base → enhanced improvement per model, and the between-model differences within each condition.
5. Optionally ablate the enhanced prompt by removing one component at a time (e.g., remove only the imaging feature guide, or only the consistency rules) to attribute the gain.
6. Stratify performance by `Enhancement Quality` (None / Mild / Marked) and `Proportion Necrosis` bins to check whether LLM accuracy tracks the imaging phenotype in the expected direction.

## Reference WHO CNS5 (2021) label space

- **Oligodendroglioma, IDH-mutant and 1p/19q-codeleted** — grade 2 or 3 (grade 4 not permitted).
- **Astrocytoma, IDH-mutant** — grade 2, 3, or 4.
- **Glioblastoma, IDH-wildtype** — grade 4 by definition.

The enhanced prompt enforces these consistency rules explicitly; the base prompt does not.

## Citations

- **BTReport**: Heras Rivera et al. *BTReport: A Framework for Brain Tumor Radiology Report Generation.* arXiv:2602.16006 (Appendix C for the long prompt, Appendix D for the short prompt).
- **MRI protocoling LLMs**: Kim SH, Schramm S, Schmitzer L, et al. *Evaluating large language model-generated brain MRI protocols: performance of GPT-4o, o3-mini, DeepSeek-R1 and Qwen2.5-72B.* Eur Radiol (2026) 36:1644–1655. https://doi.org/10.1007/s00330-025-11989-0
- **WHO CNS5**: Louis DN, Perry A, Wesseling P, et al. *The 2021 WHO Classification of Tumors of the Central Nervous System: a summary.* Neuro-Oncology (2021) 23:1231–1251.
- **BraTS2021**: Baid et al. *The RSNA-ASNR-MICCAI BraTS 2021 Benchmark.* arXiv:2107.02314.
