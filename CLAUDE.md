# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BTReport is a framework for brain tumor radiology report generation. It extracts quantitative neuroimaging features (VASARI features, midline shift, patient metadata) from MRI scans and segmentation masks, then uses local LLMs via Ollama to synthesize structured radiology reports. Published at MIDL 2025: arXiv:2602.16006.

## Environment Setup

```bash
conda env create -f environment.yml
conda activate BTReport
source docs/btreport_paths.sh   # sets required env vars: OLLAMA_SIF, OLLAMA_MODELS, SYNTHSEG_SIF, SYNTHMORPH_SIF
```

External dependencies (Singularity/Apptainer images) are downloaded from Zenodo (~11 GB):
- `ollama.sif`, `synthseg.sif`, `synthmorph_4.sif`

## Key Commands

**Start Ollama LLM server (requires GPU allocation):**
```bash
tmux new -d -s ollama3 "python3 -m btreport.ollama_server start-ollama --gpus 0,1"
python3 -m btreport.ollama_server pull-llm gpt-oss:120b   # or llama3:70b, deepseek-r1:70b
```

**Generate reports for all subjects:**
```bash
module load apptainer
conda activate BTReport
python3 -m btreport.run_all_reports --root_folder <path/to/root> --llm llama3:70b
```

**Generate report for a single subject:**
```bash
python3 -m btreport.generate_report --subject_folder <path/to/subject> --llm gpt-oss:120b
```

**Evaluate generated reports:**
```bash
python3 -m btreport.eval_json \
  --json <path/to/merged_reports_btreport.json> \
  --real_report_key "Clinical Report" \
  --synthetic_report_key "Predicted Report (gpt-oss:120b)" \
  --parse-real --parse-synthetic --devices 0,1
```

## Input Data Format

Each subject needs its own folder:
```
data/
├── subject_001/
│   ├── <id>-t1n.nii.gz     # T1 MRI scan
│   ├── <id>-seg.nii.gz     # Tumor segmentation (BraTS convention: NCR/ED/ET)
│   └── metadata.json       # Optional: ground-truth report under "Clinical Report" key
```

## Architecture

The pipeline runs four components sequentially per subject:

1. **`btreport/patient_metadata/`** — Extracts demographic/clinical info. Core: `merge_metadata.py`. Outputs survival curves and tabular features.

2. **`btreport/vasari_features/`** — Computes VASARI neuroimaging features (tumor size, location, enhancement, edema) in subject space using atlas masks. Core: `vasari_auto_v2.py`, `extract_vasari_features.py`. Uses SynthSeg for anatomical segmentation and atlas overlays for eloquent region identification.

3. **`btreport/midline_shift/`** — Estimates 3D midline shift by registering MNI152 atlas midline to patient space via SynthMorph, then computing voxel-wise distances. Core: `midline_shift3d.py`. Handles tumors that cross the anatomical midline.

4. **`btreport/llm_report_generation/`** — Formats extracted features as a JSON prompt, calls Ollama LLM via Apptainer container, and returns a structured FINDINGS section. Core: `ollama_report_gen_v2.py`.

**Supporting modules:**
- `btreport/additional_features/` — Sphericity, T2-FLAIR mismatch, transition zone thickness (`additional_features_3d.py`)
- `btreport/utils/` — Registration (`register.py`), anatomical segmentation (`anat_segmentation.py`), and bundled MNI152 atlas templates
- `btreport/evaluation/` — Metric computation (ROUGE, BERTScore, RadGraph, RATEscore) via `tbfact.py`, `make_leaderboard.py`

**Orchestration:** `generate_report.py` calls each component in sequence for a single subject. `run_all_reports.py` iterates over a directory and can split work across array jobs.

## Console Entry Points (after `pip install -e .`)

```
btreport-generate   →  btreport.generate_report:main
btreport-runall     →  btreport.run_all_reports:main
btreport-ollama     →  scripts.ollama_server:main
```

## Output

Batch runs write `root_folder/merged_reports_btreport.json` — a dict keyed by subject ID with entries for each LLM's predicted report and (if available) the ground-truth clinical report. This file is the input to `eval_json.py`.

## Pre-generated Dataset

`btreport_brats23.json` contains reports for BraTS'23 subjects generated with `gpt-oss:120b` and `llama3:70b`. Useful for evaluation without re-running inference.
