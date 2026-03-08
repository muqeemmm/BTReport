#!/usr/bin/env bash

# BTReport path configuration
# Edit these paths to match your local installation.
# Source this file once per shell/session:
#   source docs/btreport_paths.sh



# Singularity / Apptainer images
export SYNTHMORPH_SIF="/media/ist/data/Muqeem/Projects/Brain_Project/Classification Code/BTReport/btreport/utils/synthmorph_4.sif"
export SYNTHSEG_SIF="/media/ist/data/Muqeem/Projects/Brain_Project/Classification Code/BTReport/btreport/utils/synthseg.sif"
# export OLLAMA_SIF=/pscratch/sd/j/jehr/ollama/ollama.sif

export SUBJECTS_DIR="/media/ist/data/Muqeem/Projects/Brain_Project/Classification Code/BTReport/data/Dataset_AKU_WHO/Astrocytoma_IDH-mutant" # Relative directory from which subject files are referenced inside containers. Usually I set this to the root of my scratch space.
export SF="/media/ist/data/Muqeem/Projects/Brain_Project/Classification Code/BTReport/data/dataset/BraTS2021_00045" # Absolute path to the same directory on the host system. This is used for bind-mounting into the containers.

# Ollama model storage (should be on large-capacity storage)
# export OLLAMA_MODELS=/pscratch/sd/j/jehr/ollama/ollama_models
# export OLLAMA_HOST='http://127.0.0.1:11434'
# unset http_proxy https_proxy all_proxy
# unset HTTP_PROXY HTTPS_PROXY ALL_PROXY

mkdir -p $OLLAMA_MODELS


export PATH=${PATH}:/cvmfs/oasis.opensciencegrid.org/mis/apptainer/1.3.3/x86_64/bin

if [ ! -x "$SYNTHMORPH_SIF" ]; then
    chmod +x "$SYNTHMORPH_SIF"
fi


for var in SYNTHMORPH_SIF SYNTHSEG_SIF; do
    if [ ! -f "${!var}" ]; then
        echo "ERROR: $var does not exist or is not a file: ${!var}" >&2
        return 1
    fi
done

for var in SUBJECTS_DIR; do
    if [ ! -d "${!var}" ]; then
        echo "ERROR: $var does not exist or is not a directory: ${!var}" >&2
        return 1
    fi
done


echo "BTReport paths validated:"
echo "  SYNTHMORPH_SIF : $SYNTHMORPH_SIF"
echo "  SYNTHSEG_SIF  : $SYNTHSEG_SIF"
# echo "  OLLAMA_SIF    : $OLLAMA_SIF"
# echo "  OLLAMA_MODELS : $OLLAMA_MODELS"
echo "  SUBJECTS_DIR  : $SUBJECTS_DIR"
echo "  SF            : $SF"
# echo "  OLLAMA_HOST   : $OLLAMA_HOST"
