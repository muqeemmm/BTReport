
export PATH="${PATH}:/cvmfs/oasis.opensciencegrid.org/mis/apptainer/1.3.3/x86_64/bin"

# # Model directory
# export APPTAINERENV_OLLAMA_MODELS="/mmfs1/gscratch/scrubbed/juampablo/ollama_models"

# # Path to image
# IMAGE="/mmfs1/gscratch/scrubbed/juampablo/ollama.sif"

apptainer exec --nv \
   -B /pscratch:/pscratch \
   -B /cvmfs:/cvmfs \
   $IMAGE \
   ollama run symptoma/gpt-oss:120b


apptainer exec --nv \
   -B /mmfs1/:/mmfs1 \
   $IMAGE \
   ollama run symptoma/gpt-oss:120b


apptainer exec --nv \
    -B "$(dirname "$OLLAMA_MODELS")":"$(dirname "$OLLAMA_MODELS")" \
    "$OLLAMA_SIF" \
    ollama pull symptoma/medgemma3:27b
