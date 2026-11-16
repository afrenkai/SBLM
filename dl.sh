#!/bin/bash 
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=32g
#SBATCH -J "dl climbmix"
#SBATCH -p short
#SBATCH -t 04:00:00
#SBATCH --output=dl_cm.log
#SBATCH --error=dl_cm.err
export SCRATCH_DIR="/scratch/${USER}"
export HF_HOME="${SCRATCH_DIR}/hf_cache"
mkdir -p "${HF_HOME}"
[ -f "${HOME}/.cache/huggingface/token" ] && cp -n "${HOME}/.cache/huggingface/token" "${HF_HOME}/token"
export MALLOC_ARENA_MAX=2
export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_DOWNLOAD_TIMEOUT=60
export TQDM_DISABLE=1
uv run python -u dl_ds.py --version small --out_dir "${SCRATCH_DIR}/climbmix_small_bins"


