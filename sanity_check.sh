#!/bin/bash
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=256g
#SBATCH -J "pretrain"
#SBATCH -p short
#SBATCH -t 1-00:00:00
#SBATCH --gres=gpu:2
#SBATCH -C "A100-80G"
#SBATCH --output=sb_sanity.log
#SBATCH --error=sb_sanity.err

set -euo pipefail

export SCRATCH_DIR="/scratch/${USER}"
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH_DIR}/torchinductor_cache"
export TRITON_CACHE_DIR="${SCRATCH_DIR}/triton_cache"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${SCRATCH_DIR}/checkpoints"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Target pass criteria @ val: emb_std in [0.95, 1.05], vocab_cos ~ 0.00, enc_adj_cos > 0.95
uv run --no-sync --extra ampere torchrun --standalone --nproc_per_node=2 train.py \
  --stage 1 \
  --data_dir "${SCRATCH_DIR}/climbmix_small_bins" \
  --ckpt_dir "${SCRATCH_DIR}/checkpoints" \
  --max_steps 35000
