#!/bin/bash -l
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --mem=128g
#SBATCH -J pretrain
#SBATCH -p short,long
#SBATCH -t 1-00:00:00
#SBATCH --gres=gpu:1,gpumem:40G
#SBATCH --output=sb_sanity_%j.log
#SBATCH --error=sb_sanity_%j.err

set -euo pipefail
module load slurm/current cuda
export PATH="$HOME/.local/bin:$PATH"
cd /home/afrenk/SBLM

export SCRATCH_DIR="/scratch/${USER}"
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH_DIR}/torchinductor_cache"
export TRITON_CACHE_DIR="${SCRATCH_DIR}/triton_cache"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${SCRATCH_DIR}/checkpoints"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
mapfile -t allocated_hosts < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR="${allocated_hosts[0]}"
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

printf 'Launching %s node(s), %s GPU(s) per node; rendezvous %s:%s\n' \
    "$SLURM_JOB_NUM_NODES" "$SLURM_GPUS_ON_NODE" "$MASTER_ADDR" "$MASTER_PORT"

srun --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 \
    --kill-on-bad-exit=1 --gpu-bind=none bash -ec '
    exec uv run --no-sync torchrun \
        --nnodes="$SLURM_JOB_NUM_NODES" \
        --nproc_per_node="$SLURM_GPUS_ON_NODE" \
        --node_rank="$SLURM_NODEID" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        --max_restarts=0 train.py \
        --stage 1 \
        --data_dir "$SCRATCH_DIR/climbmix_small_bins" \
        --ckpt_dir "$SCRATCH_DIR/checkpoints" \
        --max_steps 35000 "$@"
    ' bash "$@"
