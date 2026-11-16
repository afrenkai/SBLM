#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=sblm-numeric
#SBATCH --partition=short,long
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1,gpumem:40G
#SBATCH --output=sb_numeric_%j.log
#SBATCH --error=sb_numeric_%j.err

set -e pipefail
cd /home/afrenk/SBLM
export TORCHINDUCTOR_COMPILE_THREADS=2
test_dir="/scratch/${USER}/sblm_numeric_${SLURM_JOB_ID}"

args=(
    --batch_size 32 --seq_len 512 --grad_accum 2
    --warmup_steps 4 --eval_interval 6 --save_interval 6 --log_interval 1
    --ckpt_dir "${test_dir}/checkpoints"
)

bash sanity_check.sh "${args[@]}" --max_steps 12
bash sanity_check.sh "${args[@]}" --max_steps 16 \
    --resume_ckpt "${test_dir}/checkpoints/stage1_last.pt"
