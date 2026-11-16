#!/bin/bash 
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=256g
#SBATCH -J "pretrain"
#SBATCH -p short
#SBATCH -t 1-00:00:00
#SBATCH --gres=gpu:2
#SBATCH -C "H100|H200|A100-80G|RTX6000B"
#SBATCH --output=%x_%j.log

# emb_std should be \in [0.95, 1.05], vocab cos is ~ 0.00 and enc_adj_cos > 0.95
uv run --extra ampere torchrun --standalone --nproc_per_node=2 train.py \
  --stage 1 --data_dir climbmix_small_bins --max_steps 35000
