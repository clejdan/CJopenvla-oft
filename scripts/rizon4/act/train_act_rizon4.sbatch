#!/bin/bash
#SBATCH --job-name=act-rizon4
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/clejahd/rizon4/logs/act_train_%j.out
#SBATCH --error=/scratch/clejahd/rizon4/logs/act_train_%j.err
# Usage: sbatch --dependency=afterok:<dump_jobid> scripts/rizon4/act/train_act_rizon4.sbatch   (env lerobot)
REPO=${REPO:-$HOME/openvla-oft}
source ~/.bashrc; module load anaconda3; conda activate lerobot
cd "$REPO"
export HF_HUB_OFFLINE=1 WANDB_MODE=offline
python scripts/rizon4/act/train_act.py \
  --data_dir /scratch/$USER/rizon4/act_data \
  --out_dir /scratch/$USER/rizon4/act_ckpt \
  --chunk_size 50 --batch_size 32 --max_steps 50000 --workers 10
