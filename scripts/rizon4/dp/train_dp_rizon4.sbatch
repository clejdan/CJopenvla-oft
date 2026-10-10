#!/bin/bash
#SBATCH --job-name=dp-rizon4
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=/scratch/clejahd/rizon4/logs/dp_train_%j.out
#SBATCH --error=/scratch/clejahd/rizon4/logs/dp_train_%j.err
# Usage: sbatch [--dependency=afterok:<dump_jobid>] scripts/rizon4/dp/train_dp_rizon4.sbatch   (env lerobot)
# Extra args pass through, e.g.:  sbatch ... train_dp_rizon4.sbatch --max_steps 200 --val_freq 100
REPO=${REPO:-$HOME/openvla-oft}
source ~/.bashrc; module load anaconda3; conda activate lerobot
cd "$REPO"
export HF_HUB_OFFLINE=1 WANDB_MODE=offline
python scripts/rizon4/dp/train_dp.py \
  --data_dir /scratch/$USER/rizon4/dp_data \
  --out_dir /scratch/$USER/rizon4/dp_ckpt \
  --horizon 64 --n_action_steps 32 --n_obs_steps 2 \
  --batch_size 64 --lr 1e-4 --max_steps 100000 --workers 10 "$@"
