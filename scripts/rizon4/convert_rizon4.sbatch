#!/bin/bash
#SBATCH --job-name=rizon4-convert
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96gb
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/clejahd/rizon4/logs/convert_%j.out
#SBATCH --error=/scratch/clejahd/rizon4/logs/convert_%j.err
# CPU-only. Two stages in two existing envs (lerobot and tensorflow cannot share one env):
#   1. extract: LeRobot v3 -> per-episode .npz        (conda env `lerobot`; parallel pyav decode, ~SLURM_CPUS_PER_TASK procs)
#   2. build:   .npz -> RLDS/TFDS                      (conda env `openvla-oft`)
# Usage: [THREE_CAM=1] PRIMARY=side sbatch scripts/rizon4/convert_rizon4.sbatch   (override sets with REPOS="a/b c/d")
set -e
source ~/.bashrc
module load anaconda3
export HF_HOME=/scratch/clejahd/hf_cache
# THREE_CAM=1 -> also store the second third-person camera; writes rizon4_three_cam_strawberry_harvest, separate npz dir
if [ "${THREE_CAM:-0}" = 1 ]; then
  NPZ=/scratch/clejahd/rizon4/npz_3cam; FLAG=--three_cam
else
  NPZ=/scratch/clejahd/rizon4/npz; FLAG=
fi
OUT=/scratch/clejahd/tensorflow_datasets
REPOS=${REPOS:-"StrawberryHarvester/MASKED_PURPLE_Pot_1_and_2_Red-3_redo_side_full-long-2026-09-17"}
cd /home/clejahd/openvla-oft

if [ "${SKIP_EXTRACT:-0}" != 1 ]; then
conda activate lerobot
python scripts/rizon4/lerobot_to_rlds.py extract --repo_ids $REPOS --npz_dir "$NPZ" --primary_camera "${PRIMARY:-scene}" $FLAG
fi

conda activate openvla-oft
mkdir -p "$OUT"
python scripts/rizon4/lerobot_to_rlds.py build --npz_dir "$NPZ" --output_dir "$OUT" $FLAG
