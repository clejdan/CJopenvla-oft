#!/bin/bash
#SBATCH --job-name=dp-dump
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/clejahd/rizon4/logs/dp_dump_%j.out
#SBATCH --error=/scratch/clejahd/rizon4/logs/dp_dump_%j.err
# Usage: sbatch [--dependency=afterok:<convert_jobid>] scripts/rizon4/dp/dump_dp_data.sbatch   (CPU only; env openvla-oft)
# Needs rizon4_three_cam_strawberry_harvest from: THREE_CAM=1 PRIMARY=side sbatch scripts/rizon4/convert_rizon4.sbatch
REPO=${REPO:-$HOME/CJopenvla-oft}
source ~/.bashrc; module load anaconda3; conda activate openvla-oft
cd "$REPO"
python scripts/rizon4/dp/dump_rlds_3cam.py /scratch/$USER/tensorflow_datasets/rizon4_three_cam_strawberry_harvest/1.0.0 /scratch/$USER/rizon4/dp_data
