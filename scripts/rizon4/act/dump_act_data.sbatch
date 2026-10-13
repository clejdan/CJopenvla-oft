#!/bin/bash
#SBATCH --job-name=act-dump
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/clejahd/rizon4/logs/act_dump_%j.out
#SBATCH --error=/scratch/clejahd/rizon4/logs/act_dump_%j.err
# Usage: sbatch scripts/rizon4/act/dump_act_data.sbatch   (CPU only; env openvla-oft)
REPO=${REPO:-$HOME/CJopenvla-oft}
source ~/.bashrc; module load anaconda3; conda activate openvla-oft
cd "$REPO"
python scripts/rizon4/act/dump_rlds.py /scratch/$USER/tensorflow_datasets/rizon4_strawberry_harvest/1.0.0 /scratch/$USER/rizon4/act_data
