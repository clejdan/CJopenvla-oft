#!/bin/bash
#SBATCH --job-name=rizon4-convert
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/clejahd/rizon4/logs/convert_%j.out
#SBATCH --error=/scratch/clejahd/rizon4/logs/convert_%j.err
# CPU-only: LeRobot v3 (StrawberryHarvester) -> RLDS. Creates a converter env on first run.
# Override datasets with: REPOS="a/b c/d" sbatch scripts/rizon4/convert_rizon4.sbatch
set -e
module load anaconda3
export HF_HOME=/scratch/clejahd/hf_cache
OUT=/scratch/clejahd/tensorflow_datasets
ENV=/scratch/clejahd/envs/rizon4-convert
REPOS=${REPOS:-"StrawberryHarvester/MASKED_PURPLE_Pot_1_and_2_Red-3_redo_side_full-long-2026-09-17"}

if [ ! -d "$ENV" ]; then
  conda create -y -p "$ENV" python=3.10
fi
source activate "$ENV"
python -c "import tensorflow_datasets, lerobot" 2>/dev/null || pip install "tensorflow-cpu==2.15.0" "tensorflow_datasets==4.9.3" "lerobot==0.4.4" av pillow

mkdir -p "$OUT"
cd /home/clejahd/openvla-oft
python scripts/rizon4/lerobot_to_rlds.py --repo_ids $REPOS --output_dir "$OUT" --primary_camera "${PRIMARY:-scene}"
