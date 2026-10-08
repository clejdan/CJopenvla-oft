#!/bin/bash
#SBATCH --job-name=oft-rizon4
#SBATCH --partition=work1
#SBATCH --gres=gpu:a100:1          # check `sinfo -o "%G %N"`; 7B + LoRA needs ~30-60 GB, so use a 40/80 GB GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=rizon4.%j.out
#SBATCH --error=rizon4.%j.err

# Usage: sbatch scripts/rizon4/finetune_rizon4.sbatch
# Paths: edit for your account.
REPO=${REPO:-$HOME/CJopenvla-oft}
DATA_ROOT=${DATA_ROOT:-/scratch/$USER/tensorflow_datasets}   # contains rizon4_strawberry_harvest/
RUN_ROOT=${RUN_ROOT:-/scratch/$USER/openvla-checkpoints}

module load miniforge3/24.3.0-0 cuda/12.3.0
source activate openvla-oft
cd "$REPO"
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache}

# NOTE: "rizon4" must appear in the command line -- prismatic/vla/constants.py uses it to pick
# the Rizon 4 constants (ACTION_DIM=8, PROPRIO_DIM=8, NUM_ACTIONS_CHUNK=20). The dataset name provides it.
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir "$DATA_ROOT" \
  --dataset_name rizon4_strawberry_harvest \
  --run_root_dir "$RUN_ROOT" \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film True \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 4 \
  --grad_accumulation_steps 4 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 20000 \
  --max_steps 30000 \
  --use_val_set True \
  --val_freq 2500 \
  --save_freq 5000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity "${WANDB_ENTITY:-your-wandb-entity}" \
  --wandb_project "${WANDB_PROJECT:-rizon4-oft}" \
  --run_id_note rizon4_strawberry
