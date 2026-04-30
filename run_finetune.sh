#!/bin/bash
cd /home/clejahd/openvla-oft
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py --vla_path openvla/openvla-7b --data_root_dir /home/clejahd/tensorflow_datasets --dataset_name so101_block_pickup --run_root_dir /home/clejahd/openvla-checkpoints --use_l1_regression True --use_film True --num_images_in_input 2 --use_proprio True --batch_size 16 --learning_rate 5e-4 --max_steps 30000 --save_freq 5000 --lora_rank 3
