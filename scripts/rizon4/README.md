# OpenVLA-OFT on Flexiv Rizon 4 (StrawberryHarvester data)

The StrawberryHarvester HF datasets are LeRobot v3.0 (`robot_type: rizon4`, 20 fps, 8-D state/action =
7 joint pos + gripper pos, cameras `scene`/`side`/`wrist`). OFT trains from RLDS, so:

1. **Convert** (any machine with `tensorflow tensorflow_datasets lerobot av pillow`):
   ```bash
   python scripts/rizon4/lerobot_to_rlds.py \
     --repo_ids StrawberryHarvester/Single_Strawberry_Pot_1_15ep_9-23-26 \
                StrawberryHarvester/Single_Strawberry_Pot_2_15ep_9-24-26 \
                StrawberryHarvester/Single_Strawberry_Pot_3_15ep_9-29-26 \
                StrawberryHarvester/Single_Strawberry_Pot_4_15ep_9-29-26 \
     --output_dir ~/tensorflow_datasets
   ```
   Produces `rizon4_strawberry_harvest/1.0.0` (train + val). Use `--primary_camera side` to swap the third-person view.
   Copy it to Palmetto scratch (`rsync -av ~/tensorflow_datasets/rizon4_strawberry_harvest/ palmetto:/scratch/$USER/tensorflow_datasets/rizon4_strawberry_harvest/`).
2. **Train**: `sbatch scripts/rizon4/finetune_rizon4.sbatch` (edit paths/GPU type/wandb first).
3. Model inputs: primary + wrist image (`--num_images_in_input 2`), proprio (8-D), predicts 20-step chunks of 8-D
   **absolute** joint targets (≈1 s at 20 Hz).

Repo changes: `RIZON4_CONSTANTS` (selected when `rizon4` appears in `sys.argv`), `ActionEncoding.JOINT_POS_RIZON4`,
and `rizon4_strawberry_harvest` entries in the OXE configs/mixtures/transforms/materialize files.

Gotchas
- Constants are picked from the command line, so any script run for training *or* deployment must have "rizon4" in its
  args (dataset name / checkpoint path both work).
- Images are squashed 640x480 -> 256x256 at conversion; do the same at deployment time.
- Actions are absolute joint positions, so at deployment clamp the target against the current joint state
  (max per-step delta) before streaming to Flexiv RDK. No Rizon 4 eval/deploy client exists in this repo yet.
- Datasets with different tasks (`Pot_1_and_2_*`, `MASKED_PURPLE_*`) can be added to `--repo_ids`; their task text becomes the language prompt.
