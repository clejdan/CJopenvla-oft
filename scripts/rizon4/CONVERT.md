# Converting LeRobot data to RLDS (Rizon 4)

Script: `scripts/rizon4/lerobot_to_rlds.py`. OFT trains on RLDS; StrawberryHarvester data is LeRobot v3.
Output dataset name is always `rizon4_strawberry_harvest`.

## Run (Palmetto, ~10 min)
```bash
cd ~/openvla-oft
PRIMARY=side sbatch scripts/rizon4/convert_rizon4.sbatch
```
| Variable | Meaning |
|---|---|
| `PRIMARY` | third-person camera: `side` or `scene` (wrist is always used) |
| `REPOS` | HF datasets, space-separated. Default: `StrawberryHarvester/MASKED_PURPLE_Pot_1_and_2_Red-3_redo_side_full-long-2026-09-17` |
| `SKIP_EXTRACT=1` | re-run only the build stage |

Example: `REPOS="StrawberryHarvester/setA StrawberryHarvester/setB" PRIMARY=side sbatch scripts/rizon4/convert_rizon4.sbatch`

Output: `/scratch/$USER/tensorflow_datasets/rizon4_strawberry_harvest/1.0.0` → pass its parent as `--data_root_dir`.
Logs: `/scratch/$USER/rizon4/logs/convert_<jobid>.{out,err}`.

## Check the result
```bash
sbatch -t 00:20:00 -c 4 --mem=24gb -o /scratch/$USER/rizon4/logs/check_%j.out --wrap \
 "source ~/.bashrc; module load anaconda3; conda activate openvla-oft; python scripts/rizon4/check_rlds.py /scratch/$USER/tensorflow_datasets/rizon4_strawberry_harvest/1.0.0"
```
Expect: train/val episode counts, images `(256,256,3)`, state/action `(8,)`, the task text.

## What it does
1. **extract** (env `lerobot`): decodes the videos once, in parallel → one `.npz` per episode in `/scratch/$USER/rizon4/npz`. Cross-checks 2 sampled episodes against `LeRobotDataset` and aborts on a mismatch.
2. **build** (env `openvla-oft`): `.npz` → TFDS. Every 10th episode goes to `val`.

Two envs because `lerobot` (numpy ≥ 2) and TensorFlow 2.15 (numpy < 2) can't share one.

## By hand (inside a job)
```bash
# env: lerobot
python scripts/rizon4/lerobot_to_rlds.py extract --repo_ids <hf repos> --npz_dir <npz dir> --primary_camera side
# env: openvla-oft
python scripts/rizon4/lerobot_to_rlds.py build --npz_dir <npz dir> --output_dir <tfds dir>
```
Extra flags: `--task_override "text"`, `--workers N`, `--verify 0` (extract); `--val_every N`, 0 = no val (build).

## Pitfalls
- **Same `--npz_dir` twice:** existing episodes are skipped, so a different dataset silently reuses old files. Pass all repos in one run, or use a new `--npz_dir`.
- **Same output twice:** the dataset name is fixed. Move or delete the old output first, or use another `--output_dir`.
- **Camera is baked in:** train and deploy with the same camera, same position.
- **Mixed tasks:** each episode keeps its own task text.
- Don't run any of this on the login node. `.npz` files (~29 GB) can be deleted after a successful build.

## 3-camera variant (side + scene + wrist)
Stores both third-person cameras plus wrist. Dataset name: `rizon4_three_cam_strawberry_harvest`.
```bash
THREE_CAM=1 PRIMARY=side sbatch scripts/rizon4/convert_rizon4.sbatch   # npz → /scratch/$USER/rizon4/npz_3cam (~45 GB)
sbatch scripts/rizon4/finetune_rizon4_3cam.sbatch                       # --num_images_in_input 3
```
- `PRIMARY` = main view, the other third-person camera becomes `secondary_image`. Image order fed to the model: primary, secondary, wrist.
- Deployment must supply all three cameras in that order, same resize.
- Needs `datasets.py` change: `image_secondary` is passed alongside the wrist view(s).
