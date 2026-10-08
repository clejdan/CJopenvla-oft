# OpenVLA-OFT on Flexiv Rizon 4 (StrawberryHarvester data)

Status (2026-10-08): code + scripts done; conversion job submitted. Converter, training and deployment **not yet run/verified**.

## Data
HF org `StrawberryHarvester`, LeRobot v3.0, `robot_type: rizon4`, 20 fps, state/action = 8-D (7 joints + gripper, absolute),
cameras `scene`/`side`/`wrist`. Used: `MASKED_PURPLE_Pot_1_and_2_Red-3_redo_side_full-long-2026-09-17` (40 eps, 76,615 frames).

```bash
# find the datasets
python3 -c "from huggingface_hub import HfApi; print([d.id for d in HfApi().list_datasets(author='StrawberryHarvester')])"
# inspect one (fps, shapes, cameras, #episodes)
python3 -c "from huggingface_hub import hf_hub_download as h; print(open(h('StrawberryHarvester/<name>','meta/info.json',repo_type='dataset')).read())"
```

## Code changes
- `prismatic/vla/constants.py`: `RIZON4_CONSTANTS` (chunk 20, action dim 8, proprio dim 8, `BOUNDS_Q99`); chosen when `rizon4` is in `sys.argv`.
- `prismatic/vla/datasets/rlds/oxe/{configs,materialize,mixtures,transforms}.py`: `ActionEncoding.JOINT_POS_RIZON4`, dataset `rizon4_strawberry_harvest`.
- `scripts/rizon4/lerobot_to_rlds.py`: LeRobot v3 -> RLDS (256x256 squashed images, primary + wrist, `--primary_camera {scene,side}`, every 10th episode -> val).
- `scripts/rizon4/convert_rizon4.sbatch`, `scripts/rizon4/finetune_rizon4.sbatch`.

`rizon4` must appear in the args of any train/eval/deploy process, else it silently falls back to LIBERO constants.

## Palmetto steps
```bash
# local: open a reusable SSH connection (one line, normal terminal; password + Duo)
ssh -fN -o ControlMaster=yes -o ControlPath=$HOME/.ssh/cm-palmetto -o ControlPersist=4h <user>@slogin.palmetto.clemson.edu
ssh -o ControlPath=$HOME/.ssh/cm-palmetto <user>@slogin.palmetto.clemson.edu '<cmd>'   # reuse it

# local: copy changes to the cluster clone (~/openvla-oft)
rsync -avR prismatic/vla/constants.py prismatic/vla/datasets/rlds/oxe/{configs,materialize,mixtures,transforms}.py scripts/rizon4 \
  <user>@slogin.palmetto.clemson.edu:openvla-oft/

# cluster: checks used
checkquota; sinfo -h -o "%G %c %m" | sort -u; module avail cuda; conda env list
~/.conda/envs/openvla-oft/bin/python -c "import torch,tensorflow,tensorflow_datasets,flash_attn,draccus,peft"

# cluster: convert (CPU job; builds env /scratch/$USER/envs/rizon4-convert on first run)
mkdir -p /scratch/$USER/rizon4/logs
cd ~/openvla-oft && PRIMARY=side sbatch scripts/rizon4/convert_rizon4.sbatch   # submitted as job 16826705
squeue -u $USER
tail /scratch/$USER/rizon4/logs/convert_<jobid>.{out,err}

# cluster: train (after verifying output)
sbatch scripts/rizon4/finetune_rizon4.sbatch
```
Converter env: python 3.10, `tensorflow-cpu==2.15.0 tensorflow_datasets==4.9.3 lerobot==0.4.4 av pillow` (kept separate from the `openvla-oft` env).
Output: `/scratch/$USER/tensorflow_datasets/rizon4_strawberry_harvest/1.0.0` -> use as `--data_root_dir`.
Edit the hard-coded `/scratch/clejahd`, `/home/clejahd` paths and `WANDB_*` in the sbatch files for your account. Keep data/envs/checkpoints on scratch (home is 250 GB).

## Verify conversion
```python
import tensorflow_datasets as tfds
b = tfds.builder_from_directory("/scratch/$USER/tensorflow_datasets/rizon4_strawberry_harvest/1.0.0")
print(b.info.splits)
s = next(iter(next(iter(b.as_dataset(split="train").take(1)))["steps"]))
print(s["observation"]["image"].shape, s["observation"]["state"].shape, s["action"].shape, s["language_instruction"])
```
Expect train ~36 / val ~4 episodes, (256,256,3), (8,), (8,); check the instruction text (it is the prompt).

## Training config (untuned)
`finetune.py`: `openvla/openvla-7b`, L1 head, FiLM, 2 images (side + wrist), proprio, batch 4 x accum 4, lr 5e-4, 30k steps (decay at 20k), LoRA r=32, 1x A100 (`--gpus-per-node=a100:1`). Lower batch / raise accum on OOM.

## Notes
- Train and deploy with the same camera (`side`, same pose) and the same 640x480 -> 256x256 squash.
- Run nothing heavy on the login node. Torch cu121 is correct for torch 2.2 (the Palmetto doc's `cu117` is not).
- No Rizon 4 deployment client exists yet. Actions are absolute joint targets: clamp per-step change and joint limits before sending via `flexivrdk`.
