# OpenVLA-OFT on Flexiv Rizon 4 (StrawberryHarvester data)

Status (2026-10-08): code, scripts and data conversion done and verified (36 train / 4 val episodes). Training and deployment **not yet run**.

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
- `scripts/rizon4/lerobot_to_rlds.py`: LeRobot v3 -> RLDS in two stages (256x256 squashed JPEG images, primary + wrist, `--primary_camera {scene,side}`, every 10th episode -> val).
  - `extract` (env `lerobot`): parallel pyav decode of each video file once -> per-episode `.npz`; `--verify` cross-checks frames vs `LeRobotDataset`.
  - `build` (env `openvla-oft`): `.npz` -> TFDS. Two stages because lerobot (numpy>=2, protobuf>=6) and TF 2.15 (numpy<2, protobuf<5) can't share an env.
- `scripts/rizon4/convert_rizon4.sbatch` (both stages), `scripts/rizon4/finetune_rizon4.sbatch`, `scripts/rizon4/check_rlds.py` (sanity check).

`rizon4` must appear in the args of any train/eval/deploy process, else it silently falls back to LIBERO constants.

## Link local machine to Palmetto (so an assistant/scripts can run remote commands)
  rm ~/.ssh/cm-palmetto
  ssh -fN -o ControlMaster=yes -o ControlPath=$HOME/.ssh/cm-palmetto -o ControlPersist=4h clejahd@slogin.palmetto.clemson.edu
  ssh -o ControlPath=$HOME/.ssh/cm-palmetto -O check clejahd@slogin.palmetto.clemson.edu
Duo 2FA can't be answered by a tool, so a human opens one authenticated master connection; later commands reuse it with no login.
Run in a normal terminal on your **local** machine (not on the cluster, not inside Claude Code's `!`), as **one line**:
```bash
ssh -fN -o ControlMaster=yes -o ControlPath=$HOME/.ssh/cm-palmetto -o ControlPersist=4h <user>@slogin.palmetto.clemson.edu
```
Enter password, approve Duo push; the prompt returns. Then:
```bash
ls ~/.ssh/cm-palmetto                                                                    # socket exists
ssh -o ControlPath=$HOME/.ssh/cm-palmetto -O check clejahd@slogin.palmetto.clemson.edu     # "Master running"
ssh -o ControlPath=$HOME/.ssh/cm-palmetto <user>@slogin.palmetto.clemson.edu '<cmd>'      # run a command, no re-auth
rsync -av -e "ssh -o ControlPath=$HOME/.ssh/cm-palmetto" <local> <user>@slogin.palmetto.clemson.edu:<dest>   # copy files
ssh -o ControlPath=$HOME/.ssh/cm-palmetto -O exit <user>@slogin.palmetto.clemson.edu       # close the link
```
- A normal interactive `ssh` in another window does **not** create this link.
- Expires after 4 h idle (`ControlPersist`); just repeat the first command.
- "ControlSocket ... already exists" = a stale socket: `rm ~/.ssh/cm-palmetto`, retry.
- Commands run as you; use the login node only for light work (`sbatch`, `squeue`, `ls`, `rsync`).

## Palmetto steps
```bash
# local: copy changes to the cluster clone (~/openvla-oft)
rsync -avR prismatic/vla/constants.py prismatic/vla/datasets/rlds/oxe/{configs,materialize,mixtures,transforms}.py scripts/rizon4 \
  <user>@slogin.palmetto.clemson.edu:openvla-oft/

# cluster: checks used
checkquota; sinfo -h -o "%G %c %m" | sort -u; module avail cuda; conda env list
~/.conda/envs/openvla-oft/bin/python -c "import torch,tensorflow,tensorflow_datasets,flash_attn,draccus,peft"

# cluster: convert (CPU job, ~10 min total: extract ~6 min, build ~4 min, peak RAM ~35 GB)
mkdir -p /scratch/$USER/rizon4/logs
cd ~/openvla-oft && PRIMARY=side sbatch scripts/rizon4/convert_rizon4.sbatch     # SKIP_EXTRACT=1 to re-run only the build
squeue -u $USER
tail /scratch/$USER/rizon4/logs/convert_<jobid>.{out,err}

# cluster: verify the output (job; don't run on login node)
sbatch -t 00:20:00 -c 4 --mem=24gb -o /scratch/$USER/rizon4/logs/check_%j.out --wrap \
 "source ~/.bashrc; module load anaconda3; conda activate openvla-oft; python scripts/rizon4/check_rlds.py /scratch/$USER/tensorflow_datasets/rizon4_strawberry_harvest/1.0.0"

# cluster: train (after verifying output)
sbatch scripts/rizon4/finetune_rizon4.sbatch
```
Uses existing envs: `lerobot` (lerobot 0.4.4, av, pandas, pyarrow) for extract and `openvla-oft` (TF 2.15, tfds 4.9.3) for build. Intermediate `.npz` (~29 GB) in `/scratch/$USER/rizon4/npz` can be deleted after the build.
Output: `/scratch/$USER/tensorflow_datasets/rizon4_strawberry_harvest/1.0.0` (3.0 GB) -> use as `--data_root_dir`.
Edit the hard-coded `/scratch/clejahd`, `/home/clejahd` paths and `WANDB_*` in the sbatch files for your account. Keep data/envs/checkpoints on scratch (home is 250 GB).

## Verified result
`check_rlds.py`: train 36 eps / 68,150 steps, val 4 eps / 8,465 steps; image + wrist (256,256,3) uint8; state/action (8,);
gripper in [0,1]; |action - state| <= 0.05 rad per joint per step; instruction: "pick each highlighted strawberry, place it into the green bin, and return home".
Pitfall fixed: do not index `NpzFile` keys inside a per-step loop (re-reads the array each time -> OOM/slow).

## Training config (untuned)
`finetune.py`: `openvla/openvla-7b`, L1 head, FiLM, 2 images (side + wrist), proprio, batch 4 x accum 4, lr 5e-4, 30k steps (decay at 20k), LoRA r=32, 1x A100 (`--gpus-per-node=a100:1`). Lower batch / raise accum on OOM.

## Notes
- Train and deploy with the same camera (`side`, same pose) and the same 640x480 -> 256x256 squash.
- Run nothing heavy on the login node. Torch cu121 is correct for torch 2.2 (the Palmetto doc's `cu117` is not).
- No Rizon 4 deployment client exists yet. Actions are absolute joint targets: clamp per-step change and joint limits before sending via `flexivrdk`.
- Source videos are AV1; the default torchcodec decoder fails on Palmetto (no system FFmpeg), so the converter decodes with pyav.
