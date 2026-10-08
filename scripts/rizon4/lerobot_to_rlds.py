"""
Convert LeRobot v3.0 datasets (e.g. the StrawberryHarvester Flexiv Rizon 4 sets on Hugging Face) into the RLDS /
TFDS format that OpenVLA-OFT's finetune.py reads.

Output dataset name: `rizon4_strawberry_harvest` (must match the entries in prismatic/vla/datasets/rlds/oxe/*).

Per-step RLDS fields:
    observation.image         primary camera (default: `scene`), 256x256x3 uint8
    observation.wrist_image   wrist camera, 256x256x3 uint8
    observation.state         8D = 7 joint positions + gripper position
    action                    8D = 7 target joint positions + gripper target (absolute)
    language_instruction      task string

Usage (needs: pip install tensorflow tensorflow_datasets lerobot av pillow):
    python scripts/rizon4/lerobot_to_rlds.py \
        --repo_ids StrawberryHarvester/Single_Strawberry_Pot_1_15ep_9-23-26 StrawberryHarvester/Single_Strawberry_Pot_2_15ep_9-24-26 \
        --output_dir ~/tensorflow_datasets

Add --task_override "pick the ripe strawberry" to replace the dataset's task text.
"""
import argparse
import os

import numpy as np
import tensorflow_datasets as tfds
from PIL import Image

DATASET_NAME = "rizon4_strawberry_harvest"
IMG_SIZE = 256  # Same as other OFT fine-tuning sets; finetune.py resizes to 224 internally.
ARGS = None  # set in main(); read by the builder


def to_uint8_hwc(img) -> np.ndarray:
    """LeRobot returns CHW float [0,1] torch tensors; convert to resized HWC uint8."""
    arr = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
    if arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(arr).resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC))


def episodes_of(repo_id):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id)  # downloads to ~/.cache/huggingface/lerobot
    eps = ds.meta.episodes
    for e in range(ds.meta.total_episodes):
        yield repo_id, ds, e, int(eps["dataset_from_index"][e]), int(eps["dataset_to_index"][e])


class Rizon4StrawberryHarvest(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Flexiv Rizon 4 strawberry harvesting (LeRobot v3 -> RLDS)."}

    def _info(self):
        img = lambda: tfds.features.Image(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8, encoding_format="jpeg")
        vec = lambda: tfds.features.Tensor(shape=(8,), dtype=np.float32)
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "image": img(), "wrist_image": img(), "state": vec(),
                    }),
                    "action": vec(),
                    "discount": tfds.features.Scalar(dtype=np.float32),
                    "reward": tfds.features.Scalar(dtype=np.float32),
                    "is_first": tfds.features.Scalar(dtype=np.bool_),
                    "is_last": tfds.features.Scalar(dtype=np.bool_),
                    "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                    "language_instruction": tfds.features.Text(),
                }),
                "episode_metadata": tfds.features.FeaturesDict({
                    "file_path": tfds.features.Text(), "episode_index": tfds.features.Scalar(dtype=np.int32),
                }),
            })
        )

    def _split_generators(self, dl_manager):
        return {"train": self._generate_examples("train"), "val": self._generate_examples("val")}

    def _generate_examples(self, split):
        for repo_id in ARGS.repo_ids:
            for _, ds, e, lo, hi in episodes_of(repo_id):
                is_val = (e % ARGS.val_every == 0) and ARGS.val_every > 0
                if (split == "val") != is_val:
                    continue
                steps = []
                for i in range(lo, hi):
                    f = ds[i]
                    task = ARGS.task_override or str(f["task"])
                    steps.append({
                        "observation": {
                            "image": to_uint8_hwc(f[f"observation.images.{ARGS.primary_camera}"]),
                            "wrist_image": to_uint8_hwc(f["observation.images.wrist"]),
                            "state": f["observation.state"].numpy().astype(np.float32),
                        },
                        "action": f["action"].numpy().astype(np.float32),
                        "discount": 1.0,
                        "reward": float(i == hi - 1),
                        "is_first": i == lo,
                        "is_last": i == hi - 1,
                        "is_terminal": i == hi - 1,
                        "language_instruction": task,
                    })
                yield f"{repo_id}/{e}", {
                    "steps": steps,
                    "episode_metadata": {"file_path": repo_id, "episode_index": e},
                }


def main():
    global ARGS
    p = argparse.ArgumentParser()
    p.add_argument("--repo_ids", nargs="+", required=True)
    p.add_argument("--output_dir", default=os.path.expanduser("~/tensorflow_datasets"))
    p.add_argument("--primary_camera", default="scene", choices=["scene", "side"])
    p.add_argument("--val_every", type=int, default=10, help="every Nth episode goes to val (0 = no val)")
    p.add_argument("--task_override", default=None)
    ARGS = p.parse_args()
    Rizon4StrawberryHarvest(data_dir=ARGS.output_dir).download_and_prepare()
    print(f"Wrote {DATASET_NAME} to {ARGS.output_dir}")


if __name__ == "__main__":
    main()
