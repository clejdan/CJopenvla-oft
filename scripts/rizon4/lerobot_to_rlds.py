"""
Convert LeRobot v3.0 datasets (e.g. the StrawberryHarvester Flexiv Rizon 4 sets on Hugging Face) into the RLDS /
TFDS format that OpenVLA-OFT's finetune.py reads. Output dataset name: `rizon4_strawberry_harvest`.

Two stages, because lerobot (numpy>=2, protobuf>=6) and TensorFlow 2.15 (numpy<2, protobuf<5) cannot share an env:

  1. extract  (env with av + pandas + pyarrow + pillow + huggingface_hub; lerobot only for --verify; NO tensorflow):
       python scripts/rizon4/lerobot_to_rlds.py extract --repo_ids <hf repos> --npz_dir /scratch/$USER/rizon4/npz \
           --primary_camera side
     -> one uncompressed .npz per episode: image, wrist_image (256x256x3 uint8), state, action (float32 x 8), task.

  2. build    (env with tensorflow + tensorflow_datasets, e.g. openvla-oft; no lerobot needed):
       python scripts/rizon4/lerobot_to_rlds.py build --npz_dir /scratch/$USER/rizon4/npz \
           --output_dir /scratch/$USER/tensorflow_datasets
     -> <output_dir>/rizon4_strawberry_harvest/1.0.0 with `train` and `val` splits.

--three_cam (both stages): also store the other third-person camera (`scene` if --primary_camera=side, else `side`) as
`secondary_image` and write dataset `rizon4_three_cam_strawberry_harvest` (use with --num_images_in_input 3).

Per-step RLDS fields: observation.{image,wrist_image,state}, action, language_instruction (+ is_first/is_last/...).
`--task_override` (extract) replaces the dataset's task text.
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image

DATASET_NAME = "rizon4_strawberry_harvest"
IMG_SIZE = 256  # Same as other OFT fine-tuning sets; finetune.py resizes to 224 internally.


def to_uint8_hwc(img) -> np.ndarray:
    """LeRobot returns CHW float [0,1] torch tensors; convert to resized HWC uint8."""
    arr = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
    if arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(arr).resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC))


# ----------------------------------------------------------------------------------------------------------------------
# Stage 1: LeRobot v3 -> per-episode .npz  (needs av, pandas, pyarrow, pillow, huggingface_hub; no tensorflow)
#
# Fast path: instead of LeRobotDataset[i] (per-frame seek + decode of every camera), each (camera, video file) is decoded
# once, front to back, in a worker process. Episodes are cut out by timestamp using meta/episodes. State/action come
# straight from the data parquet. `--verify N` cross-checks N frames against LeRobotDataset (needs lerobot installed).
# ----------------------------------------------------------------------------------------------------------------------
def _decode_task(job):
    """Decode one video file fully; write one resized .npy per episode contained in it. Runs in a worker process."""
    import av

    path, tmp_dir, cam, eps, fps = job  # eps: list of (episode_index, query_times ndarray)
    todo = [(e, q) for e, q in eps if not os.path.exists(os.path.join(tmp_dir, f"{cam}_ep{e:04d}.npy"))]
    if not todo:
        return cam, path, 0.0
    times, frames = [], []
    with av.open(path) as c:
        st = c.streams.video[0]
        st.codec_context.thread_count = 1
        for fr in c.decode(st):
            times.append(float(fr.pts * st.time_base))
            frames.append(np.asarray(Image.fromarray(fr.to_ndarray(format="rgb24")).resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)))
    times = np.asarray(times)
    order = np.argsort(times)
    times = times[order]
    frames = np.stack(frames)[order]
    max_err = 0.0
    for e, q in todo:
        j = np.clip(np.searchsorted(times, q), 1, len(times) - 1)
        j = np.where(np.abs(times[j - 1] - q) <= np.abs(times[j] - q), j - 1, j)  # nearest frame
        err = float(np.abs(times[j] - q).max())
        if err > 1.0 / fps:
            raise RuntimeError(f"{path}: episode {e} frame timestamps off by {err:.3f}s (> 1 frame)")
        max_err = max(max_err, err)
        np.save(os.path.join(tmp_dir, f"{cam}_ep{e:04d}.npy"), frames[j])
    return cam, path, max_err


def extract(args):
    import json
    from concurrent.futures import ProcessPoolExecutor

    import pandas as pd
    from huggingface_hub import snapshot_download

    os.makedirs(args.npz_dir, exist_ok=True)
    tmp_dir = os.path.join(args.npz_dir, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    secondary = "scene" if args.primary_camera == "side" else "side"
    cams = [args.primary_camera, "wrist"] + ([secondary] if args.three_cam else [])
    workers = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))

    for r, repo_id in enumerate(args.repo_ids):
        root = snapshot_download(repo_id, repo_type="dataset", allow_patterns=[
            "meta/*", "data/*", *[f"videos/observation.images.{c}/*" for c in cams]])
        info = json.load(open(os.path.join(root, "meta", "info.json")))
        fps = info["fps"]
        read = lambda pat: pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(root, pat)))],
                                     ignore_index=True)
        episodes = read("meta/episodes/*/*.parquet").sort_values("episode_index").reset_index(drop=True)
        data = read("data/*/*.parquet").sort_values("index").reset_index(drop=True)
        assert len(data) == info["total_frames"], (len(data), info["total_frames"])
        state_all = np.stack(data["observation.state"].to_numpy()).astype(np.float32)
        action_all = np.stack(data["action"].to_numpy()).astype(np.float32)
        ts_all = data["timestamp"].to_numpy(dtype=np.float64)
        print(f"{repo_id}: {len(episodes)} episodes, {len(data)} frames, {fps} fps, {workers} workers", flush=True)

        # one decode job per (camera, video file)
        jobs = {}
        todo_eps = []
        for _, ep in episodes.iterrows():
            e = int(ep["episode_index"])
            if os.path.exists(os.path.join(args.npz_dir, f"r{r}_ep{e:04d}.npz")):
                continue  # resumable
            todo_eps.append(e)
            lo, hi = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
            for cam in cams:
                k = f"videos/observation.images.{cam}"
                vp = os.path.join(root, info["video_path"].format(
                    video_key=f"observation.images.{cam}", chunk_index=int(ep[f"{k}/chunk_index"]),
                    file_index=int(ep[f"{k}/file_index"])))
                q = float(ep[f"{k}/from_timestamp"]) + ts_all[lo:hi]
                jobs.setdefault((cam, vp), []).append((e, q))
        with ProcessPoolExecutor(workers) as pool:
            for cam, path, err in pool.map(_decode_task, [(p, tmp_dir, c, eps, fps) for (c, p), eps in jobs.items()]):
                print(f"decoded {cam} {os.path.relpath(path, root)} (max timestamp err {err * 1000:.1f} ms)", flush=True)

        for _, ep in episodes.iterrows():
            e = int(ep["episode_index"])
            if e not in todo_eps:
                continue
            lo, hi = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
            img = np.load(os.path.join(tmp_dir, f"{args.primary_camera}_ep{e:04d}.npy"))
            wrist = np.load(os.path.join(tmp_dir, f"wrist_ep{e:04d}.npy"))
            extra = {}
            if args.three_cam:
                extra["secondary_image"] = np.load(os.path.join(tmp_dir, f"{secondary}_ep{e:04d}.npy"))
                assert len(extra["secondary_image"]) == hi - lo, (e, len(extra["secondary_image"]), hi - lo)
            assert len(img) == len(wrist) == hi - lo, (e, len(img), len(wrist), hi - lo)
            task = args.task_override or str(list(ep["tasks"])[0])
            out = os.path.join(args.npz_dir, f"r{r}_ep{e:04d}.npz")
            np.savez(out + ".tmp.npz", image=img, wrist_image=wrist, state=state_all[lo:hi], action=action_all[lo:hi],
                     task=np.array(task), repo_id=np.array(repo_id), episode=np.array(e), **extra)
            os.replace(out + ".tmp.npz", out)
            for c in cams:
                os.remove(os.path.join(tmp_dir, f"{c}_ep{e:04d}.npy"))
            print(f"{repo_id} ep {e}: {hi - lo} steps, task={task!r}", flush=True)

        if args.verify > 0:
            verify(args, repo_id, r, episodes)


def verify(args, repo_id, r, episodes):
    """Compare a few frames/state/action against LeRobotDataset (slow path). Aborts on mismatch."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, video_backend="pyav")
    rng = np.random.default_rng(0)
    for e in rng.choice(len(episodes), size=min(2, len(episodes)), replace=False):
        d = np.load(os.path.join(args.npz_dir, f"r{r}_ep{int(e):04d}.npz"))
        lo, n = int(episodes.loc[e, "dataset_from_index"]), len(d["action"])
        for k in rng.choice(n, size=args.verify, replace=False):
            f = ds[lo + int(k)]
            pairs = [(d["image"][k], f"observation.images.{args.primary_camera}"), (d["wrist_image"][k], "observation.images.wrist")]
            if args.three_cam:
                sec = "scene" if args.primary_camera == "side" else "side"
                pairs.append((d["secondary_image"][k], f"observation.images.{sec}"))
            for mine, key in pairs:
                diff = np.abs(mine.astype(np.float32) - to_uint8_hwc(f[key]).astype(np.float32)).mean()
                assert diff < 3.0, f"ep {e} frame {k} {key}: mean abs pixel diff {diff:.1f} (frames misaligned?)"
            assert np.allclose(d["state"][k], f["observation.state"].numpy()) and np.allclose(d["action"][k], f["action"].numpy())
        print(f"verify ok: episode {int(e)}", flush=True)


# ----------------------------------------------------------------------------------------------------------------------
# Stage 2: .npz -> RLDS / TFDS  (needs tensorflow + tensorflow_datasets; no lerobot)
# ----------------------------------------------------------------------------------------------------------------------
def build(args):
    import tensorflow_datasets as tfds

    files = sorted(glob.glob(os.path.join(args.npz_dir, "r*_ep*.npz")))
    assert files, f"no episodes found in {args.npz_dir}"

    three = args.three_cam

    class _Base(tfds.core.GeneratorBasedBuilder):
        VERSION = tfds.core.Version("1.0.0")
        RELEASE_NOTES = {"1.0.0": "Flexiv Rizon 4 strawberry harvesting (LeRobot v3 -> RLDS)."}

        def _info(self):
            img = lambda: tfds.features.Image(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8, encoding_format="jpeg")
            vec = lambda: tfds.features.Tensor(shape=(8,), dtype=np.float32)
            return tfds.core.DatasetInfo(
                builder=self,
                description="Flexiv Rizon 4 strawberry harvesting demonstrations (LeRobot v3 -> RLDS).",
                features=tfds.features.FeaturesDict({
                    "steps": tfds.features.Dataset({
                        "observation": tfds.features.FeaturesDict({
                            "image": img(), "wrist_image": img(), "state": vec(),
                            **({"secondary_image": img()} if three else {}),
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
            for path in files:
                with np.load(path) as z:  # NpzFile re-reads the array on every d[key] access -> load each once
                    d = {k: z[k] for k in z.files}
                e = int(d["episode"])
                is_val = args.val_every > 0 and e % args.val_every == 0
                if (split == "val") != is_val:
                    continue
                n = len(d["action"])
                task = str(d["task"])
                steps = [{
                    "observation": {"image": d["image"][t], "wrist_image": d["wrist_image"][t], "state": d["state"][t],
                                    **({"secondary_image": d["secondary_image"][t]} if three else {})},
                    "action": d["action"][t],
                    "discount": 1.0,
                    "reward": float(t == n - 1),
                    "is_first": t == 0,
                    "is_last": t == n - 1,
                    "is_terminal": t == n - 1,
                    "language_instruction": task,
                } for t in range(n)]
                yield os.path.basename(path), {
                    "steps": steps,
                    "episode_metadata": {"file_path": str(d["repo_id"]), "episode_index": e},
                }

    # tfds derives the dataset name from the class name (camelCase -> snake_case)
    class Rizon4StrawberryHarvest(_Base):  # -> rizon4_strawberry_harvest
        pass

    class Rizon4ThreeCamStrawberryHarvest(_Base):  # -> rizon4_three_cam_strawberry_harvest
        pass

    cls = Rizon4ThreeCamStrawberryHarvest if three else Rizon4StrawberryHarvest
    cls(data_dir=args.output_dir).download_and_prepare()
    print(f"Wrote {cls.name} to {args.output_dir}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="stage", required=True)
    a = sub.add_parser("extract")
    a.add_argument("--repo_ids", nargs="+", required=True)
    a.add_argument("--npz_dir", required=True)
    a.add_argument("--primary_camera", default="scene", choices=["scene", "side"])
    a.add_argument("--task_override", default=None)
    a.add_argument("--three_cam", action="store_true", help="also store the other third-person camera as secondary_image")
    a.add_argument("--workers", type=int, default=0, help="decode processes (default: SLURM_CPUS_PER_TASK / all cores)")
    a.add_argument("--verify", type=int, default=2, help="frames per sampled episode to cross-check vs LeRobotDataset (0=off)")
    b = sub.add_parser("build")
    b.add_argument("--npz_dir", required=True)
    b.add_argument("--three_cam", action="store_true", help="build rizon4_three_cam_strawberry_harvest (npz must come from extract --three_cam)")
    b.add_argument("--output_dir", default=os.path.expanduser("~/tensorflow_datasets"))
    b.add_argument("--val_every", type=int, default=10, help="every Nth episode goes to val (0 = no val)")
    args = p.parse_args()
    {"extract": extract, "build": build}[args.stage](args)


if __name__ == "__main__":
    main()
