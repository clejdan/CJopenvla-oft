"""Stage 1 (env: openvla-oft): dump the 3-camera Rizon4 RLDS to per-episode .npy files for Diffusion Policy.

Layout: <out>/<split>/ep0000_{image,secondary,wrist,state,action}.npy  (+ <out>/meta.json)
image = primary third-person cam, secondary = other third-person cam, wrist = wrist cam.
Needs TF, which can't share an env with lerobot (numpy>=2), hence the separate stage.
"""
import argparse, json, os
import numpy as np
import tensorflow_datasets as tfds

p = argparse.ArgumentParser()
p.add_argument("rlds_dir", help=".../rizon4_three_cam_strawberry_harvest/1.0.0")
p.add_argument("out_dir")
a = p.parse_args()

b = tfds.builder_from_directory(a.rlds_dir)
meta = {}
for split in b.info.splits:
    os.makedirs(f"{a.out_dir}/{split}", exist_ok=True)
    lens = []
    for i, ep in enumerate(tfds.as_numpy(b.as_dataset(split=split))):
        st = list(ep["steps"])
        o = [s["observation"] for s in st]
        pre = f"{a.out_dir}/{split}/ep{i:04d}"
        np.save(pre + "_image.npy", np.stack([x["image"] for x in o]))
        np.save(pre + "_secondary.npy", np.stack([x["secondary_image"] for x in o]))
        np.save(pre + "_wrist.npy", np.stack([x["wrist_image"] for x in o]))
        np.save(pre + "_state.npy", np.stack([x["state"] for x in o]).astype(np.float32))
        np.save(pre + "_action.npy", np.stack([s["action"] for s in st]).astype(np.float32))
        meta.setdefault("instruction", st[0]["language_instruction"].decode())
        lens.append(len(st))
        print(split, i, len(st), flush=True)
    meta[split] = lens
json.dump(meta, open(f"{a.out_dir}/meta.json", "w"))
print("done", {k: (len(v) if isinstance(v, list) else v) for k, v in meta.items()})
