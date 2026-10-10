"""Sanity-check the converted RLDS dataset: splits, shapes, dtypes, instruction, action/state ranges."""
import sys
import numpy as np
import tensorflow_datasets as tfds

path = sys.argv[1]
b = tfds.builder_from_directory(path)
print({k: v.num_examples for k, v in b.info.splits.items()})
for split in b.info.splits:
    n = steps = 0
    A, S = [], []
    for ep in tfds.as_numpy(b.as_dataset(split=split)):
        st = list(ep["steps"])  # nested tf.data dataset -> list of step dicts
        n += 1
        steps += len(st)
        A += [x["action"] for x in st]; S += [x["observation"]["state"] for x in st]
    A, S = np.stack(A), np.stack(S)
    print(f"{split}: {n} episodes, {steps} steps")
    print("  action min", A.min(0).round(2), "\n  action max", A.max(0).round(2))
    print("  state  q01", np.quantile(S, .01, 0).round(2), "\n  state  q99", np.quantile(S, .99, 0).round(2))
    print("  max |action-state| per dim", np.abs(A - S).max(0).round(2))
ex = list(next(iter(tfds.as_numpy(b.as_dataset(split="train").take(1))))["steps"])
o = ex[0]["observation"]
print("image", o["image"].shape, o["image"].dtype, "wrist", o["wrist_image"].shape, "state", o["state"].shape, "action", ex[0]["action"].shape)
print("instruction:", ex[0]["language_instruction"].decode())
print("is_first[0], is_last[-1]:", ex[0]["is_first"], ex[-1]["is_last"])
