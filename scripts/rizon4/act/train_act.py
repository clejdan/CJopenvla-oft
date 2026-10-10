"""Stage 2 (env: lerobot): train lerobot ACT on the dumped Rizon4 data (see dump_rlds.py).

Cameras: image (side) + wrist. State/action 8-D (7 joints + gripper, absolute). Normalization stats are
saved to <out>/stats.json and MUST be applied identically at deployment (state/action mean-std,
images /255 then ImageNet mean/std).
"""
import argparse, glob, json, os, time
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

p = argparse.ArgumentParser()
p.add_argument("--data_dir", required=True)
p.add_argument("--out_dir", required=True)
p.add_argument("--chunk_size", type=int, default=50)
p.add_argument("--batch_size", type=int, default=32)
p.add_argument("--lr", type=float, default=1e-5)
p.add_argument("--max_steps", type=int, default=50000)
p.add_argument("--val_freq", type=int, default=2500)
p.add_argument("--save_freq", type=int, default=5000)
p.add_argument("--workers", type=int, default=8)
p.add_argument("--no_vae", action="store_true")
a = p.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
IMG_KEYS = {"image": "observation.images.image", "wrist": "observation.images.wrist"}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_split(split):
    eps = sorted({f.rsplit("_", 1)[0] for f in glob.glob(f"{a.data_dir}/{split}/ep*_state.npy")})
    return [{k: np.load(f"{e}_{k}.npy", mmap_mode="r") for k in ("image", "wrist", "state", "action")} for e in eps]


class Chunks(Dataset):
    def __init__(self, eps, stats):
        self.eps, self.s = eps, stats
        self.idx = [(e, t) for e, ep in enumerate(eps) for t in range(len(ep["state"]))]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        e, t = self.idx[i]
        ep, T, n = self.eps[e], len(self.eps[e]["state"]), a.chunk_size
        act = np.asarray(ep["action"][t:t + n])
        pad = np.zeros(n, bool)
        if len(act) < n:  # past episode end: repeat last action, mask it out of the loss
            pad[len(act):] = True
            act = np.concatenate([act, np.repeat(act[-1:], n - len(act), 0)])
        out = {
            "observation.state": torch.from_numpy((np.asarray(ep["state"][t]) - self.s["state_mean"]) / self.s["state_std"]).float(),
            "action": torch.from_numpy((act - self.s["action_mean"]) / self.s["action_std"]).float(),
            "action_is_pad": torch.from_numpy(pad),
        }
        for k, key in IMG_KEYS.items():
            out[key] = torch.from_numpy(np.array(ep[k][t])).permute(2, 0, 1)  # uint8 CHW
        return out


def prep(batch, dev):
    b = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
    for key in IMG_KEYS.values():
        b[key] = ((b[key].float() / 255 - MEAN.to(dev)) / STD.to(dev))
    return b


train_eps, val_eps = load_split("train"), load_split("val")
S = np.concatenate([ep["state"] for ep in train_eps]); A = np.concatenate([ep["action"] for ep in train_eps])
stats = {"state_mean": S.mean(0), "state_std": S.std(0) + 1e-6, "action_mean": A.mean(0), "action_std": A.std(0) + 1e-6}
json.dump({k: v.tolist() for k, v in stats.items()} | {"chunk_size": a.chunk_size, "img_keys": IMG_KEYS}, open(f"{a.out_dir}/stats.json", "w"))
print(f"train eps {len(train_eps)}, val eps {len(val_eps)}", flush=True)

cfg = ACTConfig(
    input_features={"observation.state": PolicyFeature(FeatureType.STATE, (8,)),
                    **{key: PolicyFeature(FeatureType.VISUAL, (3, 256, 256)) for key in IMG_KEYS.values()}},
    output_features={"action": PolicyFeature(FeatureType.ACTION, (8,))},
    chunk_size=a.chunk_size, n_action_steps=a.chunk_size, use_vae=not a.no_vae,
    optimizer_lr=a.lr, optimizer_lr_backbone=a.lr, push_to_hub=False, device="cuda")
dev = torch.device("cuda")
policy = ACTPolicy(cfg).to(dev).train()
opt = torch.optim.AdamW(policy.get_optim_params(), lr=a.lr, weight_decay=cfg.optimizer_weight_decay)

dl = DataLoader(Chunks(train_eps, stats), batch_size=a.batch_size, shuffle=True, num_workers=a.workers,
                drop_last=True, pin_memory=True, persistent_workers=True)
vdl = DataLoader(Chunks(val_eps, stats), batch_size=a.batch_size, shuffle=True, num_workers=2, drop_last=True)


@torch.no_grad()
def validate(n_batches=50):
    policy.eval()
    tot, cnt = 0.0, 0
    for j, batch in enumerate(vdl):
        if j >= n_batches:
            break
        b = prep(batch, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = policy.predict_action_chunk({k: v for k, v in b.items() if k.startswith("observation")})
        l1 = (pred.float() - b["action"]).abs().mean(-1)
        tot += (l1 * ~b["action_is_pad"]).sum().item(); cnt += (~b["action_is_pad"]).sum().item()
    policy.train()
    return tot / max(cnt, 1)


step, t0 = 0, time.time()
while step < a.max_steps:
    for batch in dl:
        b = prep(batch, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, info = policy(b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        opt.step()
        step += 1
        if step % 100 == 0:
            print(f"step {step} loss {loss.item():.4f} {info} {(time.time()-t0)/step:.2f}s/it", flush=True)
        if step % a.val_freq == 0:
            print(f"step {step} VAL normalized L1 {validate():.4f}", flush=True)
        if step % a.save_freq == 0 or step == a.max_steps:
            policy.save_pretrained(f"{a.out_dir}/step_{step:06d}")
            print("saved", step, flush=True)
        if step >= a.max_steps:
            break
