"""Stage 2 (env: lerobot): train lerobot Diffusion Policy on the dumped 3-camera Rizon4 data (see dump_rlds_3cam.py).

Cameras (order matters at deployment): image (primary/side), secondary (scene), wrist. State/action 8-D
(7 joints + gripper, absolute). Normalization stats are saved to <out>/stats.json and MUST be applied identically
at deployment (state/action mean-std, images /255 then ImageNet mean/std). Saved checkpoints hold the EMA weights.
Sampling window follows lerobot: obs frames t-(n_obs-1)..t, action frames t-(n_obs-1)..t-(n_obs-1)+horizon-1;
indices outside the episode are clamped, and out-of-range actions are masked from the loss.
At inference the executed actions are chunk[n_obs-1 : n_obs-1+n_action_steps].
"""
import argparse, copy, glob, json, math, os, time
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

p = argparse.ArgumentParser()
p.add_argument("--data_dir", required=True)
p.add_argument("--out_dir", required=True)
p.add_argument("--n_obs_steps", type=int, default=2)
p.add_argument("--horizon", type=int, default=64)
p.add_argument("--n_action_steps", type=int, default=32)
p.add_argument("--batch_size", type=int, default=64)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--max_steps", type=int, default=100000)
p.add_argument("--val_freq", type=int, default=5000)
p.add_argument("--val_batches", type=int, default=30)
p.add_argument("--save_freq", type=int, default=10000)
p.add_argument("--workers", type=int, default=8)
p.add_argument("--crop", type=int, default=224)
p.add_argument("--infer_steps", type=int, default=10, help="DDIM steps at inference/validation")
p.add_argument("--ema_max", type=float, default=0.9999)
a = p.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
IMG_KEYS = {"image": "observation.images.image", "secondary": "observation.images.secondary",
            "wrist": "observation.images.wrist"}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 1, 3, 1, 1)
NO, H = a.n_obs_steps, a.horizon


def load_split(split):
    eps = sorted({f.rsplit("_", 1)[0] for f in glob.glob(f"{a.data_dir}/{split}/ep*_state.npy")})
    return [{k: np.load(f"{e}_{k}.npy", mmap_mode="r") for k in (*IMG_KEYS, "state", "action")} for e in eps]


class Windows(Dataset):
    def __init__(self, eps, stats):
        self.eps, self.s = eps, stats
        self.idx = [(e, t) for e, ep in enumerate(eps) for t in range(len(ep["state"]))]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        e, t = self.idx[i]
        ep, T = self.eps[e], len(self.eps[e]["state"])
        oi = np.clip(np.arange(t - NO + 1, t + 1), 0, T - 1)
        ai_raw = np.arange(t - NO + 1, t - NO + 1 + H)
        ai = np.clip(ai_raw, 0, T - 1)
        state = (np.asarray(ep["state"])[oi] - self.s["state_mean"]) / self.s["state_std"]
        act = (np.asarray(ep["action"])[ai] - self.s["action_mean"]) / self.s["action_std"]
        out = {
            OBS_STATE: torch.from_numpy(state).float(),                      # (NO, 8)
            "action": torch.from_numpy(act).float(),                         # (H, 8)
            "action_is_pad": torch.from_numpy((ai_raw < 0) | (ai_raw >= T)),  # (H,)
            # (NO, 3 cams, 3, 256, 256) uint8
            OBS_IMAGES: torch.from_numpy(np.stack([np.stack([ep[k][j] for k in IMG_KEYS]) for j in oi])).permute(0, 1, 4, 2, 3),
        }
        return out


def prep(batch, dev):
    b = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
    b[OBS_IMAGES] = (b[OBS_IMAGES].float() / 255 - MEAN.to(dev)) / STD.to(dev)
    return b


train_eps, val_eps = load_split("train"), load_split("val")
S = np.concatenate([ep["state"] for ep in train_eps]); A = np.concatenate([ep["action"] for ep in train_eps])
stats = {"state_mean": S.mean(0), "state_std": S.std(0) + 1e-6, "action_mean": A.mean(0), "action_std": A.std(0) + 1e-6}
json.dump({k: v.tolist() for k, v in stats.items()} | {k: getattr(a, k) for k in ("n_obs_steps", "horizon", "n_action_steps", "crop", "infer_steps")}
          | {"img_keys": IMG_KEYS}, open(f"{a.out_dir}/stats.json", "w"))
print(f"train eps {len(train_eps)}, val eps {len(val_eps)}", flush=True)

cfg = DiffusionConfig(
    input_features={OBS_STATE: PolicyFeature(FeatureType.STATE, (8,)),
                    **{key: PolicyFeature(FeatureType.VISUAL, (3, 256, 256)) for key in IMG_KEYS.values()}},
    output_features={"action": PolicyFeature(FeatureType.ACTION, (8,))},
    n_obs_steps=NO, horizon=H, n_action_steps=a.n_action_steps, drop_n_last_frames=H - a.n_action_steps - NO + 1,
    crop_shape=(a.crop, a.crop), crop_is_random=True, use_separate_rgb_encoder_per_camera=True,
    noise_scheduler_type="DDIM", num_inference_steps=a.infer_steps, do_mask_loss_for_padding=True,
    optimizer_lr=a.lr, push_to_hub=False, device="cuda")
dev = torch.device("cuda")
policy = DiffusionPolicy(cfg).to(dev).train()
ema = copy.deepcopy(policy).eval().requires_grad_(False)
print(f"params {sum(p.numel() for p in policy.parameters())/1e6:.1f}M", flush=True)
opt = torch.optim.AdamW(policy.get_optim_params(), lr=a.lr, betas=cfg.optimizer_betas, eps=cfg.optimizer_eps,
                        weight_decay=cfg.optimizer_weight_decay)
warm = cfg.scheduler_warmup_steps
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, a.max_steps) / a.max_steps)))

dl = DataLoader(Windows(train_eps, stats), batch_size=a.batch_size, shuffle=True, num_workers=a.workers,
                drop_last=True, pin_memory=True, persistent_workers=True)
vdl = DataLoader(Windows(val_eps, stats), batch_size=a.batch_size, shuffle=True, num_workers=2, drop_last=True)


@torch.no_grad()
def ema_update(step):
    d = min(a.ema_max, (1 + step) / (10 + step))
    for pe, pm in zip(ema.parameters(), policy.parameters()):
        pe.lerp_(pm, 1 - d)
    for be, bm in zip(ema.buffers(), policy.buffers()):
        be.copy_(bm)


@torch.no_grad()
def validate():
    """Full DDIM sampling with the EMA model; normalized L1 over the executed part of the chunk."""
    s0, s1 = NO - 1, NO - 1 + a.n_action_steps
    tot, cnt = 0.0, 0
    for j, batch in enumerate(vdl):
        if j >= a.val_batches:
            break
        b = prep(batch, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = ema.diffusion.generate_actions({OBS_STATE: b[OBS_STATE], OBS_IMAGES: b[OBS_IMAGES]})
        l1 = (pred.float() - b["action"][:, s0:s1]).abs().mean(-1)
        ok = ~b["action_is_pad"][:, s0:s1]
        tot += (l1 * ok).sum().item(); cnt += ok.sum().item()
    return tot / max(cnt, 1)


step, t0 = 0, time.time()
while step < a.max_steps:
    for batch in dl:
        b = prep(batch, dev)
        for i, key in enumerate(IMG_KEYS.values()):  # forward() stacks the per-camera keys itself
            b[key] = b[OBS_IMAGES][:, :, i]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = policy(b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        opt.step(); sched.step(); ema_update(step)
        step += 1
        if step % 100 == 0:
            print(f"step {step} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} {(time.time()-t0)/step:.2f}s/it", flush=True)
        if step % a.val_freq == 0 or step == a.max_steps:
            print(f"step {step} VAL(EMA) normalized L1 {validate():.4f}", flush=True)
        if step % a.save_freq == 0 or step == a.max_steps:
            ema.save_pretrained(f"{a.out_dir}/step_{step:06d}")
            torch.save({"model": policy.state_dict(), "opt": opt.state_dict(), "step": step}, f"{a.out_dir}/raw_latest.pt")
            print("saved", step, flush=True)
        if step >= a.max_steps:
            break
