from Utils import *



##############################################################################################
##############################################################################################


import torch
import math


def Patchify(
    image_shape,
    num_blocks=4,
    num_patches=14,
    trg_ratio=(0.15, 0.20),
    ctx_ratio=(0.85, 1.00),
    ar_range=(0.75, 1.5),
    device="cpu",):
    
    B, _, _, _ = image_shape
    H = W = num_patches
    P = H * W

    def sample_block(scale):
        s  = torch.empty(()).uniform_(*scale).item()
        ar = torch.empty(()).uniform_(*ar_range).item()
        area = max(1, int(s * P))
        h = max(1, min(H, int(round(math.sqrt(area * ar)))))
        w = max(1, min(W, int(round(area / ar))))
        y = torch.randint(0, H - h + 1, ())
        x = torch.randint(0, W - w + 1, ())
        idx = [(y+i)*W + (x+j) for i in range(h) for j in range(w)]
        return torch.tensor(idx, device=device)

    ctx_masks = []                           # per-sample context indices
    tgt_masks = [[] for _ in range(num_blocks)]  # per-block list of per-sample indices

    min_ctx = P
    min_tgt = P  # GLOBAL min across all blocks and samples

    for _ in range(B):
        occupied = torch.zeros(P, dtype=torch.bool, device=device)

        # --- target blocks ---
        for k in range(num_blocks):
            idx = sample_block(trg_ratio)
            tgt_masks[k].append(idx)
            occupied[idx] = True
            min_tgt = min(min_tgt, idx.numel())

        # --- context block (minus targets; fallback to complement) ---
        for _ in range(10):
            ctx = sample_block(ctx_ratio)
            ctx = ctx[~occupied[ctx]]
            if ctx.numel() > 0:
                break
        else:
            ctx = (~occupied).nonzero().squeeze(1)

        min_ctx = min(min_ctx, ctx.numel())
        ctx_masks.append(ctx)

    # --- unbiased random trimming ---
    ctx_out = torch.stack([
        c[torch.randperm(c.numel(), device=device)[:min_ctx]]
        for c in ctx_masks
    ])  # (B, N_ctx)

    tgt_out = [
        torch.stack([
            t[torch.randperm(t.numel(), device=device)[:min_tgt]]
            for t in tgt_masks[k]
        ])  # (B, N_tgt) — SAME N_tgt for all blocks
        for k in range(num_blocks)
    ]

    return [ctx_out], tgt_out

def _repeat_interleave_batch(x, B, repeat):
    """
    Tile x so each group of B rows is repeated `repeat` times.

    Parameters
    ----------
    x      : Tensor  (B * num_blocks, N, D)
    B      : int     original batch size
    repeat : int     number of context masks (usually 1)

    Returns
    -------
    Tensor  (B * num_blocks * repeat, N, D)
    """
    N, D  = x.size(1), x.size(2)
    num_blocks = x.size(0) // B
    x = x.view(B, num_blocks, N, D)                              # (B, M, N, D)
    x = x.unsqueeze(1).expand(-1, repeat, -1, -1, -1)           # (B, repeat, M, N, D)
    return x.reshape(B * repeat * num_blocks, N, D)


def apply_masks(x, masks):
    """
    x     : Tensor  [B, P, D]
    masks : list[Tensor]  each entry shape (B, N) 

    Returns
    -------
    Tensor  [B * len(masks), N, D]
    """
    if not isinstance(masks, list):
        masks = [masks]
    out = []
    for m in masks:
        B, N = m.shape
        D    = x.size(-1)
        idx  = m.unsqueeze(-1).expand(B, N, D)        # (B, N, D)
        out.append(torch.gather(x, dim=1, index=idx)) # (B, N, D)
    return torch.cat(out, dim=0)   # (B * len(masks), N, D)

def _gather(x, mask):
    """
    x    : Tensor (B, P, D)
    mask : Tensor (B, N)  -- integer indices

    returns: (B, N, D)
    """
    B, N = mask.shape
    D = x.size(-1)
    idx = mask.unsqueeze(-1).expand(B, N, D)
    return torch.gather(x, dim=1, index=idx)



##############################################################################################
##############################################################################################

def plot_context_and_targets(img, context_mask, target_masks, patch_size, img_size,
                             mean=None, std=None, titles=("Context","T1","T2","T3","T4")):
    if isinstance(img, np.ndarray): img = torch.from_numpy(img)
    img = img.detach().float()
    if img.ndim == 2: img = img.unsqueeze(0)
    if img.ndim == 3 and img.shape[0] not in (1,3,4) and img.shape[-1] in (1,3,4):
        img = img.permute(2,0,1)
    C,H,W = img.shape

    def _to_pixel_mask(m):
        if isinstance(m, np.ndarray): m = torch.from_numpy(m)
        m = m.detach()
        if m.dtype not in (torch.bool, torch.uint8, torch.int8, torch.int16,
                           torch.int32, torch.int64, torch.float32, torch.float64):
            m = m.bool()
        if m.dtype.is_floating_point: m = m > 0.5
        if m.ndim == 1:
            P = m.numel()
            if P == H*W:
                m = m.view(H, W)
            else:
                Hp = Wp = img_size // patch_size
                assert P == Hp*Wp, f"mask length {P} != Hp*Wp {Hp*Wp}"
                m = (m.view(Hp, Wp)
                      .repeat_interleave(patch_size, 0)
                      .repeat_interleave(patch_size, 1))
                m = m[:H, :W]
        elif m.ndim == 2:
            pass
        elif m.ndim == 3 and m.shape[-1] == 1:
            m = m.squeeze(-1)
        elif m.ndim == 3 and m.shape[0] == 1:
            m = m[0]
        else:
            raise ValueError("Unsupported mask shape.")
        return m.bool()

    cm = _to_pixel_mask(context_mask)
    if isinstance(target_masks, np.ndarray):
        target_masks = torch.from_numpy(target_masks)
    if target_masks.ndim in (2, 3):
        tms = [_to_pixel_mask(target_masks[i]) for i in range(target_masks.shape[0])]
    else:
        raise ValueError("target_masks must be (4,P) or (4,H,W).")

    img_disp = img.clone()
    if mean is not None and std is not None:
        mean = torch.tensor(mean, device=img_disp.device).view(-1,1,1)
        std  = torch.tensor(std,  device=img_disp.device).view(-1,1,1)
        img_disp = img_disp * std + mean
    else:
        mn = img_disp.amin(dim=(1,2), keepdim=True)
        mx = img_disp.amax(dim=(1,2), keepdim=True)
        img_disp = (img_disp - mn) / (mx - mn + 1e-6)

    def _apply(mask):
        out = img_disp.clone()
        out[:, ~mask] = 0.0
        x = out.permute(1,2,0).cpu().numpy()
        return x[...,0] if x.shape[-1] == 1 else x

    imgs = [_apply(cm)] + [_apply(m) for m in tms[:4]]
    plt.figure(figsize=(15,3))
    for i, (im, title) in enumerate(zip(imgs, titles)):
        ax = plt.subplot(1, 5, i+1)
        ax.imshow(im, cmap=('gray' if im.ndim==2 else None), vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("my_plot.png", dpi=200)
    plt.close()



##############################################################################################
##############################################################################################


@torch.no_grad()
def update_ema(context_encoder, target_encoder, momentum):
    for pc, pt in zip(context_encoder.parameters(), target_encoder.parameters()):
        pt.data.mul_(momentum).add_(pc.data * (1.0 - momentum))


##############################################################################################
##############################################################################################

import os
import torch
import torch.nn as nn
from . import MyModels
import torch.optim as optim


def load_frozen_context_encoder(ckpt_path, args):
    enc = MyModels.Context_Encoder(
        image_size=(args.image_size[0], args.image_size[1]),
        num_patches=args.num_patches,
        embed_dim=args.embed_dim,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    enc.load_state_dict(ckpt["models"]["context"], strict=True)
    enc.to(args.device).eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc


def resolve_ckpt_path(folder_name, args):
    ckpt_base = os.path.join(folder_name, "checkpoints")
    if getattr(args, "resume", None):
        return args.resume
    runs = [d for d in os.listdir(ckpt_base)
            if os.path.isdir(os.path.join(ckpt_base, d))]
    run = max(runs, key=lambda d: os.path.getmtime(os.path.join(ckpt_base, d)))
    return os.path.join(ckpt_base, run, "last.ckpt")


def linear_probe(feature_extractor, train_loader, test_loader,
                 num_classes, lr, epochs, device):
    with torch.no_grad():
        x, _ = next(iter(train_loader))
        feat_dim = feature_extractor(x.to(device)).shape[-1]

    clf     = torch.nn.Linear(feat_dim, num_classes).to(device)
    opt     = torch.optim.Adam(clf.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for _ in range(epochs):
        clf.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                feats = feature_extractor(x)
            loss = loss_fn(clf(feats), y)
            opt.zero_grad(); loss.backward(); opt.step()

    clf.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            feats = feature_extractor(x.to(device))
            pred  = clf(feats).argmax(1).cpu()
            correct += (pred == y).sum().item()
            total   += y.size(0)
    return correct / total


##############################################################################################
##############################################################################################


@torch.no_grad()
def build_feature_bank(feature_extractor, dataloader, device):
    features, labels = [], []
    for x, y in dataloader:
        z = feature_extractor(x.to(device))
        z = torch.nn.functional.normalize(z, dim=1)
        features.append(z.cpu())
        labels.append(y)
    return torch.cat(features), torch.cat(labels)


@torch.no_grad()
def knn_evaluate(feature_extractor, train_loader, test_loader,
                 k, num_classes, device, temperature=0.07):

    feat_train, labels_train = build_feature_bank(
        feature_extractor, train_loader, device)

    correct = total = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        z    = torch.nn.functional.normalize(feature_extractor(x), dim=1).cpu()
        sim  = z @ feat_train.T

        topk_sim, topk_idx = sim.topk(k, dim=1)
        topk_labels        = labels_train[topk_idx]
        weights            = torch.exp(topk_sim / temperature)

        scores = torch.zeros(z.size(0), num_classes)
        for c in range(num_classes):
            scores[:, c] = (weights * (topk_labels == c)).sum(dim=1)

        correct += (scores.argmax(1) == y.cpu()).sum().item()
        total   += y.size(0)

    return correct / total

##############################################################################################
##############################################################################################


class WarmupCosineSchedule:
    def __init__(self, optimizer, warmup_steps, start_lr, ref_lr, total_steps, final_lr=0.0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.T_max = total_steps - warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        if self.step_num < self.warmup_steps:
            p = self.step_num / max(1, self.warmup_steps)
            lr = self.start_lr + p * (self.ref_lr - self.start_lr)
        else:
            p = (self.step_num - self.warmup_steps) / max(1, self.T_max)
            lr = self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1 + math.cos(math.pi * p))

        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return lr
    

class CosineWDSchedule:
    def __init__(self, optimizer, ref_wd, total_steps, final_wd=0.0):
        self.optimizer = optimizer
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self.total_steps = total_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        p = self.step_num / self.total_steps
        wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (1 + math.cos(math.pi * p))

        for g in self.optimizer.param_groups:
            g["weight_decay"] = wd
        return wd
    

##############################################################################################
##############################################################################################

