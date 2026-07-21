"""
models.py — JEPA encoder + predictor (from JEPA codebase).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def get_1d_sincos_pos_embed(embed_dim, pos):
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / (10000 ** omega)
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    emb_h = get_1d_sincos_pos_embed(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def _gather(x, mask):
    B, N = mask.shape
    D = x.size(-1)
    idx = mask.unsqueeze(-1).expand(B, N, D)
    return torch.gather(x, dim=1, index=idx)


# ══════════════════════════════════════════════════════════════
#  Context Encoder
# ══════════════════════════════════════════════════════════════

class ContextEncoder(nn.Module):
    def __init__(self, image_size, num_patches, embed_dim,
                 depth=None, num_heads=None, mlp_ratio=4.0):
        super().__init__()
        H, W = image_size
        patch_h = H // num_patches
        patch_w = W // num_patches

        if num_heads is None:
            num_heads = max(4, embed_dim // 32)
        if depth is None:
            depth = min(6, embed_dim // 64 + 2)

        self.proj = nn.Conv2d(3, embed_dim,
                               kernel_size=(patch_h, patch_w),
                               stride=(patch_h, patch_w))

        pos = get_2d_sincos_pos_embed(embed_dim, num_patches)
        self.pos_embed = nn.Parameter(
            torch.tensor(pos).float().unsqueeze(0), requires_grad=False)

        enc = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, masks):
        z = self.proj(x).flatten(2).transpose(1, 2)
        z = z + self.pos_embed
        outs = []
        for m in masks:
            idx = m.unsqueeze(-1).expand(-1, -1, z.size(-1))
            outs.append(torch.gather(z, 1, idx))
        visible = torch.cat(outs, dim=0)
        z = self.encoder(visible)
        z = self.norm(z)
        return z


# ══════════════════════════════════════════════════════════════
#  Target Encoder
# ══════════════════════════════════════════════════════════════

class TargetEncoder(nn.Module):
    def __init__(self, image_size, num_patches, embed_dim,
                 depth=None, num_heads=None, mlp_ratio=4.0):
        super().__init__()
        H, W = image_size
        patch_h = H // num_patches
        patch_w = W // num_patches

        if num_heads is None:
            num_heads = max(4, embed_dim // 32)
        if depth is None:
            depth = min(6, embed_dim // 64 + 2)

        self.proj = nn.Conv2d(3, embed_dim,
                               kernel_size=(patch_h, patch_w),
                               stride=(patch_h, patch_w))

        pos = get_2d_sincos_pos_embed(embed_dim, num_patches)
        self.pos_embed = nn.Parameter(
            torch.tensor(pos).float().unsqueeze(0), requires_grad=False)

        enc = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, depth)
        self.norm = nn.LayerNorm(embed_dim)

    @torch.no_grad()
    def forward(self, x):
        z = self.proj(x).flatten(2).transpose(1, 2)
        z = z + self.pos_embed
        z = self.encoder(z)
        z = self.norm(z)
        return z


# ══════════════════════════════════════════════════════════════
#  Predictor
# ══════════════════════════════════════════════════════════════

class Predictor(nn.Module):
    def __init__(self, num_patches, embed_dim, pred_dim=None, depth=None):
        super().__init__()
        if pred_dim is None:
            pred_dim = embed_dim // 2
        if depth is None:
            depth = min(4, pred_dim // 64 + 1)

        num_heads = max(1, pred_dim // 64)
        while pred_dim % num_heads != 0:
            num_heads -= 1

        self.in_proj = nn.Linear(embed_dim, pred_dim)
        self.out_proj = nn.Linear(pred_dim, embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))

        pos = get_2d_sincos_pos_embed(pred_dim, num_patches)
        self.pos_embed = nn.Parameter(
            torch.tensor(pos).float().unsqueeze(0), requires_grad=False)

        enc = nn.TransformerEncoderLayer(
            d_model=pred_dim, nhead=num_heads,
            dim_feedforward=int(pred_dim * 4),
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, depth)
        self.norm = nn.LayerNorm(pred_dim)

    def forward(self, context, context_masks, target_masks):
        if not isinstance(context_masks, list):
            context_masks = [context_masks]
        if not isinstance(target_masks, list):
            target_masks = [target_masks]

        n_ctx = len(context_masks)
        n_tgt = len(target_masks)
        B = context.size(0) // n_ctx
        N_tgt = target_masks[0].size(1)

        x = self.in_proj(context)

        pos_full = self.pos_embed.expand(B, -1, -1)
        pos_ctx = torch.cat(
            [_gather(pos_full, m) for m in context_masks], dim=0)
        x = x + pos_ctx

        pos_tgt = torch.cat(
            [_gather(pos_full, m) for m in target_masks], dim=0)
        mask_tokens = self.mask_token.expand(
            pos_tgt.size(0), N_tgt, -1) + pos_tgt

        x = x.repeat(n_tgt, 1, 1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = self.encoder(x)
        x = self.norm(x)

        preds = x[:, -N_tgt:]
        return self.out_proj(preds)


# ══════════════════════════════════════════════════════════════
#  Feature Extractor (for evaluation)
# ══════════════════════════════════════════════════════════════

class FeatureExtractor(nn.Module):
    """Wraps encoder for eval: full mask → mean pool → feature vector."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.encoder.eval()

    def forward(self, x):
        B = x.size(0)
        P = self.encoder.pos_embed.size(1)
        device = x.device
        full_mask = [torch.arange(P, device=device).unsqueeze(0).expand(B, -1)]
        with torch.no_grad():
            z = self.encoder(x, full_mask)
        return z.mean(dim=1)


# ══════════════════════════════════════════════════════════════
#  Utilities
# ══════════════════════════════════════════════════════════════

def patchify(batch_size, num_patches, num_blocks=2,
             trg_ratio=(0.10, 0.15), ctx_ratio=(0.90, 1.00),
             ar_range=(0.75, 1.5), device="cpu"):
    """Create context + target masks for JEPA."""
    H = W = num_patches
    P = H * W

    def sample_block(scale):
        s = torch.empty(()).uniform_(*scale).item()
        ar = torch.empty(()).uniform_(*ar_range).item()
        area = max(1, int(s * P))
        h = max(1, min(H, int(round(math.sqrt(area * ar)))))
        w = max(1, min(W, int(round(area / h))))
        y = torch.randint(0, max(1, H - h + 1), ())
        x = torch.randint(0, max(1, W - w + 1), ())
        idx = [(y+i)*W + (x+j) for i in range(h) for j in range(w)]
        return torch.tensor(idx, device=device)

    ctx_masks, tgt_masks = [], [[] for _ in range(num_blocks)]
    min_ctx, min_tgt = P, P

    for _ in range(batch_size):
        occupied = torch.zeros(P, dtype=torch.bool, device=device)
        for k in range(num_blocks):
            idx = sample_block(trg_ratio)
            tgt_masks[k].append(idx)
            occupied[idx] = True
            min_tgt = min(min_tgt, idx.numel())
        for _ in range(10):
            ctx = sample_block(ctx_ratio)
            ctx = ctx[~occupied[ctx]]
            if ctx.numel() > 0:
                break
        else:
            ctx = (~occupied).nonzero().squeeze(1)
        min_ctx = min(min_ctx, ctx.numel())
        ctx_masks.append(ctx)

    ctx_out = torch.stack([
        c[torch.randperm(c.numel(), device=device)[:min_ctx]]
        for c in ctx_masks])
    tgt_out = [
        torch.stack([
            t[torch.randperm(t.numel(), device=device)[:min_tgt]]
            for t in tgt_masks[k]])
        for k in range(num_blocks)]

    return [ctx_out], tgt_out


def apply_masks(x, masks):
    out = []
    for m in masks:
        out.append(_gather(x, m))
    return torch.cat(out, dim=0)


def repeat_interleave_batch(x, B, repeat):
    N, D = x.size(1), x.size(2)
    num_blocks = x.size(0) // B
    x = x.view(B, num_blocks, N, D)
    x = x.unsqueeze(1).expand(-1, repeat, -1, -1, -1)
    return x.reshape(B * repeat * num_blocks, N, D)


@torch.no_grad()
def update_ema(context_encoder, target_encoder, momentum):
    for pc, pt in zip(context_encoder.parameters(),
                      target_encoder.parameters()):
        pt.data.mul_(momentum).add_(pc.data * (1.0 - momentum))



# ══════════════════════════════════════════════════════════════
#  CompNet — competitive CNN backbone + supervised head
# ══════════════════════════════════════════════════════════════

class GaborConv2d(nn.Module):
    """Learnable Gabor-style competitive filters (CompNet's core block)."""
    def __init__(self, in_ch, out_ch, kernel=7):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class CompNetBackbone(nn.Module):
    """CNN backbone. forward(x) -> [B, embed_dim] feature (pre-classifier).
       Mirrors FeatureExtractor's output contract so all downstream code
       (evaluate, subspace analysis) works unchanged."""
    def __init__(self, embed_dim=256, base=16, in_ch=3):
        super().__init__()
        self.stem = GaborConv2d(in_ch, base, 7)
        self.block1 = GaborConv2d(base, base * 2, 5)
        self.block2 = GaborConv2d(base * 2, base * 4, 3)
        self.block3 = GaborConv2d(base * 4, base * 8, 3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(base * 8, embed_dim)

    def forward(self, x):
        x = F.max_pool2d(self.stem(x), 2)
        x = F.max_pool2d(self.block1(x), 2)
        x = F.max_pool2d(self.block2(x), 2)
        x = self.block3(x)
        x = self.pool(x).flatten(1)          # [B, base*8]
        return self.proj(x)                   # [B, embed_dim]


class CompNet(nn.Module):
    """Backbone + linear classifier for supervised CE pretraining."""
    def __init__(self, embed_dim, n_classes, base=16, in_ch=3):
        super().__init__()
        self.backbone = CompNetBackbone(embed_dim, base, in_ch)
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        feat = self.backbone(x)               # [B, embed_dim]
        return self.classifier(feat), feat
