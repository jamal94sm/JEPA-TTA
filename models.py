"""
models.py — JEPA encoder + predictor (from JEPA codebase).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class StructureHead(nn.Module):
    """Maps embeddings (embed_dim) -> Gabor descriptor space (out_dim).

    Can be shared between A1 (fed ctx_embeds) and A2 (fed the predictor's
    structure output). Sharing forces both paths into a consistent
    structural space -- see Predictor.out_proj_struct / norm_struct_out for
    the normalization that makes sharing well-posed.
    """

    def __init__(self, embed_dim, out_dim, hidden=None):
        super().__init__()
        hidden = hidden or embed_dim
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z):
        return self.net(z)


class UncertaintyWeighting(nn.Module):
    """Kendall et al. homoscedastic task weighting.

    total = sum_i [ 0.5 * exp(-log_var_i) * L_i + 0.5 * log_var_i ]
    """

    def __init__(self, n_tasks):
        super().__init__()
        self.log_var = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses):
        total = 0.0
        for i, l in enumerate(losses):
            total = total + 0.5 * torch.exp(-self.log_var[i]) * l \
                    + 0.5 * self.log_var[i]
        return total

    def weights(self):
        with torch.no_grad():
            return [0.5 * float(torch.exp(-v)) for v in self.log_var]


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


class SupervisedViT(nn.Module):
    """JEPA's ViT encoder + a linear head, for supervised CE pretraining.
       backbone(x) -> [B, embed_dim]  (same contract as CompNetBackbone),
       forward(x)  -> (logits, feat)."""
    def __init__(self, img_size, num_patches, embed_dim, n_classes):
        super().__init__()
        enc = ContextEncoder((img_size, img_size), num_patches, embed_dim)
        self.backbone = FeatureExtractor(enc)   # wraps the encoder, pools -> [B,d]
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat), feat


class PlainViT(nn.Module):
    """Standard supervised ViT — no masking, no JEPA pipeline.
       backbone(x) -> [B, embed_dim]   (same contract as CompNetBackbone)
       forward(x)  -> (logits, feat)."""
    def __init__(self, img_size=112, patch_size=14, embed_dim=256,
                 depth=6, n_heads=8, n_classes=160, in_ch=3, mlp_ratio=4.0):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.grid = img_size // patch_size
        n_patches = self.grid * self.grid

        # patch embedding via a strided conv (standard ViT stem)
        self.patch_embed = nn.Conv2d(in_ch, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, n_patches + 1, embed_dim))    # +1 for CLS
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.0, activation="gelu",
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, n_classes)

    def backbone(self, x):
        B = x.size(0)
        z = self.patch_embed(x)                      # [B, D, g, g]
        z = z.flatten(2).transpose(1, 2)             # [B, n_patches, D]
        cls = self.cls_token.expand(B, -1, -1)       # [B, 1, D]
        z = torch.cat([cls, z], dim=1) + self.pos_embed
        z = self.encoder(z)
        z = self.norm(z)
        return z[:, 0]                               # CLS token -> [B, D]

    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat), feat


class FeatModule(nn.Module):
    """Wrap a model so forward(x) returns ONLY its feature — gives run_full_eval
       an object with .eval() and __call__ that yields [B, embed_dim]."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model.backbone(x)      # PlainViT.backbone(x) -> [B, D]


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
    """Shared predictor with task tokens.

    forward(..., predict_structure=False) -> appearance preds  [B*, N_tgt, embed_dim]
    forward(..., predict_structure=True)  -> (appearance, structure), both
                                             [B*, N_tgt, embed_dim]

    Task token 0 = appearance (JEPA target space, via out_proj).
    Task token 1 = structure  (Gabor descriptor space, via out_proj_struct,
                   which projects back to embed_dim so StructureHead can be
                   shared with the A1 path). When norm_struct_out=True, a
                   LayerNorm is appended to out_proj_struct so its output
                   distribution matches ctx_embeds (which passes through
                   ContextEncoder's own LayerNorm) -- this matters whenever
                   A1 and A2 share one StructureHead, since an unnormalized
                   projection would feed the shared head two differently-
                   scaled input distributions.

    Both query the SAME target positions in one forward pass, so the shared
    trunk learns spatial extrapolation once and only the output projection
    differs per task.
    """

    def __init__(
        self,
        num_patches,
        embed_dim,
        pred_dim=None,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        norm_struct_out=True,
    ):
        super().__init__()

        # Fixed internals (= recipe for default embed_dim=256); I/O still uses embed_dim.
        pred_dim = 128
        depth = 6
        num_heads = 2

        # --- dimensionality change ---
        self.in_proj  = nn.Linear(embed_dim, pred_dim)
        self.out_proj = nn.Linear(pred_dim, embed_dim)

        if norm_struct_out:
            self.out_proj_struct = nn.Sequential(
                nn.Linear(pred_dim, embed_dim),
                nn.LayerNorm(embed_dim),
            )
        else:
            self.out_proj_struct = nn.Linear(pred_dim, embed_dim)

        # --- mask token ---
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))

        # --- task tokens: [0] = appearance, [1] = structure ---
        self.task_embed = nn.Parameter(torch.zeros(2, 1, 1, pred_dim))
        nn.init.trunc_normal_(self.task_embed, std=0.02)

        # --- positional embeddings ---
        pos = get_2d_sincos_pos_embed(pred_dim, num_patches)
        self.pos_embed = nn.Parameter(
            torch.tensor(pos).float().unsqueeze(0),  # (1, P, pred_dim)
            requires_grad=False,
        )

        # --- transformer ---
        enc = nn.TransformerEncoderLayer(
            d_model=pred_dim,
            nhead=num_heads,
            dim_feedforward=int(pred_dim * mlp_ratio),
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, depth)
        self.norm = nn.LayerNorm(pred_dim)

    def forward(self, context, context_masks, target_masks,
                predict_structure=False):
        """
        Parameters
        ----------
        context        : (B * N_ctx_masks, N_ctx, D)
        context_masks  : list[(B, N_ctx)] index tensors
        target_masks   : list[(B, N_tgt)] index tensors
        predict_structure : bool
            If True, also queries the structure task token at the same
            target positions and returns (appearance_preds, structure_preds).

        Returns
        -------
        preds : (B * N_ctx_masks * N_target_blocks, N_tgt, D)   [predict_structure=False]
        (appear, struct) : same shape each                       [predict_structure=True]
        """
        if not isinstance(context_masks, list):
            context_masks = [context_masks]
        if not isinstance(target_masks, list):
            target_masks = [target_masks]

        n_ctx = len(context_masks)
        n_tgt = len(target_masks)

        B = context.size(0) // n_ctx
        N_tgt = target_masks[0].size(1)


        # 1. Project context embeddings
        # --------------------------------------------------
        x = self.in_proj(context)  # (B*n_ctx, N_ctx, pred_dim)


        # 2. Add positional embeddings to context tokens
        # --------------------------------------------------
        pos_full = self.pos_embed.expand(B, -1, -1)  # (B, P, pred_dim)
        pos_ctx = torch.cat( [_gather(pos_full, m) for m in context_masks], dim=0)  # (B*n_ctx, N_ctx, pred_dim)

        x = x + pos_ctx


        # 3. Build target mask tokens with position + task info
        # --------------------------------------------------
        pos_tgt = torch.cat( [_gather(pos_full, m) for m in target_masks], dim=0)  # (B*n_tgt, N_tgt, pred_dim)

        base_tokens = self.mask_token.expand(pos_tgt.size(0), N_tgt, -1) + pos_tgt
        mask_appear = base_tokens + self.task_embed[0]

        # --------------------------------------------------
        # 4. Pair each context with each target block
        # --------------------------------------------------
        x = x.repeat(n_tgt, 1, 1)  # (B*n_ctx*n_tgt, N_ctx, pred_dim)

        # --------------------------------------------------
        # 5. Concatenate + transformer
        # --------------------------------------------------
        if predict_structure:
            mask_struct = base_tokens + self.task_embed[1]
            x = torch.cat([x, mask_appear, mask_struct], dim=1)
        else:
            x = torch.cat([x, mask_appear], dim=1)

        x = self.encoder(x)
        x = self.norm(x)

        # --------------------------------------------------
        # 6. Extract predictions (mask tokens only)
        # --------------------------------------------------
        if predict_structure:
            appear = self.out_proj(x[:, -2 * N_tgt:-N_tgt])
            struct = self.out_proj_struct(x[:, -N_tgt:])
            return appear, struct

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
