import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from . import MyUtils


"""
    Encoders:
    embed_dim
    num_heads = max(4, embed_dim // 32)
    depth ≈ min(6, embed_dim // 64 + 2)

    Predictor:  predictor is lighter than encoders
    pred_dim = embed_dim / 4
    num_heads = max(1, pred_dim // 32)
    depth = min(4, pred_dim // 64 + 1)

    embed_dim = 128 → encoder (dim=128, heads=4, depth=4), predictor (dim=32, heads=1, depth=1)
    embed_dim = 256 → encoder (dim=256, heads=8, depth=6), predictor (dim=64, heads=2, depth=2)

    Mask convention (used everywhere in this file)
    -----------------------------------------------
    All masks are INTEGER index tensors of shape (B, N), produced by
    MyUtils.Patchify / _bool_to_index.  We never use boolean masks inside
    models because boolean indexing produces variable-length outputs that
    cannot be batched.  torch.gather is used throughout instead.
"""



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


##############################################################################################
##############################################################################################


class Context_Encoder(nn.Module):
    def __init__(self, image_size, num_patches, embed_dim,
                 depth=None, num_heads=None, mlp_ratio=4.0):
        super().__init__()

        H, W    = image_size
        patch_h = H // num_patches
        patch_w = W // num_patches

        if num_heads is None:
            num_heads = max(4, embed_dim // 32)
        if depth is None:
            depth = min(6, embed_dim // 64 + 2)

        self.proj = nn.Conv2d(
            3, embed_dim,
            kernel_size=(patch_h, patch_w),
            stride=(patch_h, patch_w),
        )

        pos = get_2d_sincos_pos_embed(embed_dim, num_patches)
        self.pos_embed = nn.Parameter(
            torch.tensor(pos).float().unsqueeze(0),   # (1, P, D)
            requires_grad=False,
        )

        enc = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, depth)
        self.norm     = nn.LayerNorm(embed_dim)
    def forward(self, x, masks):
        """
        x     : (B, 3, H, W)
        masks : list[Tensor] each (B, N_ctx)
        return: (B , N_ctx, D)
        """
        # ---- patchify full image ----
        z = self.proj(x).flatten(2).transpose(1, 2)  # (B, P, D)
        z = z + self.pos_embed                              # (B, P, D)


        # ---- gather visible tokens only ----
        outs = []
        for m in masks:
            idx = m.unsqueeze(-1).expand(-1, -1, z.size(-1))
            outs.append(torch.gather(z, 1, idx))

        visible_tokens = torch.cat(outs, dim=0)                          # (B*k, N_ctx, D)

        # ---- transformer sees only visible tokens ----
        z = self.encoder(visible_tokens)
        z = self.norm(z)
        return z


##############################################################################################

class Target_Encoder(nn.Module):
    def __init__(self, image_size, num_patches, embed_dim,
                 depth=None, num_heads=None, mlp_ratio=4.0):
        super().__init__()

        H, W    = image_size
        patch_h = H // num_patches
        patch_w = W // num_patches

        if num_heads is None:
            num_heads = max(4, embed_dim // 32)
        if depth is None:
            depth = min(6, embed_dim // 64 + 2)

        # --- same projection API as Context_Encoder ---
        self.proj = nn.Conv2d(
            3, embed_dim,
            kernel_size=(patch_h, patch_w),
            stride=(patch_h, patch_w),
        )

        # --- identical positional embedding construction ---
        pos = get_2d_sincos_pos_embed(embed_dim, num_patches)
        self.pos_embed = nn.Parameter(
            torch.tensor(pos).float().unsqueeze(0),   # (1, P, D)
            requires_grad=False,
        )

        enc = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, depth)
        self.norm = nn.LayerNorm(embed_dim)

    @torch.no_grad()
    def forward(self, x):
        """
        x : Tensor (B, 3, H, W)

        Returns
        -------
        full_embeddings : Tensor (B, P, D)
        """
        # ---- patchify full image ----
        z = self.proj(x).flatten(2).transpose(1, 2)   # (B, P, D)
        z = z + self.pos_embed                        # (B, P, D)

        # ---- full transformer (no masking) ----
        z = self.encoder(z)  # (B, P, D)
        z = self.norm(z)  # (B, P, D)
        return z

##############################################################################################

class Predictor(nn.Module):
    def __init__(
        self,
        num_patches,
        embed_dim,
        pred_dim=None,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
    ):
        super().__init__()

        if pred_dim is None:
            pred_dim = embed_dim // 2

        num_heads = max(1, pred_dim // 64)
        while pred_dim % num_heads != 0:
            num_heads -= 1



        # --- dimensionality change ---
        self.in_proj  = nn.Linear(embed_dim, pred_dim)
        self.out_proj = nn.Linear(pred_dim, embed_dim)

        # --- mask token ---
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))

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

    def forward(self, context, context_masks, target_masks):
        """
        Parameters
        ----------
        context        : (B * N_ctx_masks, N_ctx, D)
        context_masks  : list[(B, N_ctx)] index tensors
        target_masks   : list[(B, N_tgt)] index tensors

        Returns
        -------
        preds : (B * N_ctx_masks * N_target_blocks, N_tgt, D)
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
        pos_ctx = torch.cat( [MyUtils._gather(pos_full, m) for m in context_masks], dim=0)  # (B*n_ctx, N_ctx, pred_dim)

        x = x + pos_ctx


        # 3. Build target mask tokens with position info
        # --------------------------------------------------
        pos_tgt = torch.cat( [MyUtils._gather(pos_full, m) for m in target_masks], dim=0)  # (B*n_tgt, N_tgt, pred_dim)

        mask_tokens = self.mask_token.expand(pos_tgt.size(0), N_tgt, -1)
        mask_tokens = mask_tokens + pos_tgt

        # --------------------------------------------------
        # 4. Pair each context with each target block
        # --------------------------------------------------
        x = x.repeat(n_tgt, 1, 1)  # (B*n_ctx*n_tgt, N_ctx, pred_dim)

        # --------------------------------------------------
        # 5. Concatenate + transformer
        # --------------------------------------------------
        x = torch.cat([x, mask_tokens], dim=1)
        x = self.encoder(x)
        x = self.norm(x)

        # --------------------------------------------------
        # 6. Extract predictions (mask tokens only)
        # --------------------------------------------------
        preds = x[:, -N_tgt:]
        return self.out_proj(preds)

##############################################################################################


class FeatureExtractor(nn.Module):
    """
    Wraps a frozen Context_Encoder for evaluation.
    Passes a full-image index mask (all patches visible), then mean-pools.
    """
    def __init__(self, encoder, pool="mean"):
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()
        self.pool = pool

    def forward(self, x):
        B      = x.size(0)
        P      = self.encoder.pos_embed.size(1)
        device = x.device

        # full mask: every patch index, shape (B, P) long
        full_mask = [torch.arange(P, device=device).unsqueeze(0).expand(B, -1)]

        z = self.encoder(x, full_mask)   # (B, P, D)  — len(masks)=1 → no repeat

        if self.pool == "mean":
            return z.mean(dim=1)   # (B, D)
        return z