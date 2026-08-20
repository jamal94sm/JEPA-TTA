"""gabor.py — fixed Gabor filter bank producing per-patch line/orientation descriptors."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def gabor_kernel_pair(ksize, sigma, lambd, theta, gamma=0.25):
    """Even (cos) + odd (sin) quadrature pair. Both zero-DC, unit-norm."""
    half = ksize // 2
    y, x = torch.meshgrid(
        torch.arange(-half, half + 1, dtype=torch.float32),
        torch.arange(-half, half + 1, dtype=torch.float32),
        indexing="ij")
    xr = x * math.cos(theta) + y * math.sin(theta)
    yr = -x * math.sin(theta) + y * math.cos(theta)
    env = torch.exp(-(xr ** 2 + (gamma ** 2) * (yr ** 2)) / (2 * sigma ** 2))
    ke = env * torch.cos(2 * math.pi * xr / lambd)
    ko = env * torch.sin(2 * math.pi * xr / lambd)
    ke = ke - ke.mean()
    ko = ko - ko.mean()
    return (ke / (ke.norm() + 1e-8), ko / (ko.norm() + 1e-8))


def gabor_kernel(ksize, sigma, lambd, theta, gamma=0.5, psi=0.0):
    half = ksize // 2
    y, x = torch.meshgrid(
        torch.arange(-half, half + 1, dtype=torch.float32),
        torch.arange(-half, half + 1, dtype=torch.float32),
        indexing="ij",
    )
    xr = x * math.cos(theta) + y * math.sin(theta)
    yr = -x * math.sin(theta) + y * math.cos(theta)
    env = torch.exp(-(xr ** 2 + (gamma ** 2) * (yr ** 2)) / (2 * sigma ** 2))
    k = env * torch.cos(2 * math.pi * xr / lambd + psi)
    k = k - k.mean()                 # zero-DC -> invariant to uniform brightness
    k = k / (k.norm() + 1e-8)        # unit-norm -> comparable across scales
    return k


class GaborBank(nn.Module):
    def __init__(self, n_orient=12,
                 scales=((5, 1.5, 3.0), (9, 3.0, 6.0), (13, 4.5, 9.0)),
                 gamma=0.25, per_channel=False):
        super().__init__()
        self.n_orient, self.n_scales = n_orient, len(scales)
        self.per_channel = per_channel
        n_filters = n_orient * self.n_scales
        self.K = n_filters * (3 if per_channel else 1)   # descriptor width

        max_k = max(s[0] for s in scales)
        # Layout: [all even filters, all odd filters]
        bank = torch.zeros(2 * n_filters, 1, max_k, max_k)
        idx = 0
        for (ksize, sigma, lambd) in scales:
            for o in range(n_orient):
                theta = math.pi * o / n_orient
                ke, ko = gabor_kernel_pair(ksize, sigma, lambd, theta, gamma)
                pad = (max_k - ksize) // 2
                bank[idx,             0, pad:pad+ksize, pad:pad+ksize] = ke
                bank[idx + n_filters, 0, pad:pad+ksize, pad:pad+ksize] = ko
                idx += 1
        self.register_buffer("weight", bank)
        self.pad = max_k // 2
        self.n_filters = n_filters

    @torch.no_grad()
    def forward(self, x):
        """Returns quadrature ENERGY (B, K, H, W) -- already non-negative."""
        nf = self.n_filters
        if self.per_channel:
            w = self.weight.repeat(3, 1, 1, 1)
            r = F.conv2d(x, w, padding=self.pad, groups=3)   # (B, 6*nf, H, W)
            r = r.view(x.size(0), 3, 2, nf, *r.shape[-2:])
            e, o = r[:, :, 0], r[:, :, 1]                     # (B,3,nf,H,W)
            energy = torch.sqrt(e ** 2 + o ** 2 + 1e-8)
            return energy.reshape(x.size(0), 3 * nf, *r.shape[-2:])
        gray = x.mean(dim=1, keepdim=True)
        r = F.conv2d(gray, self.weight, padding=self.pad)     # (B, 2*nf, H, W)
        e, o = r[:, :nf], r[:, nf:]
        return torch.sqrt(e ** 2 + o ** 2 + 1e-8)


@torch.no_grad()
def patch_energy_descriptor(resp, num_patches, img_size=112, ctx_mult=2,
                             center=True, eps=1e-6):
    """Pool over a ctx_mult-patch-wide window, stride = 1 patch.
    ctx_mult=1 -> patch-local; ctx_mult=2 -> ~your old working setting."""
    patch = img_size // num_patches
    k = patch * ctx_mult
    pad = (k - patch) // 2
    pooled = F.avg_pool2d(resp, kernel_size=k, stride=patch, padding=pad)
    if pooled.shape[-1] != num_patches:                       # guard odd sizes
        pooled = F.adaptive_avg_pool2d(pooled, (num_patches, num_patches))
    desc = pooled.flatten(2).transpose(1, 2)                  # (B, P, K)
    if center:
        desc = desc - desc.mean(dim=(0, 1), keepdim=True)
    return desc / (desc.norm(dim=-1, keepdim=True) + eps)


@torch.no_grad()
def sanity_report(gabor_bank, images, num_patches):
    """One-time diagnostic. Returns a dict of numbers that must look sane."""
    resp = gabor_bank(images)
    desc = patch_energy_descriptor(resp, num_patches)
    B, P, K = desc.shape
    flat = desc.reshape(-1, K)
    # Cosine similarity between random descriptor pairs: if ~1.0, all patches
    # look identical and the target carries no information.
    i = torch.randperm(flat.size(0), device=flat.device)[:512]
    j = torch.randperm(flat.size(0), device=flat.device)[:512]
    pair_cos = F.cosine_similarity(flat[i], flat[j], dim=-1).mean().item()
    return {
        "K": K,
        "resp_absmean": resp.abs().mean().item(),
        "desc_shape": tuple(desc.shape),
        "desc_norm_mean": desc.norm(dim=-1).mean().item(),   # must be ~1.0
        "desc_dim_var": flat.var(dim=0).mean().item(),
        "desc_pair_cos": pair_cos,                            # want well below 1.0
    }
