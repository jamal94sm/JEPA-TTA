"""hog.py — dense HOG-style orientation-magnitude maps.

Drop-in alternative to GaborBank: forward(x) -> (B, K, H, W), so the existing
patch_energy_descriptor (patch pooling + per-patch L2 norm) turns these into
per-patch HOG descriptors with local contrast normalization -- the ingredient
MaskFeat found essential.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HOGBank(nn.Module):
    """Soft-binned gradient orientation histograms, optionally multi-scale.

    K = n_bins * n_scales * (3 if per_channel else 1)
    """

    def __init__(self, n_bins=9, blur_sigmas=(0.0, 1.0, 2.0), per_channel=False):
        super().__init__()
        self.n_bins = n_bins
        self.blur_sigmas = tuple(blur_sigmas)
        self.n_scales = len(self.blur_sigmas)
        self.per_channel = per_channel
        self.K = n_bins * self.n_scales * (3 if per_channel else 1)

        # Simple central-difference gradient filters (zero-DC by construction).
        gx = torch.tensor([[0., 0., 0.], [-1., 0., 1.], [0., 0., 0.]])
        gy = torch.tensor([[0., -1., 0.], [0., 0., 0.], [0., 1., 0.]])
        self.register_buffer("gx", gx.view(1, 1, 3, 3))
        self.register_buffer("gy", gy.view(1, 1, 3, 3))

    def _hist_one_scale(self, gray, sigma):
        """gray: (B, 1, H, W) -> (B, n_bins, H, W)"""
        if sigma > 0.05:
            k = max(3, int(2 * round(3 * sigma) + 1))
            gray = _gaussian_blur(gray, k, sigma)

        dx = F.conv2d(gray, self.gx, padding=1)
        dy = F.conv2d(gray, self.gy, padding=1)
        mag = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8).squeeze(1)      # (B,H,W)
        # Unsigned orientation in [0, pi) -- matches Gabor's theta = pi*o/n.
        ang = torch.atan2(dy, dx).squeeze(1) % math.pi              # (B,H,W)

        # Soft (linearly interpolated) binning between adjacent orientation bins.
        pos = ang / math.pi * self.n_bins
        lo = torch.floor(pos)
        w_hi = (pos - lo)
        w_lo = 1.0 - w_hi
        lo = (lo.long() % self.n_bins)
        hi = (lo + 1) % self.n_bins

        B, H, W = mag.shape
        out = torch.zeros(B, self.n_bins, H, W, device=mag.device, dtype=mag.dtype)
        out.scatter_add_(1, lo.unsqueeze(1), (mag * w_lo).unsqueeze(1))
        out.scatter_add_(1, hi.unsqueeze(1), (mag * w_hi).unsqueeze(1))
        return out

    @torch.no_grad()
    def forward(self, x):
        """x: (B, 3, H, W) normalized image -> (B, K, H, W) non-negative."""
        if self.per_channel:
            per_ch = []
            for c in range(3):
                ch = x[:, c:c + 1]
                per_ch.append(torch.cat(
                    [self._hist_one_scale(ch, s) for s in self.blur_sigmas], dim=1))
            return torch.cat(per_ch, dim=1)
        gray = x.mean(dim=1, keepdim=True)
        return torch.cat(
            [self._hist_one_scale(gray, s) for s in self.blur_sigmas], dim=1)


def _gaussian_blur(x, ksize, sigma):
    half = ksize // 2
    t = torch.arange(-half, half + 1, dtype=x.dtype, device=x.device)
    g = torch.exp(-(t ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    x = F.conv2d(x, g.view(1, 1, 1, -1), padding=(0, half))
    return F.conv2d(x, g.view(1, 1, -1, 1), padding=(half, 0))