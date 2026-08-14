"""corruptions.py — domain-shift-style image corruption for palmprint JEPA pretraining.
Dataset-agnostic: works on any (B,3,H,W) normalized tensor.
"""
import torch
import torchvision.transforms.functional as TF

CORRUPTION_NAMES = ["color_temp", "gamma", "channel_mix", "desaturate", "blur", "noise", "vignette"]


def _denorm(x, mean, std):
    mean = x.new_tensor(mean).view(1, -1, 1, 1)
    std = x.new_tensor(std).view(1, -1, 1, 1)
    return (x * std + mean).clamp(0.0, 1.0)


def _renorm(x, mean, std):
    mean = x.new_tensor(mean).view(1, -1, 1, 1)
    std = x.new_tensor(std).view(1, -1, 1, 1)
    return (x - mean) / std


def _apply_color_temp(xs, mask, s):
    shift = (torch.rand(xs.size(0), 1, 1, 1, device=xs.device) * 2 - 1) * s
    r = (xs[:, 0:1] * (1 + shift)).clamp(0, 1)
    b = (xs[:, 2:3] * (1 - shift)).clamp(0, 1)
    out = torch.cat([r, xs[:, 1:2], b], dim=1)
    return torch.where(mask.view(-1, 1, 1, 1), out, xs)


def _apply_gamma(xs, mask, s):
    gamma = (1.0 + (torch.rand(xs.size(0), 1, 1, 1, device=xs.device) * 2 - 1) * s).clamp(min=0.2)
    out = xs.clamp(min=1e-3).pow(gamma)
    return torch.where(mask.view(-1, 1, 1, 1), out, xs)


def _apply_channel_mix(xs, mask, s):
    B = xs.size(0)
    eye = torch.eye(3, device=xs.device).unsqueeze(0).expand(B, -1, -1)
    off = (torch.rand(B, 3, 3, device=xs.device) * 2 - 1) * s
    M = eye + off
    out = torch.einsum('bij,bjhw->bihw', M, xs).clamp(0, 1)
    return torch.where(mask.view(-1, 1, 1, 1), out, xs)


def _apply_desaturate(xs, mask, s):
    alpha = torch.rand(xs.size(0), 1, 1, 1, device=xs.device) * s
    gray = xs.mean(dim=1, keepdim=True).expand_as(xs)
    out = xs * (1 - alpha) + gray * alpha
    return torch.where(mask.view(-1, 1, 1, 1), out, xs)


def _apply_blur(xs, mask, max_sigma):
    out = xs.clone()
    idx = mask.nonzero(as_tuple=True)[0]
    for i in idx.tolist():
        sigma = float(torch.rand(1).item()) * max_sigma
        if sigma < 0.05:
            continue
        k = max(3, int(2 * round(3 * sigma) + 1))
        out[i:i + 1] = TF.gaussian_blur(xs[i:i + 1], kernel_size=k, sigma=sigma)
    return out


def _apply_noise(xs, mask, max_std):
    std = torch.rand(xs.size(0), 1, 1, 1, device=xs.device) * max_std
    out = (xs + std * torch.randn_like(xs)).clamp(0, 1)
    return torch.where(mask.view(-1, 1, 1, 1), out, xs)


def _apply_vignette(xs, mask, s):
    H, W = xs.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=xs.device),
        torch.linspace(-1, 1, W, device=xs.device), indexing="ij")
    r = torch.sqrt(xx ** 2 + yy ** 2)
    strength = torch.rand(xs.size(0), 1, 1, 1, device=xs.device) * s
    falloff = 1.0 - strength * r.unsqueeze(0).unsqueeze(0).clamp(0, 1)
    return torch.where(mask.view(-1, 1, 1, 1), (xs * falloff).clamp(0, 1), xs)


_FN = {
    "color_temp":  lambda xs, m, cfg: _apply_color_temp(xs, m, cfg.color_temp_strength),
    "gamma":       lambda xs, m, cfg: _apply_gamma(xs, m, cfg.gamma_strength),
    "channel_mix": lambda xs, m, cfg: _apply_channel_mix(xs, m, cfg.channel_mix_strength),
    "desaturate":  lambda xs, m, cfg: _apply_desaturate(xs, m, cfg.desaturate_strength),
    "blur":        lambda xs, m, cfg: _apply_blur(xs, m, cfg.blur_sigma_max),
    "noise":       lambda xs, m, cfg: _apply_noise(xs, m, cfg.corruption_std),
    "vignette":    lambda xs, m, cfg: _apply_vignette(xs, m, cfg.vignette_strength),
}


def corrupt_images(images, cfg, mean, std):
    """
    images: (B,3,H,W) normalized tensor (context view, before context_encoder).
    cfg: argparse Namespace with corruption_prob, corruption_mode, mix_prob,
         and the 7 *_strength / corruption_std / blur_sigma_max fields.
    mean, std: normalization stats used by your CASIADataset transform
               (NOT a hardcoded dataset lookup — pass the real values in).
    """
    if getattr(cfg, "corruption_prob", 0.0) <= 0.0:
        return images

    B = images.size(0)
    device = images.device
    x = _denorm(images, mean, std)

    corrupt_mask = torch.rand(B, device=device) < cfg.corruption_prob
    if not corrupt_mask.any():
        return images

    n = len(CORRUPTION_NAMES)
    if getattr(cfg, "corruption_mode", "mixed") == "single":
        chosen = torch.randint(0, n, (B,), device=device)
        type_masks = {name: corrupt_mask & (chosen == i) for i, name in enumerate(CORRUPTION_NAMES)}
    else:
        include = torch.rand(B, n, device=device) < getattr(cfg, "mix_prob", 0.4)
        none_sel = ~include.any(dim=1)
        if none_sel.any():
            forced = torch.randint(0, n, (int(none_sel.sum()),), device=device)
            include[none_sel, forced] = True
        type_masks = {name: corrupt_mask & include[:, i] for i, name in enumerate(CORRUPTION_NAMES)}

    for name in CORRUPTION_NAMES:
        m = type_masks[name]
        if m.any():
            x = _FN[name](x, m, cfg)

    return _renorm(x, mean, std)