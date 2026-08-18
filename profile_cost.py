"""
profile_cost.py — Component-wise computational cost of the proposed method
(corruption + Gabor A1/A2) versus baseline JEPA.

Reports, per component:
  - trainable parameters
  - buffer parameters (non-trainable, e.g. the Gabor bank)
  - forward MACs / FLOPs (analytic + optional measured via fvcore/thop)
  - measured wall-clock (forward, and forward+backward)
  - peak CUDA memory

Then aggregates into per-image and per-batch training cost, and prints a
final comparison table across configurations.

Usage
-----
    python profile_cost.py --data_dir /home/pai-ng/Jamal/CASIA-MS-ROI

    # match a real run's settings
    python profile_cost.py --data_dir ... --gabor_gray 0 --gabor_orient 8 \
        --batch_size 64 --img_size 112 --repeats 30

Optional (better FLOP counts for the transformer attention terms):
    pip install fvcore          # or: pip install thop

The script does NOT need the dataset -- it synthesises random tensors of the
right shape. --data_dir is only accepted so you can reuse the same cfg object.
"""

import argparse
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import (ContextEncoder, TargetEncoder, Predictor, StructureHead,
                    patchify, apply_masks, repeat_interleave_batch)
from gabor import GaborBank, patch_energy_descriptor
from struct_loss import structure_loss

try:
    from corruption import corrupt_images
    HAVE_CORRUPTION = True
except Exception:
    HAVE_CORRUPTION = False

try:
    from fvcore.nn import FlopCountAnalysis
    HAVE_FVCORE = True
except Exception:
    HAVE_FVCORE = False


# ══════════════════════════════════════════════════════════════
#  Analytic MAC estimates
# ══════════════════════════════════════════════════════════════

def macs_transformer_encoder(n_tokens, dim, depth, mlp_ratio=4.0):
    """Per-layer: QKV+out proj (4*d^2), attention (2*n*d), MLP (2*mlp*d^2)."""
    per_layer = (
        n_tokens * 4 * dim * dim          # qkv + output projection
        + 2 * n_tokens * n_tokens * dim   # attn scores + attn @ v
        + n_tokens * 2 * mlp_ratio * dim * dim
    )
    return int(depth * per_layer)


def macs_patch_embed(img_size, patch, dim, in_ch=3):
    n_patches = (img_size // patch) ** 2
    return int(n_patches * dim * in_ch * patch * patch)


def macs_linear(n_tokens, d_in, d_out):
    return int(n_tokens * d_in * d_out)


def macs_gabor_bank(img_size, K, ksize, per_channel):
    """Full-resolution depthwise-style conv: K filters of ksize^2 over HxW.
    Note GaborBank zero-pads every kernel to max_k, so ALL filters cost
    max_k^2 regardless of their nominal scale."""
    in_ch = 3 if per_channel else 1
    n_out_ch = K
    return int(n_out_ch * img_size * img_size * ksize * ksize * (1 if per_channel else in_ch))


def macs_infonce(n_tokens, dim):
    """n x n similarity matrix."""
    return int(n_tokens * n_tokens * dim)


# ══════════════════════════════════════════════════════════════
#  Measured timing / memory
# ══════════════════════════════════════════════════════════════

def timeit(fn, repeats, warmup, device, backward=False):
    """Returns (mean_ms, peak_mem_MB)."""
    for _ in range(warmup):
        out = fn()
        if backward and out is not None and out.requires_grad:
            out.backward()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    for _ in range(repeats):
        out = fn()
        if backward and out is not None and out.requires_grad:
            out.backward()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / repeats * 1000.0

    mem = (torch.cuda.max_memory_allocated() / 1024 ** 2
           if device.startswith("cuda") else float("nan"))
    return dt, mem


def count_params(module):
    if module is None:
        return 0, 0
    train = sum(p.numel() for p in module.parameters() if p.requires_grad)
    buf = sum(b.numel() for b in module.buffers())
    return train, buf


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None, help="unused; accepted for parity")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--img_size", type=int, default=112)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--num_patches", type=int, default=8)
    ap.add_argument("--num_blocks", type=int, default=2)
    ap.add_argument("--trg_ratio", type=float, nargs=2, default=[0.10, 0.15])
    ap.add_argument("--ctx_ratio", type=float, nargs=2, default=[0.90, 1.00])
    ap.add_argument("--gabor_orient", type=int, default=8)
    ap.add_argument("--gabor_gray", type=int, default=1, choices=[0, 1])
    ap.add_argument("--struct_head_hidden", type=int, default=128)
    ap.add_argument("--infonce_temp", type=float, default=0.1)
    ap.add_argument("--infonce_max_n", type=int, default=4096)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out_json", default="cost_profile.json")
    cfg = ap.parse_args()

    dev = cfg.device
    B, D, G = cfg.batch_size, cfg.embed_dim, cfg.num_patches
    P = G * G
    patch_px = cfg.img_size // G
    enc_depth = min(6, D // 64 + 2)
    pred_dim, pred_depth = 128, 6

    print(f"\n{'='*78}")
    print(f"  COST PROFILE   device={dev}  B={B}  img={cfg.img_size}  "
          f"embed_dim={D}  grid={G}x{G} (P={P})")
    print(f"{'='*78}\n")

    # ── build modules ──
    ctx_enc = ContextEncoder((cfg.img_size, cfg.img_size), G, D).to(dev)
    tgt_enc = TargetEncoder((cfg.img_size, cfg.img_size), G, D).to(dev)
    predictor = Predictor(G, D, norm_struct_out=True).to(dev)
    gabor = GaborBank(n_orient=cfg.gabor_orient,
                      per_channel=not bool(cfg.gabor_gray)).to(dev)
    head_a1 = StructureHead(D, gabor.K, hidden=cfg.struct_head_hidden).to(dev)
    head_a2 = StructureHead(D, gabor.K, hidden=cfg.struct_head_hidden).to(dev)

    images = torch.randn(B, 3, cfg.img_size, cfg.img_size, device=dev)
    ctx_masks, tgt_masks = patchify(
        B, G, cfg.num_blocks, trg_ratio=tuple(cfg.trg_ratio),
        ctx_ratio=tuple(cfg.ctx_ratio), device=dev)
    N_ctx = ctx_masks[0].size(1)
    N_tgt = tgt_masks[0].size(1)
    n_tgt = len(tgt_masks)
    print(f"  Mask geometry: N_ctx={N_ctx} visible, "
          f"N_tgt={N_tgt} x {n_tgt} blocks hidden  (of {P} patches)\n")

    # ── parameter counts ──
    rows_p = []
    for name, mod in [("Context encoder", ctx_enc),
                      ("Target encoder (EMA, frozen)", tgt_enc),
                      ("Predictor", predictor),
                      ("Gabor bank", gabor),
                      ("Structure head A1", head_a1),
                      ("Structure head A2 (if separate)", head_a2)]:
        tr, bf = count_params(mod)
        rows_p.append((name, tr, bf))

    # out_proj_struct + task_embed live inside Predictor; isolate them
    op_struct = sum(p.numel() for p in predictor.out_proj_struct.parameters())
    task_emb = predictor.task_embed.numel()

    print(f"  {'Component':<34} {'Trainable':>12} {'Buffers':>12}")
    print(f"  {'-'*34} {'-'*12} {'-'*12}")
    for name, tr, bf in rows_p:
        print(f"  {name:<34} {tr:>12,} {bf:>12,}")
    print(f"  {'  ├ of which out_proj_struct':<34} {op_struct:>12,} {0:>12,}")
    print(f"  {'  └ of which task_embed':<34} {task_emb:>12,} {0:>12,}")
    print()

    # ── analytic MACs (per image) ──
    m_ctx = (macs_patch_embed(cfg.img_size, patch_px, D)
             + macs_transformer_encoder(N_ctx, D, enc_depth))
    m_tgt = (macs_patch_embed(cfg.img_size, patch_px, D)
             + macs_transformer_encoder(P, D, enc_depth))
    m_pred_app = (macs_linear(N_ctx, D, pred_dim) * n_tgt
                  + macs_transformer_encoder(N_ctx + N_tgt, pred_dim, pred_depth) * n_tgt
                  + macs_linear(N_tgt, pred_dim, D) * n_tgt)
    m_pred_both = (macs_linear(N_ctx, D, pred_dim) * n_tgt
                   + macs_transformer_encoder(N_ctx + 2 * N_tgt, pred_dim, pred_depth) * n_tgt
                   + macs_linear(2 * N_tgt, pred_dim, D) * n_tgt)
    m_pred_a2_extra = m_pred_both - m_pred_app

    max_k = gabor.weight.shape[-1]
    m_gabor = macs_gabor_bank(cfg.img_size, gabor.K, max_k, gabor.per_channel)
    m_head_a1 = macs_linear(N_ctx, D, cfg.struct_head_hidden) \
                + macs_linear(N_ctx, cfg.struct_head_hidden, gabor.K)
    m_head_a2 = (macs_linear(N_tgt * n_tgt, D, cfg.struct_head_hidden)
                 + macs_linear(N_tgt * n_tgt, cfg.struct_head_hidden, gabor.K))

    # InfoNCE is quadratic in the BATCH token count, so compute per batch then /B
    tok_a1 = min(B * N_ctx, cfg.infonce_max_n)
    tok_a2 = min(B * N_tgt * n_tgt, cfg.infonce_max_n)
    m_nce_a1 = macs_infonce(tok_a1, gabor.K) / B
    m_nce_a2 = macs_infonce(tok_a2, gabor.K) / B

    print(f"  Gabor bank: K={gabor.K} channels, kernel {max_k}x{max_k} "
          f"({'per-channel RGB' if gabor.per_channel else 'grayscale'})\n")

    # ── measured timings ──
    def f_ctx():
        return ctx_enc(images, ctx_masks).sum()

    def f_tgt():
        with torch.no_grad():
            return tgt_enc(images).sum()

    ce = ctx_enc(images, ctx_masks).detach().requires_grad_(True)

    def f_pred_app():
        return predictor(ce, ctx_masks, tgt_masks).sum()

    def f_pred_both():
        a, s = predictor(ce, ctx_masks, tgt_masks, predict_structure=True)
        return a.sum() + s.sum()

    def f_gabor():
        with torch.no_grad():
            return patch_energy_descriptor(gabor(images), G).sum()

    with torch.no_grad():
        desc = patch_energy_descriptor(gabor(images), G)
        t_a1 = apply_masks(desc, ctx_masks)
        t_a2 = repeat_interleave_batch(apply_masks(desc, tgt_masks), B,
                                       repeat=len(ctx_masks))
    _, struct_hidden = predictor(ce, ctx_masks, tgt_masks, predict_structure=True)
    sh = struct_hidden.detach().requires_grad_(True)

    def f_a1_cos():
        l, _ = structure_loss(head_a1(ce), t_a1, kind="cosine")
        return l

    def f_a1_nce():
        l, _ = structure_loss(head_a1(ce), t_a1, kind="infonce",
                              temperature=cfg.infonce_temp,
                              max_n=cfg.infonce_max_n)
        return l

    def f_a2_cos():
        l, _ = structure_loss(head_a2(sh), t_a2, kind="cosine")
        return l

    def f_a2_nce():
        l, _ = structure_loss(head_a2(sh), t_a2, kind="infonce",
                              temperature=cfg.infonce_temp,
                              max_n=cfg.infonce_max_n)
        return l

    def f_corrupt():
        if not HAVE_CORRUPTION:
            return None
        with torch.no_grad():
            return corrupt_images(images, cfg, [0.5] * 3, [0.5] * 3).sum()

    timed = {}
    for label, fn, bwd in [
        ("Context encoder", f_ctx, True),
        ("Target encoder (no grad)", f_tgt, False),
        ("Predictor (appearance only)", f_pred_app, True),
        ("Predictor (appearance + structure)", f_pred_both, True),
        ("Gabor bank + descriptor (no grad)", f_gabor, False),
        ("A1 head + cosine loss", f_a1_cos, True),
        ("A1 head + InfoNCE loss", f_a1_nce, True),
        ("A2 head + cosine loss", f_a2_cos, True),
        ("A2 head + InfoNCE loss", f_a2_nce, True),
    ]:
        ms, mem = timeit(fn, cfg.repeats, cfg.warmup, dev, backward=bwd)
        timed[label] = (ms, mem)

    if HAVE_CORRUPTION:
        ms, mem = timeit(f_corrupt, cfg.repeats, cfg.warmup, dev, backward=False)
        timed["Corruption module (no grad)"] = (ms, mem)

    # ── component table ──
    print(f"  {'Component':<38} {'MMACs/img':>11} {'ms/batch':>10} {'peak MB':>9}")
    print(f"  {'-'*38} {'-'*11} {'-'*10} {'-'*9}")

    comp_macs = {
        "Context encoder": m_ctx,
        "Target encoder (no grad)": m_tgt,
        "Predictor (appearance only)": m_pred_app,
        "Predictor (appearance + structure)": m_pred_both,
        "Gabor bank + descriptor (no grad)": m_gabor,
        "A1 head + cosine loss": m_head_a1,
        "A1 head + InfoNCE loss": m_head_a1 + m_nce_a1,
        "A2 head + cosine loss": m_head_a2,
        "A2 head + InfoNCE loss": m_head_a2 + m_nce_a2,
        "Corruption module (no grad)": 0,
    }
    for label, (ms, mem) in timed.items():
        mm = comp_macs.get(label, 0) / 1e6
        print(f"  {label:<38} {mm:>11.1f} {ms:>10.2f} {mem:>9.0f}")
    print()

    # ── configuration comparison ──
    # Training cost model: no-grad modules count 1x, trainable modules ~3x
    # (1 forward + ~2 backward). This is the standard approximation.
    def train_macs(fwd_nograd, fwd_grad):
        return fwd_nograd + 3.0 * fwd_grad

    configs = {}
    configs["Baseline JEPA"] = dict(
        nograd=m_tgt, grad=m_ctx + m_pred_app, params=0)
    configs["+ corruption"] = dict(
        nograd=m_tgt, grad=m_ctx + m_pred_app, params=0)
    configs["+ A1 (cosine)"] = dict(
        nograd=m_tgt + m_gabor, grad=m_ctx + m_pred_app + m_head_a1,
        params=sum(p.numel() for p in head_a1.parameters()))
    configs["+ A1 (InfoNCE)"] = dict(
        nograd=m_tgt + m_gabor,
        grad=m_ctx + m_pred_app + m_head_a1 + m_nce_a1,
        params=sum(p.numel() for p in head_a1.parameters()))
    configs["+ A2 (InfoNCE)"] = dict(
        nograd=m_tgt + m_gabor,
        grad=m_ctx + m_pred_both + m_head_a2 + m_nce_a2,
        params=sum(p.numel() for p in head_a2.parameters()) + op_struct + task_emb)
    configs["+ both (shared head)"] = dict(
        nograd=m_tgt + m_gabor,
        grad=m_ctx + m_pred_both + m_head_a1 + m_head_a2 + m_nce_a2,
        params=sum(p.numel() for p in head_a1.parameters()) + op_struct + task_emb)
    configs["+ both (separate heads)"] = dict(
        nograd=m_tgt + m_gabor,
        grad=m_ctx + m_pred_both + m_head_a1 + m_head_a2 + m_nce_a2,
        params=(sum(p.numel() for p in head_a1.parameters())
                + sum(p.numel() for p in head_a2.parameters())
                + op_struct + task_emb))

    base_params = (sum(p.numel() for p in ctx_enc.parameters())
                   + sum(p.numel() for p in predictor.parameters())
                   - op_struct - task_emb)
    base_train = train_macs(configs["Baseline JEPA"]["nograd"],
                            configs["Baseline JEPA"]["grad"])

    print(f"{'='*94}")
    print(f"  TRAINING COST BY CONFIGURATION   (batch = {B} images)")
    print(f"{'='*94}")
    print(f"  {'Configuration':<26} {'Train par':>10} {'Δpar':>7} "
          f"{'GMAC/img':>9} {'GFLOP/img':>10} {'GFLOP/batch':>12} {'Δcompute':>9}")
    print(f"  {'-'*26} {'-'*10} {'-'*7} {'-'*9} {'-'*10} {'-'*12} {'-'*9}")

    results = {}
    for name, c in configs.items():
        tm = train_macs(c["nograd"], c["grad"])
        par = base_params + c["params"]
        row = {
            "train_params": par,
            "delta_params_pct": 100.0 * c["params"] / base_params,
            "gmac_per_img": tm / 1e9,
            "gflop_per_img": 2 * tm / 1e9,
            "gflop_per_batch": 2 * tm * B / 1e9,
            "delta_compute_pct": 100.0 * (tm - base_train) / base_train,
        }
        results[name] = row
        print(f"  {name:<26} {par/1e6:>9.2f}M {row['delta_params_pct']:>6.1f}% "
              f"{row['gmac_per_img']:>9.3f} {row['gflop_per_img']:>10.3f} "
              f"{row['gflop_per_batch']:>12.1f} {row['delta_compute_pct']:>8.1f}%")

    print(f"\n  INFERENCE (all configurations identical -- eval uses only the")
    print(f"  context encoder via FeatureExtractor; heads/Gabor/predictor unused)")
    infer_macs = macs_patch_embed(cfg.img_size, patch_px, D) \
                 + macs_transformer_encoder(P, D, enc_depth)
    print(f"    params  = {sum(p.numel() for p in ctx_enc.parameters())/1e6:.2f}M   "
          f"(+0.0% for every config)")
    print(f"    GFLOP/img = {2*infer_macs/1e9:.3f}   (+0.0% for every config)\n")

    # ── optional fvcore cross-check ──
    if HAVE_FVCORE:
        try:
            class _CtxWrap(nn.Module):
                def __init__(s, m, masks):
                    super().__init__(); s.m = m; s.masks = masks
                def forward(s, x):
                    return s.m(x, s.masks)
            fc = FlopCountAnalysis(_CtxWrap(ctx_enc, ctx_masks), images)
            fc.unsupported_ops_warnings(False); fc.uncalled_modules_warnings(False)
            print(f"  [fvcore cross-check] context encoder: "
                  f"{fc.total()/B/1e9:.3f} GFLOP/img "
                  f"(analytic: {2*m_ctx/1e9:.3f}; fvcore omits attention matmuls)\n")
        except Exception as e:
            print(f"  [fvcore cross-check skipped: {e}]\n")
    else:
        print("  [install fvcore or thop for an independent FLOP cross-check]\n")

    with open(cfg.out_json, "w") as f:
        json.dump({
            "settings": vars(cfg),
            "mask_geometry": {"N_ctx": N_ctx, "N_tgt": N_tgt, "n_blocks": n_tgt,
                              "P": P},
            "gabor": {"K": gabor.K, "kernel": int(max_k),
                      "per_channel": gabor.per_channel},
            "params": {n: {"trainable": t, "buffers": b} for n, t, b in rows_p},
            "timings_ms_per_batch": {k: v[0] for k, v in timed.items()},
            "peak_mem_MB": {k: v[1] for k, v in timed.items()},
            "configs": results,
        }, f, indent=2)
    print(f"  Saved: {cfg.out_json}\n")


if __name__ == "__main__":
    main()
