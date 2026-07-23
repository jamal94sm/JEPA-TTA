"""
TTA_single_target.py — Test-time adaptation on ONE target domain (v2).

Simulates the single-domain leg of NS-CTTA, assuming a new-domain detector has
already fired. Source data is NEVER used to build projectors or to adapt; it is
used only as an ORACLE DIAGNOSTIC to measure forgetting.

═══════════════════════════════════════════════════════════════════════════
  ARMS
═══════════════════════════════════════════════════════════════════════════
  tent        BN affine only, batch stats, NO projection      (TENT baseline)
  nsctta      projected layers only, BN FULLY FROZEN          (isolates the
                                                               projection)
  nsctta_bn   projected layers + BN adapt + per-domain BN pack restore

═══════════════════════════════════════════════════════════════════════════
  MULTI-LAYER NS-CTTA  (--layers)
═══════════════════════════════════════════════════════════════════════════
  Every selected layer gets its OWN projector, built from the covariance of
  THAT layer's INPUT — following Adam-NSCL:
      Linear : C = a^T a                 a  = layer input          [in, in]
      Conv2d : C = u^T u                 u  = im2col patches  [C_in*k*k, ...]
  and the Adam UPDATE is right-multiplied:   update <- update @ P .

  CompNet layer menu (im2col / input dims):
      backbone.stem.conv     3*7*7 = 147
      backbone.block1.conv  16*5*5 = 400
      backbone.block2.conv  32*3*3 = 288
      backbone.block3.conv  64*3*3 = 576
      backbone.proj                 = 128
  Examples:
      --layers proj                    (default; single-layer, as in v1)
      --layers all                     (4 convs + proj)
      --layers block3,proj
      --layers conv                    (the 4 convs only)

  classifier is ALWAYS frozen (closed-set readout is already correct, and a
  trainable head under entropy loss collapses to one confident class).
  proj.bias is ALWAYS frozen: y = Wx + b gives dy = dW x + db, and an additive
  db cannot be projected away, which would void the guarantee. The conv layers
  have bias=False already.

USAGE
-----
python TTA_single_target.py \
    --ckpt ./output_compnet/ckpt_casiams_compnet_WHT.pth \
    --data_dir /home/pai-ng/Jamal/CASIA-MS-ROI \
    --source_spectrum WHT --target_spectrum 700 \
    --layers all
"""

import os
import json
import math
import copy
import random
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import CompNet
from dataset import (scan_dataset, build_id_map, split_gallery_probe,
                     CASIADataset)
from evaluate import evaluate_rank1_eer


ALL_ARMS = ["tent", "nsctta", "nsctta_bn"]


# ══════════════════════════════════════════════════════════════════════
#  Args
# ══════════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser("Single-target TTA (TENT vs NS-CTTA)")
    # data / model
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--source_spectrum", default="WHT")
    p.add_argument("--target_spectrum", default="700")
    p.add_argument("--adapt_ratio", type=float, default=0.5)
    p.add_argument("--gallery_ratio", type=float, default=0.5)
    # which layers get a projector / are trainable
    p.add_argument("--layers", default="proj",
                   help="'proj' | 'conv' | 'all' | comma-separated names or "
                        "substrings, e.g. 'block3,proj'")
    # subspace / projector
    p.add_argument("--k0", type=int, default=0,
                   help="absolute floor on protected dims per layer (0 = off)")
    p.add_argument("--k0_frac", type=float, default=0.0,
                   help="floor as a FRACTION of each layer's dim (0 = off). "
                        "Use this instead of --k0 when layers differ in size.")
    p.add_argument("--energy", type=float, default=0.99,
                   help="k_t = #dirs holding this fraction of target energy")
    p.add_argument("--proj_normalize", default="frobenius",
                   choices=["frobenius", "none"])
    p.add_argument("--cov_batches", type=int, default=0,
                   help="cap batches used for covariance (0 = all)")
    # TTA
    p.add_argument("--arms", nargs="+", default=ALL_ARMS, choices=ALL_ARMS)
    p.add_argument("--adapt_mode", default="batch", choices=["batch", "set"])
    p.add_argument("--n_epochs", type=int, default=5)
    p.add_argument("--conf_ratio", type=float, default=0.4,
                   help="keep samples with H < conf_ratio*ln(C); 1.0 = off")
    p.add_argument("--lr", type=float, default=1e-3, help="lr for BN params")
    p.add_argument("--proj_lr", type=float, default=1e-2,
                   help="lr for PROJECTED weights (Frobenius normalisation "
                        "shrinks updates by ~1/sqrt(rank), so raise this)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--eps_src", type=float, default=0.5,
                   help="tolerance for 'source preserved' in the P2b check")
    p.add_argument("--out_dir", default="./output_tta")
    return p.parse_args()


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ══════════════════════════════════════════════════════════════════════
#  P0 — model + data
# ══════════════════════════════════════════════════════════════════════

def load_source_model(args, all_samples):
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if ckpt.get("method", "?") != "compnet":
        raise SystemExit(f"expected a CompNet checkpoint, got "
                         f"method={ckpt.get('method')}")
    arch = ckpt["arch"]
    n_cls = ckpt["classifier"]["weight"].shape[0]
    model = CompNet(arch["embed_dim"], n_cls,
                    base=arch.get("compnet_channels", 16)).to(args.device)
    model.backbone.load_state_dict(ckpt["backbone"])
    model.classifier.load_state_dict(ckpt["classifier"])
    model.eval()

    if "train_id_map" in ckpt:
        train_id_map, src = ckpt["train_id_map"], "checkpoint"
    else:
        train_id_map, src = build_id_map(all_samples), "REBUILT from data"
    if len(train_id_map) != n_cls:
        raise SystemExit(
            f"train_id_map has {len(train_id_map)} ids but the classifier has "
            f"{n_cls} outputs. Re-save with \"train_id_map\": train_id_map.")

    print(f"  model : CompNet  d={arch['embed_dim']}  "
          f"base={arch.get('compnet_channels',16)}  classes={n_cls}")
    print(f"  ckpt  : epoch={ckpt.get('epoch','?')}  "
          f"EER={ckpt.get('mean_eer', float('nan')):.2f}%  "
          f"R1={ckpt.get('mean_rank1', float('nan')):.2f}%")
    print(f"  id_map: {len(train_id_map)} training identities ({src})")
    return model, arch, train_id_map, n_cls


def split_adapt_test(samples, adapt_ratio, seed):
    """Per-identity split into an ADAPT stream and a held-out TEST set."""
    by_id = defaultdict(list)
    for s in samples:
        by_id[s["identity"]].append(s)
    rng = random.Random(seed)
    adapt, test = [], []
    for ident in sorted(by_id):
        items = by_id[ident][:]
        rng.shuffle(items)
        n_ad = int(round(len(items) * adapt_ratio))
        n_ad = max(0, min(n_ad, len(items) - 2))   # keep >=2 for gallery+probe
        adapt.extend(items[:n_ad]); test.extend(items[n_ad:])
    return adapt, test


def make_loader(samples, id_map, img_size, args, shuffle=False):
    ds = CASIADataset(samples, id_map, img_size, augment=False, aug_multiplier=1)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.num_workers, drop_last=False)


# ══════════════════════════════════════════════════════════════════════
#  Layer selection
# ══════════════════════════════════════════════════════════════════════

def select_layers(model, spec):
    """Return [(name, module)] of Conv2d/Linear layers to protect + train.
       The classifier is never selectable."""
    cand = [(n, m) for n, m in model.named_modules()
            if isinstance(m, (nn.Conv2d, nn.Linear)) and n != "classifier"]
    spec = spec.strip().lower()
    if spec == "all":
        sel = cand
    elif spec == "conv":
        sel = [(n, m) for n, m in cand if isinstance(m, nn.Conv2d)]
    elif spec == "proj":
        sel = [(n, m) for n, m in cand if n.endswith("proj")]
    else:
        keys = [k.strip() for k in spec.split(",") if k.strip()]
        sel = [(n, m) for n, m in cand if any(k in n for k in keys)]
    if not sel:
        raise SystemExit(f"--layers '{spec}' selected no layers. "
                         f"available: {[n for n,_ in cand]}")
    return sel


def layer_in_dim(module):
    """Dimension of the space the projector lives in = the layer's input dim."""
    if isinstance(module, nn.Linear):
        return module.in_features
    k = module.kernel_size
    return module.in_channels * k[0] * k[1]


# ══════════════════════════════════════════════════════════════════════
#  Per-layer covariance  (Adam-NSCL: svd_agent/svd_based.py)
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def accumulate_covariances(model, loader, layers, args):
    """Uncentered covariance of each selected layer's INPUT.
         Linear : a^T a          Conv2d : im2col patches u^T u
       Returns {name: [in_dim, in_dim]} on CPU (float64)."""
    covs = {}

    def hook_for(name, module):
        def hook(mod, fin, fout):
            a = fin[0].detach()
            if isinstance(mod, nn.Linear):
                a2 = a.reshape(-1, a.shape[-1])
            else:
                u = F.unfold(a, kernel_size=mod.kernel_size,
                             padding=mod.padding, stride=mod.stride,
                             dilation=mod.dilation)          # [B, C*k*k, L]
                a2 = u.permute(0, 2, 1).reshape(-1, u.shape[1])
            c = (a2.double().T @ a2.double()).cpu()
            covs[name] = c if name not in covs else covs[name] + c
        return hook

    handles = [m.register_forward_hook(hook_for(n, m)) for n, m in layers]
    model.eval()
    for i, (x, _) in enumerate(loader):
        if args.cov_batches and i >= args.cov_batches:
            break
        model.backbone(x.to(args.device))
    for h in handles:
        h.remove()
    return covs


def build_projectors(covs, layers, args):
    """P_name = I - U_k U_k^T for each layer, Frobenius-normalised."""
    Ps, info = {}, {}
    for name, module in layers:
        C = covs[name]
        d = C.shape[0]
        ev, U = torch.linalg.eigh(C)
        ev = ev.flip(0).clamp_min(0)
        U = U.flip(1).float()
        cum = (ev / ev.sum().clamp_min(1e-30)).cumsum(0)
        k_t = int((cum < args.energy).sum().item()) + 1
        k = max(k_t, args.k0, int(round(args.k0_frac * d)))
        k = min(k, d - 1)                      # always leave >=1 free direction
        Uk = U[:, :k]
        P = torch.eye(d) - Uk @ Uk.T
        scale = 1.0
        if args.proj_normalize == "frobenius":
            scale = float(P.norm()); P = P / scale
        part = float((ev.sum() ** 2) / (ev ** 2).sum().clamp_min(1e-30))
        Ps[name] = P
        info[name] = dict(d=d, k=k, k_t=k_t, free=d - k,
                          energy_kept=float(cum[k - 1]),
                          part_ratio=part, frob=scale)
    return Ps, info


@torch.no_grad()
def layer_rho(model, loader, layers, Ps, args):
    """Per layer: fraction of input energy that SURVIVES the projector.
       Small on source => protected. Small on target => no room to adapt."""
    num = defaultdict(float); den = defaultdict(float)

    def hook_for(name, module, P):
        def hook(mod, fin, fout):
            a = fin[0].detach()
            if isinstance(mod, nn.Linear):
                a2 = a.reshape(-1, a.shape[-1])
            else:
                u = F.unfold(a, kernel_size=mod.kernel_size,
                             padding=mod.padding, stride=mod.stride,
                             dilation=mod.dilation)
                a2 = u.permute(0, 2, 1).reshape(-1, u.shape[1])
            a2 = a2.double(); Pd = P.to(a2.device).double()
            num[name] += float((a2 @ Pd.T).pow(2).sum())
            den[name] += float(a2.pow(2).sum())
        return hook

    handles = [m.register_forward_hook(hook_for(n, m, Ps[n])) for n, m in layers]
    model.eval()
    for i, (x, _) in enumerate(loader):
        if args.cov_batches and i >= args.cov_batches:
            break
        model.backbone(x.to(args.device))
    for h in handles:
        h.remove()
    return {n: num[n] / max(den[n], 1e-30) for n, _ in layers}


# ══════════════════════════════════════════════════════════════════════
#  Feature extraction / evaluation
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_feats(model, loader, device):
    model.eval()
    X, Y = [], []
    for x, y in loader:
        X.append(model.backbone(x.to(device)).cpu()); Y.append(y)
    return torch.cat(X), torch.cat(Y)


def evaluate_domain(model, gal_loader, prb_loader, device):
    gf, gl = extract_feats(model, gal_loader, device)
    pf, pl = extract_feats(model, prb_loader, device)
    return evaluate_rank1_eer(F.normalize(gf, dim=-1), gl,
                              F.normalize(pf, dim=-1), pl)


def make_proj_pre_hook(M):
    def hook(module, inputs):
        h = inputs[0]
        return (h @ M.to(h.device, h.dtype),)
    return hook


def evaluate_through_subspace(model, M, gal_loader, prb_loader, device):
    """Evaluate with proj's INPUT passed through M (used by the P2b check)."""
    h = model.backbone.proj.register_forward_pre_hook(make_proj_pre_hook(M))
    try:
        return evaluate_domain(model, gal_loader, prb_loader, device)
    finally:
        h.remove()


# ══════════════════════════════════════════════════════════════════════
#  BN pack (per-domain state)
# ══════════════════════════════════════════════════════════════════════

def snapshot_bn(model):
    pack = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            pack[name] = {k: getattr(m, k).detach().clone() for k in
                          ("weight", "bias", "running_mean", "running_var",
                           "num_batches_tracked")}
    return pack


def restore_bn(model, pack):
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d) and name in pack:
            for k, v in pack[name].items():
                getattr(m, k).data.copy_(v)


# ══════════════════════════════════════════════════════════════════════
#  Projected Adam (project the UPDATE, per Adam-NSCL)
# ══════════════════════════════════════════════════════════════════════

class ProjectedAdam(torch.optim.Optimizer):
    def __init__(self, groups, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(groups, dict(betas=betas, eps=eps,
                                      weight_decay=weight_decay))
        self.projectors = {}

    def set_projector(self, param, P):
        self.projectors[id(param)] = P

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                st = self.state[p]
                if len(st) == 0:
                    st["step"] = 0
                    st["exp_avg"] = torch.zeros_like(p)
                    st["exp_avg_sq"] = torch.zeros_like(p)
                b1, b2 = group["betas"]
                st["step"] += 1
                if group["weight_decay"]:
                    grad = grad.add(p, alpha=group["weight_decay"])
                st["exp_avg"].mul_(b1).add_(grad, alpha=1 - b1)
                st["exp_avg_sq"].mul_(b2).addcmul_(grad, grad, value=1 - b2)
                denom = st["exp_avg_sq"].sqrt().add_(group["eps"])
                bc1 = 1 - b1 ** st["step"]; bc2 = 1 - b2 ** st["step"]
                upd = -(group["lr"] * math.sqrt(bc2) / bc1) * st["exp_avg"] / denom
                P = self.projectors.get(id(p))
                if P is not None:
                    shp = upd.shape                      # conv: [O,I,kh,kw]
                    upd = (upd.view(shp[0], -1) @ P.to(upd.device)).view(shp)
                p.add_(upd)


# ══════════════════════════════════════════════════════════════════════
#  Loss + model configuration
# ══════════════════════════════════════════════════════════════════════

def entropy_loss(logits, conf_ratio):
    p = logits.softmax(1)
    ent = -(p * p.clamp_min(1e-12).log()).sum(1)
    if conf_ratio >= 1.0:
        return ent.mean(), 1.0
    mask = ent < conf_ratio * math.log(logits.size(1))
    if mask.sum() == 0:
        return None, 0.0
    return ent[mask].mean(), float(mask.float().mean())


def configure_model(model, arm, layers):
    """Freeze everything, then enable exactly the intended surfaces.
         tent      : BN affine + batch stats
         nsctta    : projected layer weights, BN FROZEN
         nsctta_bn : projected layer weights + BN affine + batch stats
       classifier and all biases stay frozen."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    bn_adapt = arm in ("tent", "nsctta_bn")
    bn_params = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            if bn_adapt:
                m.train()
                m.weight.requires_grad_(True); m.bias.requires_grad_(True)
                bn_params += [m.weight, m.bias]
            else:
                m.eval()

    proj_params = []
    if arm in ("nsctta", "nsctta_bn"):
        for name, mod in layers:
            mod.weight.requires_grad_(True)
            proj_params.append((name, mod.weight))
            # biases are never trained: db cannot be projected away
    return bn_params, proj_params


# ══════════════════════════════════════════════════════════════════════
#  TTA loop
# ══════════════════════════════════════════════════════════════════════

def run_tta(model, loader, args, arm, layers, Ps, tag=""):
    bn_params, proj_params = configure_model(model, arm, layers)
    groups = []
    if bn_params:
        groups.append({"params": bn_params, "lr": args.lr})
    if proj_params:
        groups.append({"params": [p for _, p in proj_params],
                       "lr": args.proj_lr})
    if not groups:
        print(f"    [{tag}] nothing trainable — skipping")
        return {"steps": 0, "kept": 0.0, "loss": float("nan")}

    opt = ProjectedAdam(groups)
    for name, w in proj_params:
        opt.set_projector(w, Ps[name].to(args.device))

    n_ep = 1 if args.adapt_mode == "batch" else args.n_epochs
    print(f"    [{tag}] BN layers={len(bn_params)//2}  "
          f"projected layers={len(proj_params)}"
          f"{' (' + ','.join(n for n,_ in proj_params) + ')' if proj_params else ''}"
          f"  mode={args.adapt_mode} epochs={n_ep}")

    steps, tot_loss, tot_kept, skipped = 0, 0.0, 0.0, 0
    for ep in range(n_ep):
        el, ek, es = 0.0, 0.0, 0
        for x, _ in loader:
            x = x.to(args.device)
            logits, _f = model(x)
            loss, kept = entropy_loss(logits, args.conf_ratio)
            if loss is None:
                skipped += 1; continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            el += loss.item(); ek += kept; es += 1
        if es:
            steps += es; tot_loss += el; tot_kept += ek
            if n_ep > 1:
                print(f"      ep {ep+1}/{n_ep}: H={el/es:.4f} kept={100*ek/es:.1f}%")
    if steps == 0:
        print(f"    [{tag}] WARNING: every batch filtered "
              f"(conf_ratio={args.conf_ratio} too strict)")
        return {"steps": 0, "kept": 0.0, "loss": float("nan")}
    print(f"    [{tag}] {steps} updates | mean H={tot_loss/steps:.4f} | "
          f"kept={100*tot_kept/steps:.1f}% | skipped={skipped}")
    return {"steps": steps, "kept": tot_kept / steps, "loss": tot_loss / steps}


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

def main():
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    dev = args.device
    R = {"config": vars(args)}

    print(f"\n{'='*78}")
    print(f"  SINGLE-TARGET TTA   source={args.source_spectrum}  "
          f"target={args.target_spectrum}   layers='{args.layers}'")
    print(f"{'='*78}")

    # ── P0 ───────────────────────────────────────────────────────────
    print(f"\n{'─'*78}\n  P0  Source model + closed-set splits\n{'─'*78}")
    all_samples = scan_dataset(args.data_dir)
    model, arch, train_id_map, n_cls = load_source_model(args, all_samples)
    img_size = arch["img_size"]

    known = set(train_id_map)
    src_all = [s for s in all_samples if s["spectrum"] == args.source_spectrum
               and s["identity"] in known]
    tgt_all = [s for s in all_samples if s["spectrum"] == args.target_spectrum
               and s["identity"] in known]
    if not tgt_all:
        raise SystemExit(f"no target samples for {args.target_spectrum}")

    src_adapt, src_test = split_adapt_test(src_all, args.adapt_ratio, args.seed)
    tgt_adapt, tgt_test = split_adapt_test(tgt_all, args.adapt_ratio, args.seed)
    s_gal, s_prb = split_gallery_probe(src_test, train_id_map,
                                       args.gallery_ratio, args.seed)
    t_gal, t_prb = split_gallery_probe(tgt_test, train_id_map,
                                       args.gallery_ratio, args.seed)
    print(f"  source '{args.source_spectrum}': {len(src_all)} -> "
          f"adapt {len(src_adapt)} (unused) | test {len(src_test)} "
          f"(gal {len(s_gal)}/prb {len(s_prb)})")
    print(f"  target '{args.target_spectrum}': {len(tgt_all)} -> "
          f"adapt {len(tgt_adapt)} | test {len(tgt_test)} "
          f"(gal {len(t_gal)}/prb {len(t_prb)})")
    print(f"  NOTE: source data is an ORACLE DIAGNOSTIC only.")

    L = lambda s, sh=False: make_loader(s, train_id_map, img_size, args, sh)
    tgt_stream = L(tgt_adapt, True)
    tgt_cov = L(tgt_adapt)
    src_cov = L(src_test)
    t_gal_l, t_prb_l = L(t_gal), L(t_prb)
    s_gal_l, s_prb_l = L(s_gal), L(s_prb)

    # ── P1 ───────────────────────────────────────────────────────────
    print(f"\n{'─'*78}\n  P1  Baselines (frozen source model)\n{'─'*78}")
    base_t = evaluate_domain(model, t_gal_l, t_prb_l, dev)
    base_s = evaluate_domain(model, s_gal_l, s_prb_l, dev)
    print(f"  TARGET  EER={base_t['eer']:6.2f}%  R1={base_t['rank1']:6.2f}%")
    print(f"  SOURCE  EER={base_s['eer']:6.2f}%  R1={base_s['rank1']:6.2f}%")
    R["baseline"] = {"target": base_t, "source": base_s}

    # ── P2 ───────────────────────────────────────────────────────────
    print(f"\n{'─'*78}\n  P2  Per-layer null-space projectors from TARGET"
          f"\n{'─'*78}")
    layers = select_layers(model, args.layers)
    print(f"  selected {len(layers)} layer(s): "
          f"{', '.join(f'{n}({layer_in_dim(m)}d)' for n, m in layers)}")

    covs = accumulate_covariances(model, tgt_cov, layers, args)
    Ps, info = build_projectors(covs, layers, args)
    rho_t = layer_rho(model, tgt_cov, layers, Ps, args)
    rho_s = layer_rho(model, src_cov, layers, Ps, args)

    hdr = (f"  {'layer':<22}{'dim':>6}{'k_t':>6}{'k':>6}{'free':>6}"
           f"{'E_kept':>9}{'part':>7}{'rho_tgt':>10}{'rho_src':>10}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for name, _m in layers:
        i = info[name]
        print(f"  {name:<22}{i['d']:6d}{i['k_t']:6d}{i['k']:6d}{i['free']:6d}"
              f"{i['energy_kept']:9.4f}{i['part_ratio']:7.1f}"
              f"{rho_t[name]:10.5f}{rho_s[name]:10.5f}")
    print(f"  rho_src small => protected;  rho_tgt small => little room to adapt")
    R["projector"] = {n: {**info[n], "rho_target": rho_t[n],
                          "rho_source": rho_s[n]} for n, _ in layers}

    src_bn_pack = snapshot_bn(model)
    print(f"  snapshotted source BN pack: {len(src_bn_pack)} layers")

    # ── P2b: does the TARGET-built subspace preserve the SOURCE? ──
    if any(n.endswith("proj") for n, _ in layers):
        print(f"\n{'─'*78}\n  P2b  SOURCE through TARGET-derived subspace "
              f"(proj layer)\n{'─'*78}")
        C = covs["backbone.proj"]
        ev, U = torch.linalg.eigh(C); U = U.flip(1).float()
        d = C.shape[0]
        print(f"  {'k':>5} | {'src EER':>8} {'dEER':>7} | {'src R1':>8} "
              f"{'dR1':>7} | {'tgt EER':>8} {'tgt R1':>8}")
        curve = []
        for kk in [k for k in (4, 8, 16, 24, 32, 48, 64, 96, d) if k <= d]:
            Uk = U[:, :kk]; M = Uk @ Uk.T
            rs = evaluate_through_subspace(model, M, s_gal_l, s_prb_l, dev)
            rt = evaluate_through_subspace(model, M, t_gal_l, t_prb_l, dev)
            de = rs["eer"] - base_s["eer"]; dr = rs["rank1"] - base_s["rank1"]
            curve.append({"k": kk, "src_eer": rs["eer"], "src_rank1": rs["rank1"],
                          "d_eer": de, "d_rank1": dr,
                          "tgt_eer": rt["eer"], "tgt_rank1": rt["rank1"]})
            mark = "  <-- k used" if kk == info["backbone.proj"]["k"] else ""
            print(f"  {kk:5d} | {rs['eer']:8.2f} {de:+7.2f} | "
                  f"{rs['rank1']:8.2f} {dr:+7.2f} | {rt['eer']:8.2f} "
                  f"{rt['rank1']:8.2f}{mark}")
        k_ok = next((c["k"] for c in curve if c["d_eer"] <= args.eps_src
                     and c["d_rank1"] >= -args.eps_src), d)
        print(f"  smallest k preserving source (±{args.eps_src}) = {k_ok}")
        R["source_through_target_subspace"] = {"curve": curve, "k_ok": k_ok}

    # ── P3..P5  arms ─────────────────────────────────────────────────
    results = {}
    for i, arm in enumerate(args.arms):
        print(f"\n{'─'*78}\n  P{3+i}  Arm: {arm.upper()}\n{'─'*78}")
        m = copy.deepcopy(model)
        m_layers = select_layers(m, args.layers)     # rebind to the copy
        stats = run_tta(m, tgt_stream, args, arm, m_layers, Ps, tag=arm)
        r_t = evaluate_domain(m, t_gal_l, t_prb_l, dev)
        r_s = evaluate_domain(m, s_gal_l, s_prb_l, dev)
        print(f"    TARGET  EER={r_t['eer']:6.2f}% "
              f"({r_t['eer']-base_t['eer']:+5.2f})   "
              f"R1={r_t['rank1']:6.2f}% ({r_t['rank1']-base_t['rank1']:+5.2f})")
        print(f"    SOURCE  EER={r_s['eer']:6.2f}% "
              f"({r_s['eer']-base_s['eer']:+5.2f})   "
              f"R1={r_s['rank1']:6.2f}% ({r_s['rank1']-base_s['rank1']:+5.2f})")
        entry = {"target": r_t, "source": r_s, "stats": stats}
        if arm == "nsctta_bn":
            restore_bn(m, src_bn_pack)
            r_s2 = evaluate_domain(m, s_gal_l, s_prb_l, dev)
            print(f"    SOURCE after BN restore: EER={r_s2['eer']:6.2f}% "
                  f"({r_s2['eer']-base_s['eer']:+5.2f})   "
                  f"R1={r_s2['rank1']:6.2f}% "
                  f"({r_s2['rank1']-base_s['rank1']:+5.2f})")
            entry["source_bn_restored"] = r_s2
        results[arm] = entry

    # ── summary ──────────────────────────────────────────────────────
    print(f"\n{'='*78}\n  RESULTS   (layers='{args.layers}', "
          f"adapt_mode={args.adapt_mode})\n{'='*78}")
    hdr = (f"  {'arm':<24}{'tgt EER':>9}{'tgt R1':>9}{'src EER':>10}{'src R1':>9}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    print(f"  {'source (no TTA)':<24}{base_t['eer']:9.2f}{base_t['rank1']:9.2f}"
          f"{base_s['eer']:10.2f}{base_s['rank1']:9.2f}")
    for arm in args.arms:
        r = results[arm]
        print(f"  {arm:<24}{r['target']['eer']:9.2f}{r['target']['rank1']:9.2f}"
              f"{r['source']['eer']:10.2f}{r['source']['rank1']:9.2f}")
        if "source_bn_restored" in r:
            b = r["source_bn_restored"]
            print(f"  {'  + BN restore':<24}{'—':>9}{'—':>9}"
                  f"{b['eer']:10.2f}{b['rank1']:9.2f}")
    print(f"\n  target columns = adaptation gain;  source columns = forgetting")
    print(f"  'nsctta' isolates the projection (BN frozen); 'tent' isolates BN")

    R["results"] = results
    jp = os.path.join(args.out_dir,
                      f"tta_{args.source_spectrum}_{args.target_spectrum}_"
                      f"{args.layers.replace(',','-')}_{args.adapt_mode}.json")
    with open(jp, "w") as f:
        json.dump(R, f, indent=2, default=float)
    print(f"\n  saved {jp}\n{'='*78}\n")


if __name__ == "__main__":
    main()
