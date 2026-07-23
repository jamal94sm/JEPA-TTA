"""
TTA_single_target.py — Test-time adaptation on ONE target domain.

Simulates the single-domain leg of NS-CTTA, assuming a new-domain detector has
already fired. Source data is NEVER used to build the projector or to adapt; it
is used only as an ORACLE DIAGNOSTIC to measure forgetting.

═══════════════════════════════════════════════════════════════════════════
  PHASES
═══════════════════════════════════════════════════════════════════════════
  P0  Load frozen source model (CompNet) + train_id_map; build closed-set
      adapt/test splits for BOTH source and target domains.
  P1  Baselines: source model evaluated on target test and source test.
  P2  Build the null-space projector from TARGET pooled activations
      (proxy for the source subspace — no source data used).
        k = max(k0, k_t@energy)   P = I - U_k U_k^T   (Frobenius-normalised)
  P3  Arm A — TENT baseline: BN affine only, NO projection.
  P4  Arm B — NS-CTTA: proj.weight (projected) + BN, classifier frozen.
  P5  Source re-evaluation with the stored source BN pack restored.
  P6  Results table + JSON dump.

═══════════════════════════════════════════════════════════════════════════
  KEY DESIGN POINTS
═══════════════════════════════════════════════════════════════════════════
  * Protected layer = backbone.proj  (Linear 128 -> 256).
    A layer is protected in the space of ITS INPUT, so the projector is 128-d
    (proj's input = pooled vector), NOT 256-d. Guarantee:
        dW_proj @ h_src = 0  =>  feature x unchanged  =>  logits unchanged
    (classifier frozen), i.e. representation-level preservation.
  * proj.BIAS IS FROZEN. y = Wx + b gives dy = dW x + db; an additive db cannot
    be projected away, so training it would void the guarantee.
  * Classifier FROZEN: closed-set readout is already correct, and a trainable
    head under entropy loss collapses to one confident class.
  * Projection is applied to the ADAM UPDATE (not the raw gradient), following
    Adam-NSCL, and P is normalised by its Frobenius norm -> updates shrink by
    ~1/sqrt(rank), so the projected group needs a LARGER lr (see --proj_lr).
  * BN packs (gamma, beta, running mean/var) are snapshotted per domain BEFORE
    any target batch touches them, and restored on source recurrence.

USAGE
-----
python TTA_single_target.py \
    --ckpt ./output_compnet/ckpt_casiams_compnet_WHT.pth \
    --data_dir /home/pai-ng/Jamal/CASIA-MS-ROI \
    --source_spectrum WHT --target_spectrum 700
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
    p.add_argument("--adapt_ratio", type=float, default=0.5,
                   help="fraction of each domain used for ADAPTATION; the rest "
                        "is the held-out gallery/probe test split")
    p.add_argument("--gallery_ratio", type=float, default=0.5)
    # subspace / projector
    p.add_argument("--k0", type=int, default=16,
                   help="floor on subspace size (validate per model/layer!)")
    p.add_argument("--energy", type=float, default=0.98,
                   help="k_t = #dirs holding this fraction of target energy")
    p.add_argument("--proj_normalize", default="frobenius",
                   choices=["frobenius", "none"],
                   help="Adam-NSCL normalises P by ||P||_F")
    # TTA
    p.add_argument("--adapt_mode", default="batch", choices=["batch", "set"],
                   help="batch = online single pass; set = n_epochs over the set")
    p.add_argument("--n_epochs", type=int, default=5,
                   help="only used when --adapt_mode set")
    p.add_argument("--conf_ratio", type=float, default=0.4,
                   help="TENT confidence filter: keep samples with "
                        "H < conf_ratio * ln(C). 1.0 disables filtering")
    p.add_argument("--bn_mode", default="adapt", choices=["adapt", "freeze"],
                   help="adapt = BN affine trainable + batch stats (TENT-style); "
                        "freeze = BN fully frozen (isolates the projected layer)")
    p.add_argument("--lr", type=float, default=1e-3, help="lr for BN params")
    p.add_argument("--proj_lr", type=float, default=1e-2,
                   help="lr for the PROJECTED proj.weight (needs to be larger "
                        "because Frobenius normalisation shrinks updates)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--out_dir", default="./output_tta")
    p.add_argument("--eps_src", type=float, default=0.5,
                   help="tolerance (EER pts / R1 pts) for 'source preserved' "
                        "in the P2b functional check")
  
    return p.parse_args()


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ══════════════════════════════════════════════════════════════════════
#  P0 — model + data
# ══════════════════════════════════════════════════════════════════════

def load_source_model(args, all_samples):
    """Rebuild the frozen CompNet source model and recover train_id_map."""
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    method = ckpt.get("method", "?")
    if method != "compnet":
        raise SystemExit(f"this script expects a CompNet checkpoint, got "
                         f"method={method}")
    arch = ckpt["arch"]
    n_cls = ckpt["classifier"]["weight"].shape[0]

    model = CompNet(arch["embed_dim"], n_cls,
                    base=arch.get("compnet_channels", 16)).to(args.device)
    model.backbone.load_state_dict(ckpt["backbone"])
    model.classifier.load_state_dict(ckpt["classifier"])
    model.eval()

    # ── train_id_map: identity string -> classifier output index ──
    if "train_id_map" in ckpt:
        train_id_map = ckpt["train_id_map"]
        src = "checkpoint"
    else:
        # Fallback: for mode='all' every identity was a training identity, so
        # build_id_map over all samples reproduces it exactly. Validated below.
        train_id_map = build_id_map(all_samples)
        src = "REBUILT from data (ckpt lacks 'train_id_map')"
    if len(train_id_map) != n_cls:
        raise SystemExit(
            f"train_id_map has {len(train_id_map)} ids but the classifier has "
            f"{n_cls} outputs. Re-save the checkpoint with "
            f"\"train_id_map\": train_id_map in train_compnet().")

    print(f"  model : CompNet  d={arch['embed_dim']}  base="
          f"{arch.get('compnet_channels', 16)}  classes={n_cls}")
    print(f"  ckpt  : epoch={ckpt.get('epoch','?')}  "
          f"EER={ckpt.get('mean_eer', float('nan')):.2f}%  "
          f"R1={ckpt.get('mean_rank1', float('nan')):.2f}%")
    print(f"  id_map: {len(train_id_map)} training identities ({src})")
    return model, arch, train_id_map, n_cls


def split_adapt_test(samples, adapt_ratio, seed):
    """Per-identity split into an ADAPT stream and a held-out TEST set.
       Every identity appears in both (closed-set), with >=2 test samples so
       gallery/probe is well defined."""
    by_id = defaultdict(list)
    for s in samples:
        by_id[s["identity"]].append(s)
    rng = random.Random(seed)
    adapt, test = [], []
    for ident in sorted(by_id):
        items = by_id[ident][:]
        rng.shuffle(items)
        n_ad = int(round(len(items) * adapt_ratio))
        n_ad = max(0, min(n_ad, len(items) - 2))   # leave >=2 for gallery+probe
        adapt.extend(items[:n_ad])
        test.extend(items[n_ad:])
    return adapt, test


def make_loader(samples, id_map, img_size, args, shuffle=False):
    ds = CASIADataset(samples, id_map, img_size, augment=False, aug_multiplier=1)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.num_workers, drop_last=False)


# ══════════════════════════════════════════════════════════════════════
#  Feature / activation extraction
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_feats(model, loader, device):
    """256-d features (backbone output) + labels — used for EER/R1."""
    model.eval()
    X, Y = [], []
    for x, y in loader:
        X.append(model.backbone(x.to(device)).cpu())
        Y.append(y)
    return torch.cat(X), torch.cat(Y)


@torch.no_grad()
def extract_pooled(model, loader, device):
    """128-d POOLED vectors = proj's INPUT — the space the projector lives in."""
    model.eval()
    store, H = {}, []
    h = model.backbone.proj.register_forward_hook(
        lambda m, fin, fout: store.__setitem__("h", fin[0].detach()))
    for x, _ in loader:
        model.backbone(x.to(device))
        H.append(store["h"].cpu())
    h.remove()
    return torch.cat(H)


def evaluate_domain(model, gal_loader, prb_loader, device):
    gf, gl = extract_feats(model, gal_loader, device)
    pf, pl = extract_feats(model, prb_loader, device)
    return evaluate_rank1_eer(F.normalize(gf, dim=-1), gl,
                              F.normalize(pf, dim=-1), pl)


# ══════════════════════════════════════════════════════════════════════
#  P2 — null-space projector
# ══════════════════════════════════════════════════════════════════════

def build_projector(H, k0, energy_thresh, normalize):
    """P = I - U_k U_k^T from the uncentered covariance of pooled activations.
       k = max(k0, k_t) where k_t holds `energy_thresh` of the target energy."""
    d = H.size(1)
    C = (H.double().T @ H.double()) / H.size(0)
    ev, U = torch.linalg.eigh(C)
    ev = ev.flip(0).clamp_min(0)
    U = U.flip(1).float()

    cum = (ev / ev.sum()).cumsum(0)
    k_t = int((cum < energy_thresh).sum().item()) + 1
    k = min(max(k0, k_t), d)

    Uk = U[:, :k]
    P = torch.eye(d) - Uk @ Uk.T
    scale = 1.0
    if normalize == "frobenius":
        scale = float(P.norm())
        P = P / scale
    part = float((ev.sum() ** 2) / (ev ** 2).sum())
    return P, dict(d=d, k=k, k_t=k_t, k0=k0, rank=d - k,
                   energy_kept=float(cum[k - 1]), part_ratio=part,
                   frob_scale=scale, eig=ev.tolist())


def residual_ratio(H, P):
    """rho = mean fraction of energy that SURVIVES the projector (free space).
       Small rho => the projector kills almost everything => little room."""
    Hd = H.double()
    num = (Hd @ P.double().T).pow(2).sum(1)
    den = Hd.pow(2).sum(1).clamp_min(1e-12)
    return float((num / den).mean())

def make_proj_pre_hook(M):
    """forward_pre_hook that replaces proj's INPUT h with h @ M (M symmetric)."""
    def hook(module, inputs):
        h = inputs[0]
        return (h @ M.to(h.device, h.dtype),)
    return hook


def evaluate_through_subspace(model, M, gal_loader, prb_loader, device):
    """Evaluate with proj's input passed through the projector M."""
    h = model.backbone.proj.register_forward_pre_hook(make_proj_pre_hook(M))
    try:
        return evaluate_domain(model, gal_loader, prb_loader, device)
    finally:
        h.remove()
      
# ══════════════════════════════════════════════════════════════════════
#  BN pack  (per-domain state: gamma, beta, running mean/var)
# ══════════════════════════════════════════════════════════════════════

def snapshot_bn(model):
    pack = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            pack[name] = {
                "weight": m.weight.detach().clone(),
                "bias": m.bias.detach().clone(),
                "running_mean": m.running_mean.detach().clone(),
                "running_var": m.running_var.detach().clone(),
                "num_batches_tracked": m.num_batches_tracked.detach().clone(),
            }
    return pack


def restore_bn(model, pack):
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d) and name in pack:
            s = pack[name]
            m.weight.data.copy_(s["weight"])
            m.bias.data.copy_(s["bias"])
            m.running_mean.data.copy_(s["running_mean"])
            m.running_var.data.copy_(s["running_var"])
            m.num_batches_tracked.data.copy_(s["num_batches_tracked"])


# ══════════════════════════════════════════════════════════════════════
#  Projected Adam  (Adam-NSCL: project the UPDATE, not the gradient)
# ══════════════════════════════════════════════════════════════════════

class ProjectedAdam(torch.optim.Optimizer):
    """Adam whose update for registered params is right-multiplied by P:
           update <- update @ P          ([out,in] @ [in,in])
       mirroring optim/adam_svd.py of Adam-NSCL."""

    def __init__(self, param_groups, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0):
        defaults = dict(betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(param_groups, defaults)
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
                bc1 = 1 - b1 ** st["step"]
                bc2 = 1 - b2 ** st["step"]
                step_size = group["lr"] * math.sqrt(bc2) / bc1
                update = -step_size * st["exp_avg"] / denom
                Pm = self.projectors.get(id(p))
                if Pm is not None:                       # [out,in] @ [in,in]
                    update = update @ Pm
                p.add_(update)


# ══════════════════════════════════════════════════════════════════════
#  TENT loss + model configuration
# ══════════════════════════════════════════════════════════════════════

def entropy_loss(logits, conf_ratio):
    """Mean entropy over CONFIDENT samples (H < conf_ratio * ln C).
       Returns (loss, kept_fraction); loss is None if nothing passes."""
    p = logits.softmax(1)
    ent = -(p * p.clamp_min(1e-12).log()).sum(1)
    if conf_ratio >= 1.0:
        return ent.mean(), 1.0
    mask = ent < conf_ratio * math.log(logits.size(1))
    if mask.sum() == 0:
        return None, 0.0
    return ent[mask].mean(), float(mask.float().mean())


def configure_model(model, method, bn_mode):
    """Freeze everything, then re-enable exactly the intended surfaces.
         method 'tent'   -> BN affine only
         method 'nsctta' -> BN affine (optional) + proj.weight (projected)
       classifier and proj.bias are ALWAYS frozen."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    bn_params, proj_params = [], []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            if bn_mode == "adapt":
                m.train()                       # batch stats + running update
                m.weight.requires_grad_(True)
                m.bias.requires_grad_(True)
                bn_params += [m.weight, m.bias]
            else:
                m.eval()                        # frozen stats, frozen affine

    if method == "nsctta":
        w = model.backbone.proj.weight
        w.requires_grad_(True)
        proj_params.append(w)
        # proj.bias stays frozen: dy = dW x + db, and db cannot be projected.
    return bn_params, proj_params


# ══════════════════════════════════════════════════════════════════════
#  The TTA loop
# ══════════════════════════════════════════════════════════════════════

def run_tta(model, loader, args, method, P=None, tag=""):
    bn_params, proj_params = configure_model(model, method, args.bn_mode)

    groups = []
    if bn_params:
        groups.append({"params": bn_params, "lr": args.lr})
    if proj_params:
        groups.append({"params": proj_params, "lr": args.proj_lr})
    if not groups:
        print(f"    [{tag}] nothing trainable — skipping adaptation")
        return {"steps": 0, "kept": 0.0, "loss": float("nan")}

    opt = ProjectedAdam(groups)
    if method == "nsctta" and P is not None:
        for w in proj_params:
            opt.set_projector(w, P.to(args.device))

    n_ep = 1 if args.adapt_mode == "batch" else args.n_epochs
    n_bn = len(bn_params) // 2
    print(f"    [{tag}] trainable: {n_bn} BN layers"
          f"{' + proj.weight (PROJECTED)' if proj_params else ''}"
          f" | mode={args.adapt_mode} epochs={n_ep} bn={args.bn_mode}")

    steps, tot_loss, tot_kept, skipped = 0, 0.0, 0.0, 0
    for ep in range(n_ep):
        ep_loss, ep_kept, ep_steps = 0.0, 0.0, 0
        for x, _ in loader:
            x = x.to(args.device)
            logits, _feat = model(x)
            loss, kept = entropy_loss(logits, args.conf_ratio)
            if loss is None:
                skipped += 1
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ep_loss += loss.item(); ep_kept += kept; ep_steps += 1
        if ep_steps:
            steps += ep_steps; tot_loss += ep_loss; tot_kept += ep_kept
            if n_ep > 1:
                print(f"      ep {ep+1}/{n_ep}: H={ep_loss/ep_steps:.4f}  "
                      f"kept={100*ep_kept/ep_steps:.1f}%")
    if steps == 0:
        print(f"    [{tag}] WARNING: every batch filtered out "
              f"(conf_ratio={args.conf_ratio} too strict)")
        return {"steps": 0, "kept": 0.0, "loss": float("nan")}
    print(f"    [{tag}] {steps} updates | mean H={tot_loss/steps:.4f} | "
          f"kept={100*tot_kept/steps:.1f}% | skipped batches={skipped}")
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
          f"target={args.target_spectrum}")
    print(f"{'='*78}")

    # ── P0 ────────────────────────────────────────────────────────────
    print(f"\n{'─'*78}\n  P0  Source model + closed-set splits\n{'─'*78}")
    all_samples = scan_dataset(args.data_dir)
    model, arch, train_id_map, n_cls = load_source_model(args, all_samples)
    img_size = arch["img_size"]

    known = set(train_id_map)                       # closed-set restriction
    src_all = [s for s in all_samples
               if s["spectrum"] == args.source_spectrum and s["identity"] in known]
    tgt_all = [s for s in all_samples
               if s["spectrum"] == args.target_spectrum and s["identity"] in known]
    if not tgt_all:
        raise SystemExit(f"no target samples for {args.target_spectrum}")

    src_adapt, src_test = split_adapt_test(src_all, args.adapt_ratio, args.seed)
    tgt_adapt, tgt_test = split_adapt_test(tgt_all, args.adapt_ratio, args.seed)
    s_gal, s_prb = split_gallery_probe(src_test, train_id_map,
                                       args.gallery_ratio, args.seed)
    t_gal, t_prb = split_gallery_probe(tgt_test, train_id_map,
                                       args.gallery_ratio, args.seed)

    print(f"  source '{args.source_spectrum}': {len(src_all)} imgs -> "
          f"adapt {len(src_adapt)} (unused) | test {len(src_test)} "
          f"(gal {len(s_gal)} / prb {len(s_prb)})")
    print(f"  target '{args.target_spectrum}': {len(tgt_all)} imgs -> "
          f"adapt {len(tgt_adapt)} | test {len(tgt_test)} "
          f"(gal {len(t_gal)} / prb {len(t_prb)})")
    print(f"  NOTE: source data is an ORACLE DIAGNOSTIC only — never adapted on.")

    L = lambda s, sh=False: make_loader(s, train_id_map, img_size, args, sh)
    tgt_adapt_loader = L(tgt_adapt, True)           # shuffled TTA stream
    tgt_adapt_eval = L(tgt_adapt)                   # deterministic, for cov
    t_gal_l, t_prb_l = L(t_gal), L(t_prb)
    s_gal_l, s_prb_l = L(s_gal), L(s_prb)

    # ── P1 ────────────────────────────────────────────────────────────
    print(f"\n{'─'*78}\n  P1  Baselines (frozen source model)\n{'─'*78}")
    base_t = evaluate_domain(model, t_gal_l, t_prb_l, dev)
    base_s = evaluate_domain(model, s_gal_l, s_prb_l, dev)
    print(f"  TARGET  EER={base_t['eer']:6.2f}%  R1={base_t['rank1']:6.2f}%")
    print(f"  SOURCE  EER={base_s['eer']:6.2f}%  R1={base_s['rank1']:6.2f}%")
    R["baseline"] = {"target": base_t, "source": base_s}

    # ── P2 ────────────────────────────────────────────────────────────
    print(f"\n{'─'*78}\n  P2  Null-space projector from TARGET activations\n{'─'*78}")
    H_t = extract_pooled(model, tgt_adapt_eval, dev)
    P, info = build_projector(H_t, args.k0, args.energy, args.proj_normalize)
    print(f"  pooled activations (proj input): {tuple(H_t.shape)}")
    print(f"  participation ratio = {info['part_ratio']:.1f} / {info['d']}")
    print(f"  k_t@{args.energy:.0%} energy = {info['k_t']}   k0 = {info['k0']}"
          f"   ->  k = {info['k']}   free rank = {info['rank']}")
    print(f"  energy kept by top-{info['k']} = {info['energy_kept']:.4f}")
    if args.proj_normalize == "frobenius":
        print(f"  ||P||_F = {info['frob_scale']:.2f}  -> updates shrink ~"
              f"{1/info['frob_scale']:.3f}x  (hence --proj_lr {args.proj_lr})")

    H_s = extract_pooled(model, L(src_test), dev)   # oracle diagnostic
    rho_t, rho_s = residual_ratio(H_t, P), residual_ratio(H_s, P)
    print(f"  rho (energy surviving P):  target={rho_t:.4f}   source={rho_s:.4f}")
    print(f"    small source rho => P protects the source")
    print(f"    small target rho => little room to adapt (watch this)")
    R["projector"] = {k: v for k, v in info.items() if k != "eig"}
    R["projector"].update({"rho_target": rho_t, "rho_source": rho_s})

    # snapshot the SOURCE BN pack before any target batch touches BN
    src_bn_pack = snapshot_bn(model)
    print(f"  snapshotted source BN pack: {len(src_bn_pack)} layers")

    # ── P2b  Does the TARGET-derived subspace preserve the SOURCE? ──
    print(f"\n{'─'*78}\n  P2b  Functional check: SOURCE through TARGET-derived "
          f"subspace\n{'─'*78}")
    d = info["d"]; k = info["k"]
    # rebuild the un-normalised bases (P was Frobenius-scaled)
    C = (H_t.double().T @ H_t.double()) / H_t.size(0)
    ev_t, U_t = torch.linalg.eigh(C)
    U_t = U_t.flip(1).float()

    print(f"  {'k':>5} | {'src EER':>8} {'dEER':>7} | {'src R1':>8} {'dR1':>7} "
          f"| {'tgt EER':>8} {'tgt R1':>8}")
    curve = []
    for kk in [k_ for k_ in (4, 8, 16, 24, 32, 48, 64, 96, d) if k_ <= d]:
        Uk = U_t[:, :kk]
        M_keep = Uk @ Uk.T                       # project ONTO the kept subspace
        rs = evaluate_through_subspace(model, M_keep, s_gal_l, s_prb_l, dev)
        rt = evaluate_through_subspace(model, M_keep, t_gal_l, t_prb_l, dev)
        de = rs["eer"] - base_s["eer"]
        dr = rs["rank1"] - base_s["rank1"]
        curve.append({"k": kk, "src_eer": rs["eer"], "src_rank1": rs["rank1"],
                      "d_eer": de, "d_rank1": dr,
                      "tgt_eer": rt["eer"], "tgt_rank1": rt["rank1"]})
        mark = "  <-- k used" if kk == k else ""
        print(f"  {kk:5d} | {rs['eer']:8.2f} {de:+7.2f} | "
              f"{rs['rank1']:8.2f} {dr:+7.2f} | "
              f"{rt['eer']:8.2f} {rt['rank1']:8.2f}{mark}")

    eps_src = 0.5
    k_ok = next((c["k"] for c in curve
                 if c["d_eer"] <= eps_src and c["d_rank1"] >= -eps_src), d)
    print(f"  smallest k preserving source (±{args.eps_src}) = {k_ok}")
    print(f"  if source survives at k={k}, the target-built P0 spans what the "
          f"source needs => protection is justified")
    R["source_through_target_subspace"] = {"curve": curve, "k_ok": k_ok}

  
    # ── P3 / P4 ───────────────────────────────────────────────────────
    results = {}
    for arm, method in [("TENT", "tent"), ("NS-CTTA", "nsctta")]:
        phase = "P3" if method == "tent" else "P4"
        print(f"\n{'─'*78}\n  {phase}  Arm: {arm}\n{'─'*78}")
        m = copy.deepcopy(model)
        stats = run_tta(m, tgt_adapt_loader, args, method,
                        P=P if method == "nsctta" else None, tag=arm)
        r_t = evaluate_domain(m, t_gal_l, t_prb_l, dev)
        r_s = evaluate_domain(m, s_gal_l, s_prb_l, dev)
        print(f"    TARGET  EER={r_t['eer']:6.2f}% ({r_t['eer']-base_t['eer']:+5.2f})"
              f"   R1={r_t['rank1']:6.2f}% ({r_t['rank1']-base_t['rank1']:+5.2f})")
        print(f"    SOURCE  EER={r_s['eer']:6.2f}% ({r_s['eer']-base_s['eer']:+5.2f})"
              f"   R1={r_s['rank1']:6.2f}% ({r_s['rank1']-base_s['rank1']:+5.2f})")
        results[arm] = {"target": r_t, "source": r_s, "stats": stats}

        # ── P5: source recurrence — restore the source BN pack ──
        if args.bn_mode == "adapt":
            restore_bn(m, src_bn_pack)
            r_s2 = evaluate_domain(m, s_gal_l, s_prb_l, dev)
            print(f"    SOURCE after BN restore: EER={r_s2['eer']:6.2f}% "
                  f"({r_s2['eer']-base_s['eer']:+5.2f})   "
                  f"R1={r_s2['rank1']:6.2f}% ({r_s2['rank1']-base_s['rank1']:+5.2f})")
            results[arm]["source_bn_restored"] = r_s2

    # ── P6 ────────────────────────────────────────────────────────────
    print(f"\n{'='*78}\n  P6  RESULTS   (bn_mode={args.bn_mode}, "
          f"adapt_mode={args.adapt_mode})\n{'='*78}")
    hdr = (f"  {'model':<22}{'tgt EER':>9}{'tgt R1':>9}"
           f"{'src EER':>10}{'src R1':>9}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    print(f"  {'source (no TTA)':<22}{base_t['eer']:9.2f}{base_t['rank1']:9.2f}"
          f"{base_s['eer']:10.2f}{base_s['rank1']:9.2f}")
    for arm in ("TENT", "NS-CTTA"):
        r = results[arm]
        print(f"  {arm:<22}{r['target']['eer']:9.2f}{r['target']['rank1']:9.2f}"
              f"{r['source']['eer']:10.2f}{r['source']['rank1']:9.2f}")
        if "source_bn_restored" in r:
            b = r["source_bn_restored"]
            print(f"  {'  + BN restore':<22}{'—':>9}{'—':>9}"
                  f"{b['eer']:10.2f}{b['rank1']:9.2f}")

    print(f"\n  read: target columns = adaptation gain;  "
          f"source columns = forgetting")
    R["results"] = results
    jp = os.path.join(args.out_dir,
                      f"tta_{args.source_spectrum}_{args.target_spectrum}_"
                      f"{args.bn_mode}_{args.adapt_mode}.json")
    with open(jp, "w") as f:
        json.dump(R, f, indent=2, default=float)
    print(f"\n  saved {jp}\n{'='*78}\n")


if __name__ == "__main__":
    main()
