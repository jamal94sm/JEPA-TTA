"""
subspace_analysis.py — Source-model subspace extraction and analysis.

Answers four questions, all offline, from a trained JEPA checkpoint:

  Q1  How many directions does the source model actually need for
      identity matching?          -> retention curve (keep top-k)
  Q2  Are the HIGH-ENERGY directions the important ones?
                                  -> ablation curve (remove top-N)
  Q3  Which directions carry IDENTITY and which carry SPECTRUM
      (illumination / sensor response)?
                                  -> per-direction Fisher attribution
  Q4  Is the free space (the d-k directions we would release to
      NS-CTTA) actually usable by a target domain?
                                  -> complementary target experiment

Usage
-----
python subspace_analysis.py \
    --data_dir /home/pai-ng/Jamal/CASIA-MS-ROI \
    --ckpt ./output_jepa/ckpt_source_CASIA-MS-ROI_cross_domain_WHT-940_8x.pth \
    --source_spectrum WHT --target_spectrum 940
"""

import os
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import ContextEncoder, FeatureExtractor
from dataset import (scan_dataset, build_id_map, split_gallery_probe,
                     CASIADataset)
from evaluate import evaluate_rank1_eer


# ══════════════════════════════════════════════════════════════
#  Args
# ══════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser("Source subspace analysis")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--ckpt", required=True,
                   help="ckpt_source_*.pth written by main.py")
    p.add_argument("--source_spectrum", default="WHT")
    p.add_argument("--target_spectrum", default="940")
    p.add_argument("--gallery_ratio", type=float, default=0.5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--eps_eer", type=float, default=0.5,
                   help="tolerance (EER pts) for choosing k*")
    p.add_argument("--out_dir", default="./output_subspace")
    # only needed if the checkpoint predates the 'arch' dict
    p.add_argument("--img_size", type=int, default=None)
    p.add_argument("--num_patches", type=int, default=None)
    p.add_argument("--embed_dim", type=int, default=None)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════
#  Model / features
# ══════════════════════════════════════════════════════════════

def load_source_model(args):
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    arch = ckpt.get("arch")
    if arch is None:
        if None in (args.img_size, args.num_patches, args.embed_dim):
            raise SystemExit(
                "Checkpoint has no 'arch' dict. Either re-save it with arch "
                "(see the main.py patch) or pass --img_size --num_patches "
                "--embed_dim explicitly.")
        arch = {"img_size": args.img_size,
                "num_patches": args.num_patches,
                "embed_dim": args.embed_dim}
        print("  ! no 'arch' in ckpt — using CLI overrides")

    enc = ContextEncoder((arch["img_size"], arch["img_size"]),
                         arch["num_patches"], arch["embed_dim"])
    enc.load_state_dict(ckpt["context_encoder"])
    enc.to(args.device).eval()
    for p in enc.parameters():
        p.requires_grad = False

    print(f"  loaded {args.ckpt}")
    print(f"    epoch={ckpt.get('epoch','?')}  "
          f"EER={ckpt.get('mean_eer', float('nan')):.2f}%  "
          f"R1={ckpt.get('mean_rank1', float('nan')):.2f}%")
    print(f"    arch: d={arch['embed_dim']}  "
          f"patches={arch['num_patches']}  img={arch['img_size']}")
    return FeatureExtractor(enc), arch, ckpt


def make_loader(samples, id_map, img_size, args):
    ds = CASIADataset(samples, id_map, img_size,
                      augment=False, aug_multiplier=1)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                      num_workers=args.num_workers, drop_last=False)


@torch.no_grad()
def extract_raw(fe, loader, device):
    """RAW (un-normalized) features. Normalization happens after projection."""
    feats, labels = [], []
    fe.eval()
    for x, y in loader:
        feats.append(fe(x.to(device)).cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


# ══════════════════════════════════════════════════════════════
#  Projection helpers
# ══════════════════════════════════════════════════════════════

def eval_with_projector(g_raw, g_lab, p_raw, p_lab, Pmat=None):
    """
    Apply the SAME projector to gallery and probe, then normalize BOTH.

    Two-sided is required: normalization breaks the identity
    (Pi x_p)^T x_g = (Pi x_p)^T (Pi x_g), because ||Pi x_g|| varies per
    gallery item and changes the ranking. In deployment both sides pass
    through the same adapter, so both must be projected.
    """
    if Pmat is None:
        g, p = g_raw, p_raw
    else:
        g, p = g_raw @ Pmat, p_raw @ Pmat
    g = F.normalize(g, dim=-1)
    p = F.normalize(p, dim=-1)
    return evaluate_rank1_eer(g, g_lab, p, p_lab)


def projector(U, order, n, mode, d):
    """Projector for the n directions ranked highest by `order`.
       order : 1-D index array into U's columns, highest priority first
       mode  : 'keep'   -> project ONTO those n directions
               'remove' -> project onto the COMPLEMENT (I - that)"""
    V = U[:, order[:n]]
    P = V @ V.T
    return P if mode == "keep" else (torch.eye(d) - P)


def residual_energy(X, U, k):
    """rho = mean fraction of each sample's energy OUTSIDE span(U[:, :k])."""
    Uk = U[:, :k]
    inside = (X @ Uk).pow(2).sum(1)
    total = X.pow(2).sum(1).clamp_min(1e-12)
    return (1.0 - inside / total)


# ══════════════════════════════════════════════════════════════
#  Per-direction attribution
# ══════════════════════════════════════════════════════════════

def fisher_ratios(P, groups):
    """
    Per-direction Fisher ratio: between-group scatter / within-group scatter.

    P      : [N, d] projections of every sample onto every direction
    groups : [N]    group id per sample (identity, or spectrum)
    returns: [d]    high = this direction separates the groups well
    """
    P = P.double()
    N, d = P.shape
    grand = P.mean(0)
    sb = torch.zeros(d, dtype=torch.float64)
    sw = torch.zeros(d, dtype=torch.float64)
    for g in torch.unique(groups):
        m = (groups == g)
        n_g = int(m.sum())
        if n_g < 2:
            continue
        Pg = P[m]
        mu = Pg.mean(0)
        sb += n_g * (mu - grand).pow(2)
        sw += (Pg - mu).pow(2).sum(0)
    return (sb / N) / ((sw / N) + 1e-12)


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n{'='*78}\n  SOURCE SUBSPACE ANALYSIS\n{'='*78}\n")

    # ── model ──
    fe, arch, ckpt = load_source_model(args)
    d = arch["embed_dim"]
    dev = args.device

    # ── data ──
    all_samples = scan_dataset(args.data_dir)
    id_map = build_id_map(all_samples)                 # global, as in main.py
    src = [s for s in all_samples if s["spectrum"] == args.source_spectrum]
    tgt = [s for s in all_samples if s["spectrum"] == args.target_spectrum]
    if not src:
        raise SystemExit(f"no samples for spectrum {args.source_spectrum}")
    print(f"\n  source '{args.source_spectrum}': {len(src)} samples")
    print(f"  target '{args.target_spectrum}': {len(tgt)} samples")

    s_gal, s_prb = split_gallery_probe(src, id_map, args.gallery_ratio,
                                       args.seed)
    t_gal, t_prb = split_gallery_probe(tgt, id_map, args.gallery_ratio,
                                       args.seed) if tgt else ([], [])

    sg_raw, sg_lab = extract_raw(
        fe, make_loader(s_gal, id_map, arch["img_size"], args), dev)
    sp_raw, sp_lab = extract_raw(
        fe, make_loader(s_prb, id_map, arch["img_size"], args), dev)
    print(f"  source gallery {tuple(sg_raw.shape)}  probe {tuple(sp_raw.shape)}")

    # ══════════════════════════════════════════════════════════
    #  Baseline
    # ══════════════════════════════════════════════════════════
    base = eval_with_projector(sg_raw, sg_lab, sp_raw, sp_lab, None)
    print(f"\n  BASELINE (no projection):  "
          f"EER={base['eer']:.2f}%   R1={base['rank1']:.2f}%")

    # ══════════════════════════════════════════════════════════
    #  C0 from GALLERY only -> eigendecomposition
    # ══════════════════════════════════════════════════════════
    C0 = (sg_raw.double().T @ sg_raw.double()) / sg_raw.size(0)
    evals, U = torch.linalg.eigh(C0)          # ASCENDING
    evals = evals.flip(0)                     # -> descending
    U = U.flip(1).float()                     # -> descending (columns)
    d = U.shape[0]
    order_energy = np.arange(d)                       # eigh already sorted desc
    part_ratio = float((evals.sum()**2) / (evals**2).sum())   # effective rank
    print(f"  participation ratio (effective rank) = {part_ratio:.1f} / {d}")
    evals = evals.clamp_min(0)
    energy = (evals / evals.sum()).numpy()
    cum_energy = np.cumsum(energy)
    print(f"  C0 rank≈{int((evals > 1e-10 * evals[0]).sum())}/{d}   "
          f"lambda_max={evals[0]:.4g}  lambda_min={evals[-1]:.4g}")

    ks = sorted(set([1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 80, 96, 128,
                     160, 192, 224, d]))
    ks = [k for k in ks if 1 <= k <= d]

    # ══════════════════════════════════════════════════════════
    #  Q3 — attribution: identity vs spectrum, per direction
    # ══════════════════════════════════════════════════════════
    print(f"\n  ── Q3  direction attribution ──")
    fid = fdom = None
    if tgt:
        # Control for identity: use only IDs present in BOTH spectra, so a
        # direction scoring high on 'spectrum' really is spectrum, not ID mix.
        src_ids = set(s["identity"] for s in src)
        tgt_ids = set(s["identity"] for s in tgt)
        shared = src_ids & tgt_ids
        both = [s for s in src + tgt if s["identity"] in shared]
        print(f"    {len(shared)} identities shared across "
              f"{args.source_spectrum}/{args.target_spectrum}  "
              f"({len(both)} samples)")

        b_raw, b_lab = extract_raw(
            fe, make_loader(both, id_map, arch["img_size"], args), dev)
        dom_lab = torch.tensor(
            [0 if s["spectrum"] == args.source_spectrum else 1 for s in both])

        Pj = b_raw @ U                      # [N, d] projections onto each direction
        fid = fisher_ratios(Pj, b_lab).numpy()
        fdom = fisher_ratios(Pj, dom_lab).numpy()
        sel = (fid - fdom) / (fid + fdom + 1e-12)

        order = np.argsort(-sel)
        print(f"    most IDENTITY-selective dirs (rank, sel, F_id, F_dom):")
        for j in order[:5]:
            print(f"      #{j:3d}  sel={sel[j]:+.3f}  F_id={fid[j]:.3f}  "
                  f"F_dom={fdom[j]:.3f}  energy_rank={j}")
        print(f"    most SPECTRUM-selective dirs:")
        for j in order[-5:][::-1]:
            print(f"      #{j:3d}  sel={sel[j]:+.3f}  F_id={fid[j]:.3f}  "
                  f"F_dom={fdom[j]:.3f}  energy_rank={j}")
        rho_e = float(np.corrcoef(np.arange(d), fid)[0, 1])
        print(f"    corr(energy_rank, F_id) = {rho_e:+.3f}   "
              f"(~0 => variance ordering is NOT task ordering)")
    else:
        sel = None
        print("    skipped (no target spectrum samples)")

    ### 
    order_fid  = np.argsort(-fid)      # most IDENTITY-discriminative first
    order_fdom = np.argsort(-fdom)     # most SPECTRUM/nuisance first
    order_rand = np.random.default_rng(args.seed).permutation(d)   # control
    
    RANKINGS = {
        "energy":     order_energy,    # what the covariance ranks by
        "fisher_id":  order_fid,       # what identity matching ranks by
        "fisher_dom": order_fdom,      # what illumination/sensor ranks by
        "random":     order_rand,      # control: does ranking matter at all?
    }

    # ══════════════════════════════════════════════════════════
    #  Q1 and Q2
    # ══════════════════════════════════════════════════════════
    ns = [n for n in [1,2,4,8,12,16,24,32,48,64,96,128,192,256] if 1 <= n <= d]
    sweeps, nstar = {}, {}
    
    for rank_name, order in RANKINGS.items():
        for mode in ("keep", "remove"):
            rows = []
            for n in ns:
                if   mode == "keep"   and n >= d: P = None      # keep-all = identity
                elif mode == "remove" and n == 0: P = None
                else:                             P = projector(U, order, n, mode, d)
                r = eval_with_projector(sg_raw, sg_lab, sp_raw, sp_lab, P)
                rows.append({"n": n, "eer": r["eer"], "rank1": r["rank1"]})
            sweeps[f"{rank_name}_{mode}"] = rows
    
            # smallest n reaching baseline (keep) / first n that breaks it (remove)
            if mode == "keep":
                hit = [x["n"] for x in rows if x["eer"] <= base["eer"] + args.eps_eer]
                nstar[f"{rank_name}_keep"] = min(hit) if hit else d
            print(f"  [{rank_name:11s} {mode:6s}] " +
                  "  ".join(f"{x['n']}:E{x['eer']:.1f}/R{x['rank1']:.0f}" for x in rows))
    
    print(f"\n  n* to retain baseline (keep):")
    for r in RANKINGS:
        print(f"    {r:11s} n*={nstar.get(r+'_keep')}")
    # ══════════════════════════════════════════════════════════
    #  Q4 — complementary: is the free space usable by the target?
    # ══════════════════════════════════════════════════════════
    print(f"\n  ── Q4  target '{args.target_spectrum}' complementary ──")
    tgt_res = None
    if tgt:
        tg_raw, tg_lab = extract_raw(
            fe, make_loader(t_gal, id_map, arch["img_size"], args), dev)
        tp_raw, tp_lab = extract_raw(
            fe, make_loader(t_prb, id_map, arch["img_size"], args), dev)
        t_base = eval_with_projector(tg_raw, tg_lab, tp_raw, tp_lab, None)
        print(f"    target baseline: EER={t_base['eer']:.2f}%  "
              f"R1={t_base['rank1']:.2f}%")

        tgt_res = {"k": [], "eer_inS0": [], "r1_inS0": [],
                   "eer_free": [], "r1_free": [],
                   "rho_target": [], "rho_source": []}
        for k in ks:
            if k >= d:
                continue
            Pin = keep_top_k(U, k)
            Pfree = drop_top_n(U, k, d)
            a = eval_with_projector(tg_raw, tg_lab, tp_raw, tp_lab, Pin)
            b = eval_with_projector(tg_raw, tg_lab, tp_raw, tp_lab, Pfree)
            rt = residual_energy(tp_raw, U, k).mean().item()
            rs = residual_energy(sp_raw, U, k).mean().item()
            tgt_res["k"].append(k)
            tgt_res["eer_inS0"].append(a["eer"]); tgt_res["r1_inS0"].append(a["rank1"])
            tgt_res["eer_free"].append(b["eer"]); tgt_res["r1_free"].append(b["rank1"])
            tgt_res["rho_target"].append(rt); tgt_res["rho_source"].append(rs)
            print(f"    k={k:3d} | in S0: EER={a['eer']:5.2f} R1={a['rank1']:5.2f}"
                  f" | free: EER={b['eer']:5.2f} R1={b['rank1']:5.2f}"
                  f" | rho_tgt={rt:.4f} rho_src={rs:.4f}")

        i = tgt_res["k"].index(k_star) if k_star in tgt_res["k"] else 0
        print(f"\n    At k*={tgt_res['k'][i]}:  rho_target="
              f"{tgt_res['rho_target'][i]:.4f}  "
              f"free-space target EER={tgt_res['eer_free'][i]:.2f}%")
        print(f"    (free-space EER near 50% => NS-CTTA has no usable room)")

    # ══════════════════════════════════════════════════════════
    #  Figures
    # ══════════════════════════════════════════════════════════
    tag = f"{args.source_spectrum}"

    # Fig 1 — retention + ablation
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    a = ax[0, 0]
    a.plot(keep["k"], keep["eer"], "o-", color="#C0392B", label="EER (keep top-k)")
    a.axhline(base["eer"], ls="--", c="k", lw=1, label=f"baseline {base['eer']:.2f}%")
    a.axhline(base["eer"] + args.eps_eer, ls=":", c="gray", lw=1,
              label=f"tolerance +{args.eps_eer}")
    a.axvline(k_star, ls="-.", c="#2E86C1", lw=1.5, label=f"k*={k_star}")
    a.set_xscale("log", base=2); a.set_xlabel("k (directions kept)")
    a.set_ylabel("EER (%)"); a.set_title("(a) Retention — EER vs k")
    a.legend(fontsize=8); a.grid(alpha=.3)

    a = ax[0, 1]
    a.plot(keep["k"], keep["rank1"], "o-", color="#1E8449", label="Rank-1")
    a.axhline(base["rank1"], ls="--", c="k", lw=1, label=f"baseline {base['rank1']:.2f}%")
    a.axvline(k_star, ls="-.", c="#2E86C1", lw=1.5, label=f"k*={k_star}")
    a.set_xscale("log", base=2); a.set_xlabel("k (directions kept)")
    a.set_ylabel("Rank-1 (%)"); a.set_title("(b) Retention — Rank-1 vs k")
    a.legend(fontsize=8); a.grid(alpha=.3)

    a = ax[1, 0]
    a.plot(drop["n"], drop["eer"], "s-", color="#C0392B")
    a.axhline(base["eer"], ls="--", c="k", lw=1, label="baseline")
    a.set_xlabel("N (leading directions REMOVED)"); a.set_ylabel("EER (%)")
    a.set_title("(c) Ablation — EER after removing top-N")
    a.legend(fontsize=8); a.grid(alpha=.3)

    a = ax[1, 1]
    a.plot(drop["n"], drop["rank1"], "s-", color="#1E8449")
    a.axhline(base["rank1"], ls="--", c="k", lw=1, label="baseline")
    a.set_xlabel("N (leading directions REMOVED)"); a.set_ylabel("Rank-1 (%)")
    a.set_title("(d) Ablation — Rank-1 after removing top-N")
    a.legend(fontsize=8); a.grid(alpha=.3)
    fig.suptitle(f"Source subspace: retention & ablation ({tag})", y=1.00)
    fig.tight_layout()
    f1 = os.path.join(args.out_dir, f"fig1_retention_ablation_{tag}.png")
    fig.savefig(f1, dpi=150, bbox_inches="tight"); plt.close(fig)

    # Fig 2 — spectrum + cumulative energy
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].semilogy(np.arange(1, d + 1), evals.numpy().clip(1e-20), lw=1.5)
    ax[0].axvline(k_star, ls="-.", c="#2E86C1", label=f"k*={k_star}")
    ax[0].set_xlabel("direction index"); ax[0].set_ylabel("eigenvalue")
    ax[0].set_title("(a) C0 spectrum"); ax[0].legend(); ax[0].grid(alpha=.3)

    ax[1].plot(np.arange(1, d + 1), cum_energy, lw=1.5)
    ax[1].axvline(k_star, ls="-.", c="#2E86C1", label=f"k*={k_star}")
    ax[1].axhline(0.99, ls=":", c="gray", label="99% energy")
    ax[1].set_xlabel("k"); ax[1].set_ylabel("cumulative energy fraction")
    ax[1].set_title("(b) Energy captured by top-k"); ax[1].legend()
    ax[1].grid(alpha=.3)
    fig.tight_layout()
    f2 = os.path.join(args.out_dir, f"fig2_spectrum_energy_{tag}.png")
    fig.savefig(f2, dpi=150, bbox_inches="tight"); plt.close(fig)

    # Fig 3 — attribution
    f3 = None
    if fid is not None:
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        sc = ax[0].scatter(fdom, fid, c=np.log10(evals.numpy().clip(1e-20)),
                           cmap="viridis", s=18)
        lim = max(fid.max(), fdom.max()) * 1.05
        ax[0].plot([0, lim], [0, lim], "k--", lw=1)
        ax[0].set_xlabel("F_spectrum (domain separability)")
        ax[0].set_ylabel("F_identity (identity separability)")
        ax[0].set_title("(a) What each direction encodes")
        plt.colorbar(sc, ax=ax[0], label="log10 eigenvalue")
        ax[0].text(.55, .1, "nuisance\n(illumination/sensor)", fontsize=8,
                   transform=ax[0].transAxes, color="#B03A2E")
        ax[0].text(.05, .85, "identity", fontsize=8,
                   transform=ax[0].transAxes, color="#1E8449")
        ax[0].grid(alpha=.3)

        ax[1].plot(fid, label="F_identity", lw=1.2, color="#1E8449")
        ax[1].plot(fdom, label="F_spectrum", lw=1.2, color="#B03A2E")
        ax[1].axvline(k_star, ls="-.", c="#2E86C1", label=f"k*={k_star}")
        ax[1].set_xlabel("direction index (sorted by ENERGY)")
        ax[1].set_ylabel("Fisher ratio")
        ax[1].set_title("(b) Is energy order = task order?")
        ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)

        ax[2].plot(sel, lw=1.0, color="#6C3483")
        ax[2].axhline(0, ls="--", c="k", lw=1)
        ax[2].axvline(k_star, ls="-.", c="#2E86C1", label=f"k*={k_star}")
        ax[2].set_xlabel("direction index (sorted by ENERGY)")
        ax[2].set_ylabel("selectivity  (F_id - F_dom)/(F_id + F_dom)")
        ax[2].set_title("(c) Identity-selective (+) vs nuisance (-)")
        ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
        fig.tight_layout()
        f3 = os.path.join(args.out_dir, f"fig3_attribution_{tag}.png")
        fig.savefig(f3, dpi=150, bbox_inches="tight"); plt.close(fig)

    # Fig 4 — target complementary
    f4 = None
    if tgt_res:
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        ax[0].plot(tgt_res["k"], tgt_res["rho_target"], "o-",
                   label=f"target {args.target_spectrum}", color="#B03A2E")
        ax[0].plot(tgt_res["k"], tgt_res["rho_source"], "s-",
                   label=f"source {args.source_spectrum}", color="#1E8449")
        ax[0].axvline(k_star, ls="-.", c="#2E86C1", label=f"k*={k_star}")
        ax[0].set_xscale("log", base=2)
        ax[0].set_xlabel("k (size of S0)")
        ax[0].set_ylabel(r"$\rho$  = energy fraction OUTSIDE $S_0$")
        ax[0].set_title(r"(a) $\rho$ diagnostic")
        ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

        ax[1].plot(tgt_res["k"], tgt_res["eer_inS0"], "o-",
                   label="target projected INTO S0", color="#1E8449")
        ax[1].plot(tgt_res["k"], tgt_res["eer_free"], "s-",
                   label="target projected into FREE space", color="#B03A2E")
        ax[1].axhline(t_base["eer"], ls="--", c="k", lw=1, label="target baseline")
        ax[1].axhline(50, ls=":", c="gray", lw=1, label="chance (EER 50%)")
        ax[1].axvline(k_star, ls="-.", c="#2E86C1")
        ax[1].set_xscale("log", base=2); ax[1].set_xlabel("k")
        ax[1].set_ylabel("target EER (%)")
        ax[1].set_title("(b) Is the free space usable?")
        ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)

        ax[2].plot(tgt_res["k"], tgt_res["r1_inS0"], "o-",
                   label="INTO S0", color="#1E8449")
        ax[2].plot(tgt_res["k"], tgt_res["r1_free"], "s-",
                   label="FREE space", color="#B03A2E")
        ax[2].axhline(t_base["rank1"], ls="--", c="k", lw=1, label="target baseline")
        ax[2].axvline(k_star, ls="-.", c="#2E86C1")
        ax[2].set_xscale("log", base=2); ax[2].set_xlabel("k")
        ax[2].set_ylabel("target Rank-1 (%)")
        ax[2].set_title("(c) Target Rank-1")
        ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
        fig.tight_layout()
        f4 = os.path.join(args.out_dir, f"fig4_target_{args.target_spectrum}.png")
        fig.savefig(f4, dpi=150, bbox_inches="tight"); plt.close(fig)

    # ══════════════════════════════════════════════════════════
    #  Dump
    # ══════════════════════════════════════════════════════════
    out = {
        "ckpt": args.ckpt,
        "source_spectrum": args.source_spectrum,
        "target_spectrum": args.target_spectrum,
        "d": d,
        "baseline": base,
        "k_star": k_star,
        "eps_eer": args.eps_eer,
        "eigenvalues": evals.numpy().tolist(),
        "energy_fraction": energy.tolist(),
        "retention": keep,
        "ablation": drop,
        "attribution": ({"F_identity": fid.tolist(),
                         "F_spectrum": fdom.tolist(),
                         "selectivity": sel.tolist()} if fid is not None else None),
        "target": tgt_res,
    }
    jp = os.path.join(args.out_dir, f"subspace_{tag}.json")
    with open(jp, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n{'='*78}")
    print(f"  k* = {k_star}/{d}   free capacity = {d - k_star}")
    for p in [f1, f2, f3, f4, jp]:
        if p:
            print(f"  saved: {p}")
    print(f"{'='*78}\n")


if __name__ == "__main__":
    main()
