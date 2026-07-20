"""
subspace_reconstruct.py — Reconstructing the source subspace from TARGET data.

(Phase 1, the data-free weight route, is intentionally omitted: for a
residual + LayerNorm ViT there is no single feature-producing matrix W_feat, so
that route is not applicable to this model. Phase 2 is the sound approach.)

IDEA
----
We never touch source data to build the subspace. We push the TARGET (e.g. 940)
through the frozen source model, take the activation covariance, SVD it, and keep
the top-k directions such that removing the rest does not hurt the source model
ON THE TARGET. We then ask two questions:

  GEOMETRIC : how close is that target-built subspace to the TRUE source subspace
              (principal angles / Grassmann distance / overlap / energy)?
  FUNCTIONAL: if we project SOURCE data through the TARGET-built subspace, how
              much does source-model performance degrade? (two subspaces can
              overlap yet still break performance — this is the stronger test.)

SOURCE data (WHT) is used ONLY to (a) form the true baseline subspace and (b)
grade reconstruction. At deployment we would not have it.

USAGE
-----
python subspace_reconstruct.py \
    --data_dir /path/CASIA-MS-ROI \
    --ckpt ./output_jepa/ckpt_source_XXX.pth \
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
    p = argparse.ArgumentParser("Source subspace reconstruction (Phase 2)")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--source_spectrum", default="WHT")
    p.add_argument("--target_spectrum", default="940")
    p.add_argument("--gallery_ratio", type=float, default=0.5)
    p.add_argument("--eps_eer", type=float, default=0.5)
    p.add_argument("--eps_r1", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--out_dir", default="./output_reconstruct")
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
            raise SystemExit("ckpt has no 'arch'; pass --img_size --num_patches "
                             "--embed_dim, or re-save with the main.py patch.")
        arch = {"img_size": args.img_size, "num_patches": args.num_patches,
                "embed_dim": args.embed_dim}
    enc = ContextEncoder((arch["img_size"], arch["img_size"]),
                         arch["num_patches"], arch["embed_dim"])
    enc.load_state_dict(ckpt["context_encoder"])
    enc.to(args.device).eval()
    for q in enc.parameters():
        q.requires_grad = False
    print(f"  ckpt {os.path.basename(args.ckpt)}  epoch={ckpt.get('epoch','?')}"
          f"  d={arch['embed_dim']}")
    return enc, FeatureExtractor(enc), arch, ckpt


def make_loader(samples, id_map, img_size, args):
    ds = CASIADataset(samples, id_map, img_size, augment=False, aug_multiplier=1)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                      num_workers=args.num_workers, drop_last=False)


@torch.no_grad()
def extract_raw(fe, loader, device):
    feats, labels = [], []
    fe.eval()
    for x, y in loader:
        feats.append(fe(x.to(device)).cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


# ══════════════════════════════════════════════════════════════
#  Subspace utilities
# ══════════════════════════════════════════════════════════════

def cov_eig(X):
    """Uncentered covariance -> descending eigenpairs. X: [N, d]."""
    C = (X.double().T @ X.double()) / X.size(0)
    ev, U = torch.linalg.eigh(C)
    return ev.flip(0).clamp_min(0).float(), U.flip(1).float(), C.float()


def principal_angles(A, B):
    """Principal angles (deg) between column spaces of A, B."""
    A = torch.linalg.qr(A).Q if A.shape[1] > 0 else A
    B = torch.linalg.qr(B).Q if B.shape[1] > 0 else B
    s = torch.linalg.svdvals(A.T @ B).clamp(-1, 1)
    return torch.rad2deg(torch.arccos(s))


def grassmann(A, B):
    """Grassmann (geodesic) distance between equal-dim subspaces = ||theta||_2."""
    return float(torch.deg2rad(principal_angles(A, B)).norm())


def overlap(A, B):
    """Mean projection energy of A's directions captured by B's span, in [0,1].
       1 = identical span, 0 = orthogonal."""
    if A.shape[1] == 0 or B.shape[1] == 0:
        return 0.0
    Bo = torch.linalg.qr(B).Q
    return float((Bo @ (Bo.T @ A)).pow(2).sum() / A.shape[1])


def energy_recovered(U_true, ev_true, U_hat, k):
    """Fraction of TRUE top-k ENERGY captured by U_hat's top-k span."""
    At = U_true[:, :k]
    w = ev_true[:k]
    Bh = torch.linalg.qr(U_hat[:, :k]).Q
    proj = (Bh @ (Bh.T @ At)).pow(2).sum(0)
    return float((proj * w).sum() / w.sum())


def eval_through(U, k, g, gl, p, pl):
    """Project BOTH gallery and probe onto span(U[:, :k]), normalize BOTH, eval.
       Two-sided is required: one-sided projection + normalization reorders
       matches because ||P x_g|| varies per gallery item."""
    Uk = U[:, :k]
    P = Uk @ Uk.T
    g = F.normalize(g @ P, dim=-1)
    p = F.normalize(p @ P, dim=-1)
    return evaluate_rank1_eer(g, gl, p, pl)


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def main():
    args = get_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\n{'='*78}\n  SOURCE SUBSPACE RECONSTRUCTION FROM TARGET (Phase 2)\n"
          f"{'='*78}\n")

    enc, fe, arch, ckpt = load_source_model(args)
    d, dev = arch["embed_dim"], args.device

    all_samples = scan_dataset(args.data_dir)
    id_map = build_id_map(all_samples)
    src = [s for s in all_samples if s["spectrum"] == args.source_spectrum]
    tgt = [s for s in all_samples if s["spectrum"] == args.target_spectrum]
    if not tgt:
        raise SystemExit(f"no target samples for {args.target_spectrum}")
    print(f"  source '{args.source_spectrum}': {len(src)}   "
          f"target '{args.target_spectrum}': {len(tgt)}")

    # ── source features: TRUE reference (deployment would not have these) ──
    s_gal, s_prb = split_gallery_probe(src, id_map, args.gallery_ratio, args.seed)
    sg, sgl = extract_raw(fe, make_loader(s_gal, id_map, arch["img_size"], args), dev)
    sp, spl = extract_raw(fe, make_loader(s_prb, id_map, arch["img_size"], args), dev)
    ev_s, U_s, C_s = cov_eig(sg)

    # ── target features under the frozen source model ──
    t_gal, t_prb = split_gallery_probe(tgt, id_map, args.gallery_ratio, args.seed)
    tg, tgl = extract_raw(fe, make_loader(t_gal, id_map, arch["img_size"], args), dev)
    tp, tpl = extract_raw(fe, make_loader(t_prb, id_map, arch["img_size"], args), dev)
    ev_t, U_t, C_t = cov_eig(tg)

    part_s = float((ev_s.double().sum()**2) / (ev_s.double()**2).sum())
    part_t = float((ev_t.double().sum()**2) / (ev_t.double()**2).sum())
    s_base = evaluate_rank1_eer(F.normalize(sg, dim=-1), sgl,
                                F.normalize(sp, dim=-1), spl)
    t_base = evaluate_rank1_eer(F.normalize(tg, dim=-1), tgl,
                                F.normalize(tp, dim=-1), tpl)
    print(f"  source baseline: EER={s_base['eer']:.2f}%  R1={s_base['rank1']:.2f}%"
          f"   part.ratio={part_s:.1f}")
    print(f"  target baseline: EER={t_base['eer']:.2f}%  R1={t_base['rank1']:.2f}%"
          f"   part.ratio={part_t:.1f}")

    ns = [k for k in [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256]
          if 1 <= k <= d]

    # ══════════════════════════════════════════════════════════
    #  (A) pick k on the target's own subspace by target performance
    # ══════════════════════════════════════════════════════════
    print(f"\n  ── keep target top-k, evaluate TARGET (reconstruction size) ──")
    k_t = d
    keep_curve = []
    for k in ns:
        r = eval_through(U_t, k, tg, tgl, tp, tpl)
        keep_curve.append({"k": int(k), "eer": r["eer"], "rank1": r["rank1"]})
        if k_t == d and r["eer"] <= t_base["eer"] + args.eps_eer \
                and r["rank1"] >= t_base["rank1"] - args.eps_r1:
            k_t = k
    print(f"    k* (target retains own performance) = {k_t}")

    # ══════════════════════════════════════════════════════════
    #  (B) GEOMETRIC comparison: target-built subspace vs TRUE source
    # ══════════════════════════════════════════════════════════
    print(f"\n  ── GEOMETRIC: target-reconstructed vs TRUE source subspace ──")
    print(f"    {'k':>4} | {'p.angle':>8} {'grass':>7} {'overlap':>8} {'E_rec':>7}")
    geo = {}
    for k in ns:
        At, Bt = U_s[:, :k], U_t[:, :k]
        geo[str(k)] = {"angle": principal_angles(At, Bt).mean().item(),
                       "grassmann": grassmann(At, Bt),
                       "overlap": overlap(At, Bt),
                       "energy_recovered": energy_recovered(U_s, ev_s, U_t, k)}
        if k in (4, 8, 16, 32, 64, 128):
            gk = geo[str(k)]
            print(f"    {k:>4} | {gk['angle']:7.1f}° {gk['grassmann']:7.3f} "
                  f"{gk['overlap']:8.3f} {gk['energy_recovered']:7.3f}")

    # ══════════════════════════════════════════════════════════
    #  (C) FUNCTIONAL: SOURCE data through the TARGET-built subspace
    #      (the stronger test — does U_t preserve what the model does on WHT?)
    # ══════════════════════════════════════════════════════════
    print(f"\n  ── FUNCTIONAL: SOURCE data projected through TARGET subspace ──")
    print(f"    source baseline: EER={s_base['eer']:.2f}  R1={s_base['rank1']:.2f}")
    print(f"    {'k':>4} | {'EER':>6} {'dEER':>6} | {'R1':>6} {'dR1':>7} | {'ovlp':>6}")
    src_thru_tgt = []
    k_src_ok = d
    for k in ns:
        r = eval_through(U_t, k, sg, sgl, sp, spl)      # SOURCE data, TARGET basis
        de = r["eer"] - s_base["eer"]                   # + = degraded
        dr = r["rank1"] - s_base["rank1"]               # - = degraded
        ov = geo[str(k)]["overlap"]
        src_thru_tgt.append({"k": int(k), "eer": r["eer"], "rank1": r["rank1"],
                             "d_eer": de, "d_rank1": dr, "overlap": ov})
        if k_src_ok == d and de <= args.eps_eer and dr >= -args.eps_r1:
            k_src_ok = k
        if k in (4, 8, 16, 32, 64, 128, 256):
            print(f"    {k:>4} | {r['eer']:6.2f} {de:+6.2f} | "
                  f"{r['rank1']:6.2f} {dr:+7.2f} | {ov:6.3f}")
    print(f"    k* (source preserved within tol) = {k_src_ok}")

    # ══════════════════════════════════════════════════════════
    #  (D) SYMMETRIC: TARGET data through the SOURCE subspace
    #      lined up against (C) to test whether the damage is symmetric
    # ══════════════════════════════════════════════════════════
    print(f"\n  ── SYMMETRIC: TARGET data projected through SOURCE subspace ──")
    print(f"    {'k':>4} | src->tgt-sub dEER | tgt->src-sub dEER  (drop when the")
    print(f"         |  (source degrades)|  (target degrades)   OTHER basis is used)")
    tgt_thru_src = []
    for k in ns:
        r = eval_through(U_s, k, tg, tgl, tp, tpl)      # TARGET data, SOURCE basis
        de = r["eer"] - t_base["eer"]
        dr = r["rank1"] - t_base["rank1"]
        tgt_thru_src.append({"k": int(k), "eer": r["eer"], "rank1": r["rank1"],
                             "d_eer": de, "d_rank1": dr})
        if k in (4, 8, 16, 32, 64, 128):
            s_de = next(x["d_eer"] for x in src_thru_tgt if x["k"] == k)
            print(f"    {k:>4} | {s_de:+16.2f} | {de:+16.2f}")
    # asymmetry summary at k*
    kk = min(k_t, 64)
    a_src = next(x["d_eer"] for x in src_thru_tgt if x["k"] == kk)
    a_tgt = next(x["d_eer"] for x in tgt_thru_src if x["k"] == kk)
    print(f"    at k={kk}:  source loses {a_src:+.2f} EER through target basis,  "
          f"target loses {a_tgt:+.2f} EER through source basis")
    if abs(a_src) > abs(a_tgt) + 1.0:
        print(f"    => ASYMMETRIC: target subspace is (roughly) a SUBSET of the "
              f"source's\n       (target lives inside source geometry; expected "
              f"for a shift into an easier domain).")
    elif abs(a_tgt) > abs(a_src) + 1.0:
        print(f"    => ASYMMETRIC the other way: source subspace misses target "
              f"structure.")
    else:
        print(f"    => roughly SYMMETRIC: the two subspaces cover each other "
              f"comparably.")

    # ══════════════════════════════════════════════════════════
    #  (E) connection operators (where target diverges from source geometry)
    # ══════════════════════════════════════════════════════════
    proj = (tg.double() @ U_s.double())                 # [N, d] onto source dirs
    tgt_energy_in_s = proj.pow(2).mean(0)
    ratio = (tgt_energy_in_s / ev_s.double().clamp_min(1e-12)).float()

    eps = 1e-4 * ev_s[0].double()
    Cs_inv_half = U_s.double() @ torch.diag(
        1.0 / (ev_s.double() + eps).sqrt()) @ U_s.double().T
    M = Cs_inv_half @ C_t.double() @ Cs_inv_half
    sig_M = torch.linalg.eigvalsh(M).flip(0).clamp_min(0).float()

    top_shift = torch.argsort(ratio, descending=True)[:8]
    print(f"\n  ── connection: where TARGET over-excites SOURCE geometry ──")
    print(f"    source dirs with highest tgt/src energy ratio:")
    for j in top_shift.tolist():
        print(f"      src dir #{j:3d}  ratio={ratio[j]:7.2f}  (src energy rank {j})")
    print(f"    whitened operator M: sigma_max={sig_M[0]:.3g}  "
          f"#{{sigma>1}}={int((sig_M>1).sum())}")

    # ── overall verdict ──
    ov_kt = overlap(U_s[:, :k_t], U_t[:, :k_t])
    print(f"\n  >>> overlap(true top-{k_t}, target top-{k_t}) = {ov_kt:.3f}")
    print(f"  >>> source through target-subspace stays within tol up to k*={k_src_ok}")

    # ══════════════════════════════════════════════════════════
    #  Figures
    # ══════════════════════════════════════════════════════════
    tag = f"{args.source_spectrum}_{args.target_spectrum}"

    # F1 — geometric + functional together
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    ax[0].plot(ns, [geo[str(k)]["overlap"] for k in ns], "o-", color="#1E8449",
               label="subspace overlap")
    ax[0].plot(ns, [geo[str(k)]["energy_recovered"] for k in ns], "s-",
               color="#2E86C1", label="energy recovered")
    ax[0].axvline(k_t, ls="-.", c="#B03A2E", label=f"k*={k_t}")
    ax[0].axhline(1, ls="--", c="k", lw=1)
    ax[0].set_xscale("log", base=2); ax[0].set_xlabel("k")
    ax[0].set_title("(a) GEOMETRIC: target-built vs true source")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3); ax[0].set_ylim(0, 1.05)

    axa = ax[1]
    axa.plot(ns, [x["eer"] for x in src_thru_tgt], "o-", color="#B03A2E",
             label="source EER (thru target sub)")
    axa.axhline(s_base["eer"], ls="--", c="#B03A2E", lw=1, label="source baseline")
    axb = axa.twinx()
    axb.plot(ns, [x["rank1"] for x in src_thru_tgt], "s-", color="#1E8449",
             label="source R1")
    axb.axhline(s_base["rank1"], ls="--", c="#1E8449", lw=1)
    axa.axvline(k_src_ok, ls="-.", c="#2E86C1", label=f"k*={k_src_ok}")
    axa.set_xscale("log", base=2); axa.set_xlabel("k")
    axa.set_ylabel("source EER (%)", color="#B03A2E")
    axb.set_ylabel("source Rank-1 (%)", color="#1E8449")
    axa.set_title("(b) FUNCTIONAL: source thru target subspace")
    axa.legend(fontsize=7, loc="center right"); axa.grid(alpha=.3)

    ax[2].plot(ns, [x["d_eer"] for x in src_thru_tgt], "o-", color="#B03A2E",
               label="source loses (target basis)")
    ax[2].plot(ns, [x["d_eer"] for x in tgt_thru_src], "s-", color="#1E8449",
               label="target loses (source basis)")
    ax[2].axhline(0, ls="--", c="k", lw=1)
    ax[2].set_xscale("log", base=2); ax[2].set_xlabel("k")
    ax[2].set_ylabel("Δ EER (%)  (+ = degraded)")
    ax[2].set_title("(c) SYMMETRIC damage")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
    fig.suptitle(f"Phase 2 — reconstruct source subspace from target ({tag})")
    fig.tight_layout(); f = f"{args.out_dir}/phase2_{tag}.png"
    fig.savefig(f, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {f}")

    # F2 — spectra + divergence
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].semilogy(ev_s.numpy().clip(1e-20), label="source (true)")
    ax[0].semilogy(ev_t.numpy().clip(1e-20), label="target (in src model)")
    ax[0].set_xlabel("direction"); ax[0].set_ylabel("eigenvalue")
    ax[0].set_title("(a) spectra"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].plot(ratio.numpy(), lw=1)
    ax[1].axhline(1, ls="--", c="k", lw=1, label="tgt = src")
    ax[1].axhline(2, ls=":", c="#B03A2E", lw=1, label="shift (>2x)")
    ax[1].set_xlabel("source direction j"); ax[1].set_ylabel("tgt/src energy")
    ax[1].set_yscale("log"); ax[1].set_title("(b) where target diverges")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    fig.tight_layout(); f = f"{args.out_dir}/phase2_divergence_{tag}.png"
    fig.savefig(f, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {f}")

    # ── dump ──
    result = {
        "d": d, "source_spectrum": args.source_spectrum,
        "target_spectrum": args.target_spectrum,
        "source_baseline": s_base, "target_baseline": t_base,
        "participation_ratio": {"source": part_s, "target": part_t},
        "source_eig": ev_s.tolist(), "target_eig": ev_t.tolist(),
        "k_target": k_t, "k_source_ok": k_src_ok,
        "overlap_at_ktarget": ov_kt,
        "target_keep_curve": keep_curve,
        "geometric": geo,
        "source_through_target": src_thru_tgt,
        "target_through_source": tgt_thru_src,
        "ratio_source_dirs": ratio.tolist(),
        "M_spectrum": sig_M.tolist(),
    }
    jp = f"{args.out_dir}/reconstruct_{tag}.json"
    with open(jp, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n{'='*78}\n  saved {jp}\n{'='*78}\n")


if __name__ == "__main__":
    main()
