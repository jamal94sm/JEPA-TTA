"""
subspace_analysis.py — Source-model subspace extraction and analysis.

Characterizes the source model's feature subspace from a trained JEPA
checkpoint, offline, no further training.

SECTIONS
--------
Q1+Q2  Directional sweeps (UNIFIED): keep / remove the top-n directions under
       several RANKINGS (energy, Fisher-identity, Fisher-spectrum, random),
       reported as structured tables.
         keep  top-n  -> smallest subspace that RETAINS matching  -> n*
         remove top-n -> does dropping the strongest dirs hurt?   -> exposes
                         high-energy but task-irrelevant directions (e.g. DC)
Q3     Attribution: per-direction Fisher ratio vs identity and vs spectrum.
Q5     Target complement: is the free space usable by an unseen domain?
Q6     Calibration: why EER breaks while Rank-1 survives (DC + d').
Q7     Pseudo-Fisher: can label-free k-means prototypes replace label Fisher?

PROTOCOLS
---------
  within : gallery = target spectrum, probe = target spectrum
  cross  : gallery = SOURCE spectrum, probe = target spectrum   (realistic:
           enroll once at the source, probe after the shift)

USAGE
-----
python subspace_analysis.py \
    --data_dir /path/CASIA-MS-ROI \
    --ckpt ./output_jepa/ckpt_source_XXX.pth \
    --source_spectrum WHT --target_spectrum 940 \
    --protocol both --pseudo
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
    p.add_argument("--ckpt", required=True)
    p.add_argument("--source_spectrum", default="WHT")
    p.add_argument("--target_spectrum", default="940")
    p.add_argument("--protocol", default="both",
                   choices=["within", "cross", "both"])
    p.add_argument("--gallery_ratio", type=float, default=0.5)
    p.add_argument("--eps_eer", type=float, default=0.5)
    p.add_argument("--eps_r1", type=float, default=1.0)
    p.add_argument("--pseudo", action="store_true")
    p.add_argument("--n_clusters", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--out_dir", default="./output_subspace")
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
                             "--embed_dim, or re-save with the pretraining patch.")
        arch = {"img_size": args.img_size, "num_patches": args.num_patches,
                "embed_dim": args.embed_dim}

    method = ckpt.get("method", "jepa")          # default keeps old ckpts working

    if method == "compnet":
        from models import CompNetBackbone
        enc = CompNetBackbone(arch["embed_dim"],
                              base=arch.get("compnet_channels", 16))
        enc.load_state_dict(ckpt["backbone"])
        enc.to(args.device).eval()
        for q in enc.parameters():
            q.requires_grad = False
        print(f"  ckpt {os.path.basename(args.ckpt)}  method=compnet  "
              f"d={arch['embed_dim']}")
        fe = enc                                 # backbone IS the feature extractor
    else:                                        # jepa (unchanged path)
        enc = ContextEncoder((arch["img_size"], arch["img_size"]),
                             arch["num_patches"], arch["embed_dim"])
        enc.load_state_dict(ckpt["context_encoder"])
        enc.to(args.device).eval()
        for q in enc.parameters():
            q.requires_grad = False
        print(f"  ckpt {os.path.basename(args.ckpt)}  method=jepa  "
              f"d={arch['embed_dim']}")
        fe = FeatureExtractor(enc)

    return enc, fe, arch, ckpt


def make_loader(samples, id_map, img_size, args):
    ds = CASIADataset(samples, id_map, img_size, augment=False, aug_multiplier=1)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                      num_workers=args.num_workers, drop_last=False)


@torch.no_grad()
def extract_raw(fe, loader, device):
    """RAW (un-normalized) features; normalization happens AFTER projection."""
    feats, labels = [], []
    fe.eval()
    for x, y in loader:
        feats.append(fe(x.to(device)).cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


# ══════════════════════════════════════════════════════════════
#  Projectors — P = V V^T for an orthonormal column set V
# ══════════════════════════════════════════════════════════════

def projector(U, order, n, mode, d):
    """Projector for the n directions ranked highest by `order`.
       order : 1-D index array into U's columns, highest priority first
       mode  : 'keep'   -> project ONTO those n directions
               'remove' -> project onto the COMPLEMENT (I - that)"""
    V = U[:, order[:n]]
    P = V @ V.T
    return P if mode == "keep" else (torch.eye(d) - P)


# back-compat wrappers (energy prefix) so older call sites keep working
def keep_top_k(U, k):
    d = U.shape[0]
    return projector(U, np.arange(d), k, "keep", d)


def drop_top_n(U, n, d):
    return projector(U, np.arange(d), n, "remove", d)


def residual_energy(X, U, order, k):
    """rho = mean fraction of each sample's energy OUTSIDE the kept span."""
    V = U[:, order[:k]]
    inside = (X @ V).pow(2).sum(1)
    total = X.pow(2).sum(1).clamp_min(1e-12)
    return 1.0 - inside / total


def eval_proj(g_raw, g_lab, p_raw, p_lab, P=None):
    """
    Apply the SAME projector to gallery and probe, then normalize BOTH.

    Two-sided is mandatory: normalization breaks the identity
        (Pi x_p)^T x_g == (Pi x_p)^T (Pi x_g)
    because ||Pi x_g|| varies per gallery item and reorders the ranking.
    """
    g = g_raw if P is None else g_raw @ P
    p = p_raw if P is None else p_raw @ P
    return evaluate_rank1_eer(F.normalize(g, dim=-1), g_lab,
                              F.normalize(p, dim=-1), p_lab)


# ══════════════════════════════════════════════════════════════
#  Fisher / calibration
# ══════════════════════════════════════════════════════════════

def fisher_ratios(P, groups):
    """
    Per-direction 1-D Fisher discriminant ratio  F_j = S_b,j / S_w,j.

    High F_j -> direction j separates the groups.
    NOTE: F scales with the number of groups, so F_identity (many groups) and
    F_spectrum (2 groups) are NOT comparable in magnitude.
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


def score_stats(g_raw, g_lab, p_raw, p_lab, P=None):
    """d' over per-probe genuine/impostor means: measures whether scores are
       comparable ACROSS probes (what a single EER threshold requires)."""
    g = g_raw if P is None else g_raw @ P
    p = p_raw if P is None else p_raw @ P
    sim = F.normalize(p, dim=-1) @ F.normalize(g, dim=-1).T
    gen_m, imp_m = [], []
    for i in range(len(p_lab)):
        m = (g_lab == p_lab[i])
        if m.any():
            gen_m.append(sim[i][m].mean().item())
        if (~m).any():
            imp_m.append(sim[i][~m].mean().item())
    gen_m, imp_m = np.array(gen_m), np.array(imp_m)
    dp = (gen_m.mean() - imp_m.mean()) / np.sqrt(
        0.5 * (gen_m.var() + imp_m.var()) + 1e-12)
    return {"gen_mean": float(gen_m.mean()), "imp_mean": float(imp_m.mean()),
            "separation": float(gen_m.mean() - imp_m.mean()),
            "d_prime": float(dp)}


# ══════════════════════════════════════════════════════════════
#  Unified Q1+Q2 sweep and structured printing
# ══════════════════════════════════════════════════════════════

def run_sweeps(g, gl, p, pl, U, rankings, ns, d):
    """For every ranking x {keep, remove}: EER and R1 at each n.
       Returns {f'{rank}_{mode}': {'n','eer','rank1'}} plus n* for keep."""
    out, nstar = {}, {}
    for rname, order in rankings.items():
        for mode in ("keep", "remove"):
            eer, r1 = [], []
            for n in ns:
                if mode == "keep" and n >= d:
                    P = None
                elif mode == "remove" and n == 0:
                    P = None
                else:
                    P = projector(U, order, n, mode, d)
                r = eval_proj(g, gl, p, pl, P)
                eer.append(r["eer"]); r1.append(r["rank1"])
            out[f"{rname}_{mode}"] = {"n": list(ns), "eer": eer, "rank1": r1}
        # n* = smallest n whose KEEP retains baseline within tolerance
    return out


def find_nstar(sweeps, rankings, base, eps_eer, eps_r1, d):
    nstar = {}
    for rname in rankings:
        s = sweeps[f"{rname}_keep"]
        hit = [n for n, e, r in zip(s["n"], s["eer"], s["rank1"])
               if e <= base["eer"] + eps_eer and r >= base["rank1"] - eps_r1]
        nstar[rname] = min(hit) if hit else d
    return nstar


def _cells(vals, w=7):
    out = []
    for v in vals:
        out.append(" " * (w - 3) + "  —" if v is None else f"{v:{w}.2f}")
    return "".join(out)


def print_metric_table(title, sweeps, rankings, ns, metric, base_val, d):
    """One table: rows = ranking x {keep, remove}, cols = n, cells = metric."""
    w = 7
    print(f"\n  {title}    (baseline {base_val:6.2f})")
    head = "  " + f"{'ranking':<12}{'op':<8}" + "".join(
        f"{('n=' + str(n)):>{w}}" for n in ns)
    print(head)
    print("  " + "-" * (len(head) - 2))
    for rname in rankings:
        for mode in ("keep", "remove"):
            s = sweeps[f"{rname}_{mode}"]
            vals = []
            for n, v in zip(s["n"], s[metric]):
                # keep-all and remove-none are the untouched baseline; mark blank
                if (mode == "keep" and n >= d) or (mode == "remove" and n == 0):
                    vals.append(None)
                else:
                    vals.append(v)
            print(f"  {rname:<12}{mode:<8}{_cells(vals, w)}")


def print_nstar(nstar, base, d):
    print(f"\n  n* = smallest n whose KEEP-top-n retains baseline "
          f"(EER & Rank-1):")
    for rname, n in nstar.items():
        free = d - n
        print(f"    {rname:<12} n*={n:<4d}  free capacity = {free}/{d}")


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def main():
    args = get_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\n{'=' * 78}\n  SOURCE SUBSPACE ANALYSIS\n{'=' * 78}\n")

    enc, fe, arch, ckpt = load_source_model(args)
    d, dev = arch["embed_dim"], args.device

    all_samples = scan_dataset(args.data_dir)
    id_map = build_id_map(all_samples)
    src = [s for s in all_samples if s["spectrum"] == args.source_spectrum]
    tgt = [s for s in all_samples if s["spectrum"] == args.target_spectrum]
    if not src:
        raise SystemExit(f"no samples for spectrum {args.source_spectrum}")
    print(f"  source '{args.source_spectrum}': {len(src)}   "
          f"target '{args.target_spectrum}': {len(tgt)}")

    s_gal, s_prb = split_gallery_probe(src, id_map, args.gallery_ratio, args.seed)
    sg, sgl = extract_raw(fe, make_loader(s_gal, id_map, arch["img_size"], args), dev)
    sp, spl = extract_raw(fe, make_loader(s_prb, id_map, arch["img_size"], args), dev)
    print(f"  source gallery {tuple(sg.shape)}  probe {tuple(sp.shape)}")

    base = eval_proj(sg, sgl, sp, spl, None)
    base_sc = score_stats(sg, sgl, sp, spl, None)
    print(f"\n  BASELINE (source, within): EER={base['eer']:.2f}%  "
          f"R1={base['rank1']:.2f}%  d'={base_sc['d_prime']:.3f}")

    # ── C0 from GALLERY only -> eigendecomposition ──
    C0 = (sg.double().T @ sg.double()) / sg.size(0)
    evals, U = torch.linalg.eigh(C0)          # ASCENDING
    evals = evals.flip(0).clamp_min(0)        # -> DESCENDING
    U = U.flip(1).float()
    energy = (evals / evals.sum()).numpy()
    cum = np.cumsum(energy)
    order_energy = np.arange(d)               # eigh already sorted by energy
    part_ratio = float((evals.sum() ** 2) / (evals ** 2).sum())
    print(f"  participation ratio (effective rank) = {part_ratio:.1f} / {d}")

    # ── DC diagnostic ──
    xbar = sg.mean(0)
    dc_frac = float(xbar.pow(2).sum() / sg.pow(2).sum(1).mean())
    dc_align = float((U[:, 0] @ F.normalize(xbar, dim=0)).abs())
    print(f"\n  ── DC diagnostic ──")
    print(f"    energy in direction #0  = {energy[0]:.4f}")
    print(f"    |<u_0, xbar/||xbar||>|  = {dc_align:.4f}  "
          f"(~1 => direction #0 IS the mean direction)")
    print(f"    lambda_max={evals[0]:.4g}  lambda_min={evals[-1]:.4g}  "
          f"cond={float(evals[0] / evals[-1].clamp_min(1e-30)):.3g}")

    # ── Q3 attribution (needed to build Fisher rankings) ──
    fid = fdom = sel = None
    f_order = None
    shared = set()
    b_raw = Pj = None
    if tgt:
        src_ids = set(s["identity"] for s in src)
        tgt_ids = set(s["identity"] for s in tgt)
        shared = src_ids & tgt_ids
        both = [s for s in src + tgt if s["identity"] in shared]
        b_raw, b_lab = extract_raw(
            fe, make_loader(both, id_map, arch["img_size"], args), dev)
        dom_lab = torch.tensor(
            [0 if s["spectrum"] == args.source_spectrum else 1 for s in both])
        Pj = b_raw @ U
        fid = fisher_ratios(Pj, b_lab).numpy()
        fdom = fisher_ratios(Pj, dom_lab).numpy()
        sel = (fid - fdom) / (fid + fdom + 1e-12)
        f_order = np.argsort(-fid)
        print(f"\n  ── Q3 attribution ({len(shared)} shared IDs, "
              f"{len(both)} samples) ──")
        cc = float(np.corrcoef(np.arange(d), fid)[0, 1])
        print(f"    corr(energy_rank, F_id) = {cc:+.3f}  "
              f"(~0 => energy order is NOT task order)")
        print(f"    top-5 identity dirs (by F_id):  " +
              ", ".join(f"#{j}(E-rank {j})" for j in f_order[:5]))
        print(f"    top-5 spectrum dirs (by F_dom): " +
              ", ".join(f"#{j}" for j in np.argsort(-fdom)[:5]))

    # ── build rankings ──
    rankings = {"energy": order_energy}
    if fid is not None:
        rankings["fisher_id"] = np.argsort(-fid)
        rankings["fisher_dom"] = np.argsort(-fdom)
    rankings["random"] = np.random.default_rng(args.seed).permutation(d)

    # ══════════════════════════════════════════════════════════
    #  Q1 + Q2  — UNIFIED keep / remove sweeps
    # ══════════════════════════════════════════════════════════
    ns = [n for n in [1, 2, 4, 8, 16, 32, 64, 128, 256] if 1 <= n <= d]
    print(f"\n{'─' * 78}")
    print(f"  Q1+Q2  DIRECTIONAL SWEEPS  (d={d})")
    print(f"    keep  top-n : smallest subspace that RETAINS matching")
    print(f"    remove top-n: does dropping the strongest directions hurt?")
    print(f"{'─' * 78}")

    sweeps = run_sweeps(sg, sgl, sp, spl, U, rankings, ns, d)
    nstar = find_nstar(sweeps, rankings, base, args.eps_eer, args.eps_r1, d)

    print_metric_table("EER (%)  — lower is better", sweeps, rankings, ns,
                       "eer", base["eer"], d)
    print_metric_table("Rank-1 (%)  — higher is better", sweeps, rankings, ns,
                       "rank1", base["rank1"], d)
    print_nstar(nstar, base, d)

    # headline contrasts
    k_star = nstar["energy"]
    if "fisher_id" in nstar:
        m_star = nstar["fisher_id"]
        print(f"\n  contrast: k*(energy)={k_star}  vs  n*(Fisher-id)={m_star}"
              + ("   => discriminative selection frees more capacity"
                 if m_star < k_star else ""))
    # DC finding from the energy-remove row
    er = sweeps["energy_remove"]
    if er["eer"][0] < base["eer"]:
        print(f"  note: removing the top energy direction IMPROVES EER "
              f"({base['eer']:.2f}->{er['eer'][0]:.2f}); "
              f"it is DC/nuisance, not identity.")

    # ── Q6 calibration (DC removed) ──
    sc_nodc = score_stats(sg, sgl, sp, spl, drop_top_n(U, 1, d))
    print(f"\n  ── Q6 calibration ──")
    print(f"    with DC   : sep={base_sc['separation']:.4f}  d'={base_sc['d_prime']:.3f}")
    print(f"    DC removed: sep={sc_nodc['separation']:.4f}  d'={sc_nodc['d_prime']:.3f}")

    # ── Q7 pseudo-Fisher ──
    pseudo = None
    if args.pseudo and fid is not None:
        from sklearn.cluster import KMeans
        n_cl = args.n_clusters or len(shared)
        Xn = F.normalize(b_raw, dim=-1).numpy()
        km = KMeans(n_clusters=n_cl, n_init=10, random_state=args.seed).fit(Xn)
        pf = fisher_ratios(Pj, torch.tensor(km.labels_)).numpy()
        top_t = set(f_order[:k_star].tolist())
        top_p = set(np.argsort(-pf)[:k_star].tolist())
        ov = len(top_t & top_p) / max(k_star, 1)
        pseudo = {"n_clusters": n_cl,
                  "pearson": float(np.corrcoef(pf, fid)[0, 1]),
                  "spearman": _spearman(pf, fid),
                  "top_overlap": ov, "F_pseudo": pf.tolist()}
        print(f"\n  ── Q7 pseudo-Fisher (k-means, {n_cl} clusters, NO labels) ──")
        print(f"    corr(F_pseudo, F_id): pearson={pseudo['pearson']:+.3f}  "
              f"spearman={pseudo['spearman']:+.3f}   "
              f"top-{k_star} overlap={ov:.3f}")

    # ── Q5 target complement ──
    tgt_res = {}
    if tgt:
        t_gal, t_prb = split_gallery_probe(tgt, id_map, args.gallery_ratio,
                                           args.seed)
        tg, tgl = extract_raw(fe, make_loader(t_gal, id_map, arch["img_size"], args), dev)
        tp, tpl = extract_raw(fe, make_loader(t_prb, id_map, arch["img_size"], args), dev)
        protos = []
        if args.protocol in ("within", "both"):
            protos.append(("within", tg, tgl))
        if args.protocol in ("cross", "both"):
            protos.append(("cross", sg, sgl))
        for pname, G, GL in protos:
            tb = eval_proj(G, GL, tp, tpl, None)
            gal = args.target_spectrum if pname == "within" else args.source_spectrum
            print(f"\n  ── Q5 target [{pname}] gallery={gal} "
                  f"probe={args.target_spectrum} ──")
            print(f"    baseline: EER={tb['eer']:.2f}%  R1={tb['rank1']:.2f}%")
            r = {"k": [], "eer_in": [], "r1_in": [], "eer_free": [],
                 "r1_free": [], "rho_tgt": [], "rho_src": [], "baseline": tb}
            for k in ns:
                if k >= d:
                    continue
                a = eval_proj(G, GL, tp, tpl, projector(U, order_energy, k, "keep", d))
                b = eval_proj(G, GL, tp, tpl, projector(U, order_energy, k, "remove", d))
                rt = float(residual_energy(tp, U, order_energy, k).mean())
                rs = float(residual_energy(sp, U, order_energy, k).mean())
                r["k"].append(int(k))
                r["eer_in"].append(a["eer"]); r["r1_in"].append(a["rank1"])
                r["eer_free"].append(b["eer"]); r["r1_free"].append(b["rank1"])
                r["rho_tgt"].append(rt); r["rho_src"].append(rs)
                print(f"    k={k:3d} | inS0 EER={a['eer']:5.2f} R1={a['rank1']:5.2f}"
                      f" | free EER={b['eer']:5.2f} R1={b['rank1']:5.2f}"
                      f" | rho_t={rt:.4f} rho_s={rs:.4f} ratio={rt / max(rs, 1e-9):.2f}x")
            tgt_res[pname] = r

    # ══════════════════════════════════════════════════════════
    #  Figures
    # ══════════════════════════════════════════════════════════
    tag = args.source_spectrum
    figs = []
    colors = {"energy": "#B03A2E", "fisher_id": "#1E8449",
              "fisher_dom": "#D4820A", "random": "#7F8C8D"}

    # F1 — unified keep / remove, EER & R1, all rankings
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    for col, metric, lab in [(0, "eer", "EER (%)"), (1, "rank1", "Rank-1 (%)")]:
        for row, mode in enumerate(("keep", "remove")):
            a = ax[row, col]
            for rname in rankings:
                s = sweeps[f"{rname}_{mode}"]
                xs = [n for n in s["n"]
                      if not (mode == "keep" and n >= d)
                      and not (mode == "remove" and n == 0)]
                ys = [v for n, v in zip(s["n"], s[metric])
                      if not (mode == "keep" and n >= d)
                      and not (mode == "remove" and n == 0)]
                a.plot(xs, ys, "o-" if mode == "keep" else "s-",
                       color=colors.get(rname, "#555"), label=rname, ms=4)
            a.axhline(base[metric], ls="--", c="k", lw=1, label="baseline")
            if row == 0:
                a.axvline(k_star, ls="-.", c="#2E86C1", lw=1, label=f"k*={k_star}")
            a.set_xscale("log", base=2)
            a.set_xlabel(f"n {mode}"); a.set_ylabel(lab)
            a.set_title(f"{mode.upper()} top-n — {lab}")
            a.legend(fontsize=7); a.grid(alpha=.3)
    fig.suptitle(f"Q1+Q2 directional sweeps ({tag})")
    fig.tight_layout(); f = f"{args.out_dir}/fig1_sweeps_{tag}.png"
    fig.savefig(f, dpi=150, bbox_inches="tight"); plt.close(fig); figs.append(f)

    # F2 — spectrum / energy / DC
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].semilogy(np.arange(1, d + 1), evals.numpy().clip(1e-20), lw=1.5)
    ax[0].axvline(k_star, ls="-.", c="#2E86C1", label=f"k*={k_star}")
    ax[0].set_title("(a) C0 spectrum"); ax[0].set_xlabel("direction")
    ax[0].set_ylabel("eigenvalue"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].plot(np.arange(1, d + 1), cum, lw=1.5)
    ax[1].axvline(k_star, ls="-.", c="#2E86C1"); ax[1].axhline(.99, ls=":", c="gray")
    ax[1].set_title("(b) cumulative energy"); ax[1].set_xlabel("k"); ax[1].grid(alpha=.3)
    ax[2].bar(np.arange(8), energy[:8], color="#7F8C8D")
    ax[2].bar([0], energy[:1], color="#B03A2E",
              label=f"dir#0: {energy[0] * 100:.1f}%  |<u0,xbar>|={dc_align:.2f}")
    ax[2].set_title("(c) energy of leading dirs"); ax[2].set_xlabel("direction")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
    fig.tight_layout(); f = f"{args.out_dir}/fig2_spectrum_{tag}.png"
    fig.savefig(f, dpi=150, bbox_inches="tight"); plt.close(fig); figs.append(f)

    # F3 — attribution
    if fid is not None:
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        sc = ax[0].scatter(fdom, fid, c=np.log10(evals.numpy().clip(1e-20)),
                           cmap="viridis", s=18)
        ax[0].set_xlabel("F_spectrum (2 groups)")
        ax[0].set_ylabel("F_identity (many groups)")
        ax[0].set_title("(a) what each direction encodes\n(axes NOT on common scale)")
        plt.colorbar(sc, ax=ax[0], label="log10 eigenvalue"); ax[0].grid(alpha=.3)
        ax[1].plot(fid, lw=1.1, color="#1E8449", label="F_identity")
        ax[1].plot(fdom, lw=1.1, color="#B03A2E", label="F_spectrum")
        ax[1].axvline(k_star, ls="-.", c="#2E86C1", label=f"k*={k_star}")
        ax[1].set_xlabel("direction (ENERGY order)"); ax[1].set_ylabel("Fisher ratio")
        ax[1].set_title("(b) is energy order = task order?")
        ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
        ax[2].plot(np.sort(fid)[::-1], lw=1.3, color="#6C3483")
        ax[2].set_xlabel("direction (F_id order)"); ax[2].set_ylabel("F_identity")
        ax[2].set_title("(c) identity info concentration"); ax[2].grid(alpha=.3)
        fig.tight_layout(); f = f"{args.out_dir}/fig3_attribution_{tag}.png"
        fig.savefig(f, dpi=150, bbox_inches="tight"); plt.close(fig); figs.append(f)

    # F4 — calibration
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for j, (P, ttl, st) in enumerate([
            (None, "with DC", base_sc),
            (drop_top_n(U, 1, d), "DC removed", sc_nodc)]):
        g = sg if P is None else sg @ P
        pp = sp if P is None else sp @ P
        sim = F.normalize(pp, dim=-1) @ F.normalize(g, dim=-1).T
        gen, imp = [], []
        for i in range(len(spl)):
            m = (sgl == spl[i])
            gen += sim[i][m].tolist(); imp += sim[i][~m].tolist()
        ax[j].hist(imp, bins=60, alpha=.6, density=True, label="impostor", color="#B03A2E")
        ax[j].hist(gen, bins=60, alpha=.6, density=True, label="genuine", color="#1E8449")
        ax[j].set_title(f"({'ab'[j]}) {ttl}   d'={st['d_prime']:.3f}")
        ax[j].set_xlabel("cosine similarity"); ax[j].legend(fontsize=8); ax[j].grid(alpha=.3)
    fig.suptitle("EER needs ONE global threshold; Rank-1 only needs per-probe order")
    fig.tight_layout(); f = f"{args.out_dir}/fig4_calibration_{tag}.png"
    fig.savefig(f, dpi=150, bbox_inches="tight"); plt.close(fig); figs.append(f)

    # F5 — target, per protocol
    for pname, r in tgt_res.items():
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        ax[0].plot(r["k"], r["rho_tgt"], "o-", color="#B03A2E", label="target")
        ax[0].plot(r["k"], r["rho_src"], "s-", color="#1E8449", label="source")
        ax[0].axvline(k_star, ls="-.", c="#2E86C1", label=f"k*={k_star}")
        ax[0].set_xscale("log", base=2); ax[0].set_xlabel("k (size of S0)")
        ax[0].set_ylabel(r"$\rho$ (energy outside $S_0$)")
        ax[0].set_title(r"(a) $\rho$ detection signal")
        ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
        for j, m in enumerate(["eer", "r1"]):
            a = ax[j + 1]
            a.plot(r["k"], r[f"{m}_in"], "o-", color="#1E8449", label="into $S_0$")
            a.plot(r["k"], r[f"{m}_free"], "s-", color="#B03A2E", label="into FREE")
            a.axhline(r["baseline"]["eer" if m == "eer" else "rank1"],
                      ls="--", c="k", lw=1, label="target baseline")
            if m == "eer":
                a.axhline(50, ls=":", c="gray", lw=1, label="chance")
            a.axvline(k_star, ls="-.", c="#2E86C1")
            a.set_xscale("log", base=2); a.set_xlabel("k")
            a.set_ylabel("EER (%)" if m == "eer" else "Rank-1 (%)")
            a.set_title(f"({'bc'[j]}) target {m.upper()}")
            a.legend(fontsize=8); a.grid(alpha=.3)
        gal = args.target_spectrum if pname == "within" else args.source_spectrum
        fig.suptitle(f"Target {args.target_spectrum} — protocol={pname} "
                     f"(gallery={gal})")
        fig.tight_layout()
        f = f"{args.out_dir}/fig5_target_{args.target_spectrum}_{pname}.png"
        fig.savefig(f, dpi=150, bbox_inches="tight"); plt.close(fig); figs.append(f)

    # F6 — pseudo vs true Fisher
    if pseudo:
        pf = np.array(pseudo["F_pseudo"])
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].scatter(fid, pf, s=16, alpha=.7, color="#6C3483")
        ax[0].set_xlabel("F_identity (labels)"); ax[0].set_ylabel("F_pseudo (k-means)")
        ax[0].set_title(f"(a) pearson={pseudo['pearson']:+.3f}  "
                        f"spearman={pseudo['spearman']:+.3f}")
        ax[0].grid(alpha=.3)
        ax[1].plot(fid / max(fid.max(), 1e-9), lw=1.1, color="#1E8449",
                   label="F_identity (labels)")
        ax[1].plot(pf / max(pf.max(), 1e-9), lw=1.1, color="#B03A2E", alpha=.8,
                   label="F_pseudo (no labels)")
        ax[1].set_xlabel("direction (energy order)"); ax[1].set_ylabel("normalized F")
        ax[1].set_title(f"(b) top-{k_star} overlap = {pseudo['top_overlap']:.2f}")
        ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
        fig.suptitle("Can label-free prototypes replace label Fisher?")
        fig.tight_layout(); f = f"{args.out_dir}/fig6_pseudo_{tag}.png"
        fig.savefig(f, dpi=150, bbox_inches="tight"); plt.close(fig); figs.append(f)

    # ── dump ──
    out = {"ckpt": args.ckpt, "d": d, "protocol": args.protocol,
           "source_spectrum": args.source_spectrum,
           "target_spectrum": args.target_spectrum,
           "baseline": base, "baseline_scores": base_sc,
           "scores_dc_removed": sc_nodc,
           "participation_ratio": part_ratio,
           "dc": {"energy_dir0": float(energy[0]), "align_u0_xbar": dc_align},
           "eigenvalues": evals.numpy().tolist(),
           "energy_fraction": energy.tolist(),
           "sweeps": sweeps, "nstar": nstar,
           "k_star_energy": k_star,
           "attribution": (None if fid is None else
                           {"F_identity": fid.tolist(),
                            "F_spectrum": fdom.tolist(),
                            "selectivity": sel.tolist(),
                            "fisher_order": f_order.tolist()}),
           "pseudo": pseudo, "target": tgt_res}
    jp = f"{args.out_dir}/subspace_{tag}_{args.target_spectrum}.json"
    with open(jp, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n{'=' * 78}")
    print(f"  k*(energy)={k_star}/{d}   participation_ratio={part_ratio:.1f}   "
          f"free={d - k_star}")
    for f in figs + [jp]:
        print(f"  saved: {f}")
    print(f"{'=' * 78}\n")


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


if __name__ == "__main__":
    main()
