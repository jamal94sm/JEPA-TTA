"""
calibrate_domain_gap.py

Measures real pixel-statistic gaps between:
  - CASIA-MS spectral bands (within-dataset domain shift)
  - XJTU-UP device/condition combos (within-dataset domain shift)
  - CASIA-MS <-> XJTU-UP (cross-dataset domain shift)

...and converts each measured gap into a suggested value for the corruption
severity hyperparameters (color_temp_strength, gamma_strength, etc.) used in
MyFuncs.corrupt_visible_patches.

Usage:
    python calibrate_domain_gap.py \
        --casia_dir /home/pai-ng/Jamal/CASIA-MS-ROI \
        --xjtu_dir  /home/pai-ng/Jamal/XJTU-UP \
        --n_per_domain 300 \
        --out_json domain_gap_calibration.json \
        --out_plot domain_gap_calibration.png

Requires: numpy, scipy, Pillow, matplotlib (only for --out_plot).
"""
import argparse
import json
import os
import random
import re
from collections import defaultdict

import numpy as np
from PIL import Image
from scipy import ndimage


# --------------------------------------------------------------------------
# 1. Domain grouping / file discovery
# --------------------------------------------------------------------------

def scan_casia_domains(casia_dir, exts=(".jpg", ".jpeg", ".png", ".bmp")):
    """
    Groups CASIA-MS-ROI files by spectrum band, parsed from filenames of the
    form {subjectID}_{handSide}_{spectrum}_{iteration}.ext
    Adjust the regex if your actual filenames differ.
    """
    pattern = re.compile(r"^(?P<id>[^_]+)_(?P<hand>[^_]+)_(?P<spectrum>[^_]+)_(?P<iter>[^_.]+)")
    domains = defaultdict(list)
    for root, _, files in os.walk(casia_dir):
        for f in files:
            if not f.lower().endswith(exts):
                continue
            m = pattern.match(f)
            if not m:
                continue
            domains[f"CASIA_{m.group('spectrum')}"].append(os.path.join(root, f))
    return domains


def scan_xjtu_domains(xjtu_dir, exts=(".jpg", ".jpeg", ".png", ".bmp")):
    """
    Groups XJTU-UP files by device_condition, assuming a
    device/condition/id_folder/*.ext directory layout. Adjust if your layout differs.
    """
    domains = defaultdict(list)
    for root, _, files in os.walk(xjtu_dir):
        imgs = [f for f in files if f.lower().endswith(exts)]
        if not imgs:
            continue
        rel = os.path.relpath(root, xjtu_dir)
        parts = rel.split(os.sep)
        if len(parts) >= 2:
            device, condition = parts[0], parts[1]
        elif len(parts) == 1:
            device, condition = parts[0], "unknown"
        else:
            device, condition = "unknown", "unknown"
        key = f"XJTU_{device}_{condition}"
        for f in imgs:
            domains[key].append(os.path.join(root, f))
    return domains


# --------------------------------------------------------------------------
# 2. Per-domain statistics
# --------------------------------------------------------------------------

def _load_rgb(path, max_side=256):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return np.asarray(img).astype(np.float64) / 255.0  # (H, W, 3) in [0,1]


def _laplacian_var(gray):
    # Sharpness proxy: higher = sharper. Used to estimate device/optics blur gaps.
    return float(ndimage.laplace(gray).var())


def _noise_std_estimate(gray):
    """
    Robust high-frequency noise estimate: residual after a 3x3 median filter,
    scaled to a std via the MAD constant. Approximates sensor noise floor
    largely independent of image content/texture.
    """
    med = ndimage.median_filter(gray, size=3)
    resid = gray - med
    mad = np.median(np.abs(resid - np.median(resid)))
    return float(mad * 1.4826)


def _radial_falloff(gray):
    """
    Fits intensity ~ a - b*r^2 (r normalized to [0,1] from image center).
    Returns b: positive => corners darker than center (vignetting-like falloff).
    """
    H, W = gray.shape
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = H / 2.0, W / 2.0
    r = np.sqrt(((yy - cy) / (H / 2.0)) ** 2 + ((xx - cx) / (W / 2.0)) ** 2)
    r = np.clip(r, 0, 1).ravel()
    v = gray.ravel()
    A = np.stack([np.ones_like(r), r ** 2], axis=1)
    coef, *_ = np.linalg.lstsq(A, v, rcond=None)
    _, b = coef
    return float(-b)


def compute_domain_stats(paths, n_samples, seed=0):
    rng = random.Random(seed)
    sample_paths = paths if len(paths) <= n_samples else rng.sample(paths, n_samples)

    r_g_ratios, b_g_ratios, log_lums, sats = [], [], [], []
    sharpness, noise, vignette = [], [], []

    for p in sample_paths:
        try:
            img = _load_rgb(p)
        except Exception:
            continue
        R, G, B = img[..., 0], img[..., 1], img[..., 2]
        eps = 1e-4
        r_g_ratios.append(float((R.mean() + eps) / (G.mean() + eps)))
        b_g_ratios.append(float((B.mean() + eps) / (G.mean() + eps)))

        gray = 0.299 * R + 0.587 * G + 0.114 * B
        log_lums.append(float(np.log(gray.mean() + eps)))

        mx, mn = img.max(axis=-1), img.min(axis=-1)
        sat = np.where(mx > eps, (mx - mn) / (mx + eps), 0.0)
        sats.append(float(sat.mean()))

        sharpness.append(_laplacian_var(gray))
        noise.append(_noise_std_estimate(gray))
        vignette.append(_radial_falloff(gray))

    def _stat(vals):
        arr = np.asarray(vals, dtype=np.float64)
        return {"mean": float(arr.mean()), "std": float(arr.std()), "n": int(arr.size)}

    return {
        "r_g_ratio": _stat(r_g_ratios),
        "b_g_ratio": _stat(b_g_ratios),
        "log_luminance": _stat(log_lums),
        "saturation": _stat(sats),
        "sharpness_laplacian_var": _stat(sharpness),
        "noise_std": _stat(noise),
        "vignette_falloff": _stat(vignette),
        "_sample_paths": sample_paths,
    }


# --------------------------------------------------------------------------
# 3. Channel-mix matrix between two domains (distribution-level, unpaired)
# --------------------------------------------------------------------------

def estimate_channel_mix(stats_a, stats_b, n_pixels=20000, seed=0):
    """
    Fits a 3x3 matrix M via least squares that approximately maps domain A's
    RGB pixel distribution onto domain B's, using per-channel sorted-rank
    matching (no paired images needed since the two domains have different
    subjects/scenes). Returns M - I; its off-diagonal magnitude estimates
    channel_mix_strength (spectral crosstalk between channels/sensors).
    """
    rng = np.random.RandomState(seed)

    def _sample_pixels(paths, n):
        pix = []
        per_img = max(1, n // max(1, len(paths)))
        for p in paths:
            try:
                img = _load_rgb(p, max_side=128)
            except Exception:
                continue
            flat = img.reshape(-1, 3)
            idx = rng.choice(flat.shape[0], size=min(per_img, flat.shape[0]), replace=False)
            pix.append(flat[idx])
        return np.concatenate(pix, axis=0) if pix else np.zeros((0, 3))

    pa = _sample_pixels(stats_a["_sample_paths"], n_pixels)
    pb = _sample_pixels(stats_b["_sample_paths"], n_pixels)
    if len(pa) == 0 or len(pb) == 0:
        return np.zeros((3, 3))

    n = min(len(pa), len(pb))
    pa_sorted = np.sort(pa[:n], axis=0)
    pb_sorted = np.sort(pb[:n], axis=0)
    M, *_ = np.linalg.lstsq(pa_sorted, pb_sorted, rcond=None)
    return M.T - np.eye(3)


# --------------------------------------------------------------------------
# 4. Convert measured gaps -> corruption hyperparameters
# --------------------------------------------------------------------------

def stats_to_severity(stats_a, stats_b, mix_matrix):
    # color_temp_strength: relative shift in R/G and B/G channel ratios
    # (approximates illuminant / spectral-band / white-balance differences).
    dr = abs(stats_b["r_g_ratio"]["mean"] - stats_a["r_g_ratio"]["mean"]) / max(stats_a["r_g_ratio"]["mean"], 1e-4)
    db = abs(stats_b["b_g_ratio"]["mean"] - stats_a["b_g_ratio"]["mean"]) / max(stats_a["b_g_ratio"]["mean"], 1e-4)
    color_temp_strength = float(max(dr, db))

    # gamma_strength: converts log-luminance gap into an equivalent gamma
    # deviation (sensor exposure / tone-curve differences).
    la, lb = stats_a["log_luminance"]["mean"], stats_b["log_luminance"]["mean"]
    gamma_est = lb / la if abs(la) > 1e-6 else 1.0
    gamma_strength = float(abs(gamma_est - 1.0))

    # channel_mix_strength: mean |off-diagonal| of the fitted M - I
    # (spectral crosstalk / sensor response differences).
    off_diag = mix_matrix.copy()
    np.fill_diagonal(off_diag, 0.0)
    channel_mix_strength = float(np.abs(off_diag).mean()) if mix_matrix.size else 0.0

    # desaturate_strength: relative saturation drop A -> B
    # (relevant for CASIA's near-IR bands vs. WHT / XJTU's RGB captures).
    sat_a, sat_b = stats_a["saturation"]["mean"], stats_b["saturation"]["mean"]
    desaturate_strength = float(max(0.0, (sat_a - sat_b) / max(sat_a, 1e-4)))

    # blur_sigma_max: heuristic conversion of sharpness ratio to an
    # approximate Gaussian sigma (Var(Laplacian) ~ 1/sigma^4 for blurred
    # signals) — order-of-magnitude only; sanity-check visually.
    sharp_a = max(stats_a["sharpness_laplacian_var"]["mean"], 1e-8)
    sharp_b = max(stats_b["sharpness_laplacian_var"]["mean"], 1e-8)
    ratio = sharp_a / sharp_b
    blur_sigma_max = float(max(0.0, (max(ratio, 1.0) ** 0.25) - 1.0) * 2.0)

    # corruption_std: absolute gap in estimated sensor noise floor
    noise_gap = float(abs(stats_b["noise_std"]["mean"] - stats_a["noise_std"]["mean"]))

    # vignette_strength: gap in radial falloff coefficient
    # (acquisition-geometry / lens differences).
    vig_gap = float(abs(stats_b["vignette_falloff"]["mean"] - stats_a["vignette_falloff"]["mean"]))
    vignette_strength = float(np.clip(vig_gap, 0.0, 1.0))

    return {
        "color_temp_strength": round(min(color_temp_strength, 1.0), 4),
        "gamma_strength": round(min(gamma_strength, 1.0), 4),
        "channel_mix_strength": round(min(channel_mix_strength, 0.5), 4),
        "desaturate_strength": round(min(desaturate_strength, 1.0), 4),
        "blur_sigma_max": round(min(blur_sigma_max, 4.0), 4),
        "corruption_std": round(min(noise_gap, 0.3), 4),
        "vignette_strength": round(vignette_strength, 4),
    }


# --------------------------------------------------------------------------
# 5. Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--casia_dir", type=str, default=None)
    ap.add_argument("--xjtu_dir", type=str, default=None)
    ap.add_argument("--n_per_domain", type=int, default=300)
    ap.add_argument("--n_pixels_channel_mix", type=int, default=20000)
    ap.add_argument("--out_json", type=str, default="domain_gap_calibration.json")
    ap.add_argument("--out_plot", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    all_domains = {}
    if args.casia_dir:
        all_domains.update(scan_casia_domains(args.casia_dir))
    if args.xjtu_dir:
        all_domains.update(scan_xjtu_domains(args.xjtu_dir))
    if not all_domains:
        raise SystemExit("No domains found — check --casia_dir / --xjtu_dir paths.")

    print("Discovered domains:")
    for k, v in all_domains.items():
        print(f"  {k}: {len(v)} images")

    print("\nComputing per-domain statistics...")
    stats = {}
    for name, paths in all_domains.items():
        stats[name] = compute_domain_stats(paths, args.n_per_domain, seed=args.seed)
        print(f"  done: {name} (n={stats[name]['r_g_ratio']['n']})")

    domain_names = list(all_domains.keys())
    results = {"domains": {}, "pairs": {}}
    for name in domain_names:
        results["domains"][name] = {k: v for k, v in stats[name].items() if k != "_sample_paths"}

    print("\nComputing pairwise gaps + suggested severities...")
    for a in domain_names:
        for b in domain_names:
            if a == b:
                continue
            mix = estimate_channel_mix(stats[a], stats[b], n_pixels=args.n_pixels_channel_mix, seed=args.seed)
            results["pairs"][f"{a} -> {b}"] = stats_to_severity(stats[a], stats[b], mix)

    casia_names = [d for d in domain_names if d.startswith("CASIA_")]
    xjtu_names = [d for d in domain_names if d.startswith("XJTU_")]
    keys = ["color_temp_strength", "gamma_strength", "channel_mix_strength",
            "desaturate_strength", "blur_sigma_max", "corruption_std", "vignette_strength"]

    def _pairs_within(names):
        return [f"{a} -> {b}" for a in names for b in names if a != b]

    def _pairs_across(a_names, b_names):
        return [f"{a} -> {b}" for a in a_names for b in b_names] + \
               [f"{a} -> {b}" for a in b_names for b in a_names]

    def _agg(pair_keys, fn):
        vals_by_key = {k: [] for k in keys}
        for pk in pair_keys:
            if pk in results["pairs"]:
                for k in keys:
                    vals_by_key[k].append(results["pairs"][pk][k])
        return {k: (round(float(fn(vals_by_key[k])), 4) if vals_by_key[k] else None) for k in keys}

    if casia_names:
        results["suggested_within_casia"] = _agg(_pairs_within(casia_names), np.max)
    if xjtu_names:
        results["suggested_within_xjtu"] = _agg(_pairs_within(xjtu_names), np.max)
    if casia_names and xjtu_names:
        results["suggested_cross_dataset"] = _agg(_pairs_across(casia_names, xjtu_names), np.max)

    # Overall: cover the largest real gap observed anywhere. Since --*_strength
    # values are used as MAXIMUMS (actual per-sample severity ~ U(0, max)),
    # setting them to the worst-case gap still leaves milder shifts in the
    # training distribution too.
    scopes = [results.get(k) for k in
              ("suggested_within_casia", "suggested_within_xjtu", "suggested_cross_dataset")
              if results.get(k) is not None]
    if scopes:
        results["suggested_config"] = {
            k: round(float(np.max([d[k] for d in scopes if d[k] is not None])), 4)
            for k in keys
        }

    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved calibration report to {args.out_json}")

    if "suggested_config" in results:
        print("\nSuggested values:")
        for k, v in results["suggested_config"].items():
            print(f"  --{k} {v}")

    if args.out_plot:
        _plot_report(results, args.out_plot)
        print(f"Saved plot to {args.out_plot}")


def _plot_report(results, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [k for k in ("suggested_within_casia", "suggested_within_xjtu", "suggested_cross_dataset")
              if k in results]
    keys = ["color_temp_strength", "gamma_strength", "channel_mix_strength",
            "desaturate_strength", "blur_sigma_max", "corruption_std", "vignette_strength"]

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / max(len(groups), 1)
    x = np.arange(len(keys))
    for gi, g in enumerate(groups):
        vals = [results[g][k] if results[g][k] is not None else 0.0 for k in keys]
        ax.bar(x + gi * width, vals, width=width, label=g.replace("suggested_", ""))
    ax.set_xticks(x + width * (len(groups) - 1) / 2)
    ax.set_xticklabels(keys, rotation=30, ha="right")
    ax.set_ylabel("Suggested severity (max)")
    ax.legend()
    ax.set_title("Calibrated corruption severities by domain-gap scope")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    main()