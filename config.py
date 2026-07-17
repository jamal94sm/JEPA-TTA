"""
config.py — JEPA on CASIA-MS: 3 evaluation modes.
"""
import argparse


def get_cfg(args=None):
    p = argparse.ArgumentParser(description="JEPA on CASIA-MS")

    # ─── Dataset ──────────────────────────────────────────────
    
    #"data_root"        : "/home/pai-ng/Jamal/CASIA-MS-ROI",
    #"xjtu_data_root"   : "/home/pai-ng/Jamal/XJTU-UP",
    #"xpalm_data_root"  : "/home/pai-ng/Jamal/xpalm",
    
    p.add_argument("--data_dir", required=True, default = "/home/pai-ng/Jamal/CASIA-MS-ROI")
    p.add_argument("--img_size", type=int, default=112)

    # ─── Mode ─────────────────────────────────────────────────
    p.add_argument("--mode", default="all",
                   choices=["all", "cross_domain", "cross_domain_openset"],
                   help="'all' = all domains+IDs, "
                        "'cross_domain' = selected domains, all IDs, "
                        "'cross_domain_openset' = selected domains+IDs")
    p.add_argument("--train_spectrums", nargs="*", default=["WHT", "940"],
                   help="Spectrums for training (cross_domain modes)")
    p.add_argument("--train_id_ratio", type=float, default=0.8,
                   help="Fraction of IDs for training (openset mode)")
    p.add_argument("--test_sample_ratio", type=float, default=0.2,
                   help="Fraction of training samples held out for eval")
    p.add_argument("--gallery_ratio", type=float, default=0.5,
                   help="Fraction of test samples used as gallery")
    p.add_argument("--aug_multiplier", type=int, default=8,
                   help="Augmentation multiplier for training data")

    # ─── JEPA architecture ────────────────────────────────────
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--num_patches", type=int, default=8,
                   help="Grid size (8 → 8×8=64 patches for 112px)")
    p.add_argument("--num_blocks", type=int, default=2,
                   help="Number of target mask blocks")
    p.add_argument("--trg_ratio", type=float, nargs=2,
                   default=[0.10, 0.15])
    p.add_argument("--ctx_ratio", type=float, nargs=2,
                   default=[0.90, 1.00])

    # ─── Training ─────────────────────────────────────────────
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--start_lr", type=float, default=1e-5)
    p.add_argument("--final_lr", type=float, default=1e-6)
    p.add_argument("--final_weight_decay", type=float, default=0.4)
    p.add_argument("--ema_start", type=float, default=0.996)
    p.add_argument("--ema_end", type=float, default=1.0)

    # ─── Evaluation ───────────────────────────────────────────
    p.add_argument("--eval_every", type=int, default=5)

    # ─── Misc ─────────────────────────────────────────────────
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_dir", default="./output_jepa")

    return p.parse_args(args)
