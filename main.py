"""
main.py — JEPA pretraining on CASIA-MS with 3 evaluation modes.

Mode 1 (all):                  All domains × all IDs
Mode 2 (cross_domain):         Selected domains × all IDs
Mode 3 (cross_domain_openset): Selected domains × selected IDs
"""

import os
import json
import time
import random
import math
import numpy as np
import torch
import torch.nn.functional as F

from config import get_cfg
from dataset import build_datasets
from models import (ContextEncoder, TargetEncoder, Predictor,
                    FeatureExtractor, patchify, apply_masks,
                    repeat_interleave_batch, update_ema)
from evaluate import run_full_eval

from torch.utils.data import DataLoader
from dataset import build_datasets, CASIADataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():
    cfg = get_cfg()
    set_seed(cfg.seed)

    print(f"\n{'='*80}")
    print(f"  JEPA on CASIA-MS")
    print(f"  Mode: {cfg.mode}")
    print(f"  embed_dim={cfg.embed_dim}, patches={cfg.num_patches}, "
          f"blocks={cfg.num_blocks}")
    print(f"  epochs={cfg.epochs}, batch_size={cfg.batch_size}, "
          f"aug={cfg.aug_multiplier}×")
    print(f"{'='*80}\n")

    os.makedirs(cfg.output_dir, exist_ok=True)
  
    # ── Source checkpoint name ──
    ds_name = os.path.basename(os.path.normpath(cfg.data_dir))
    domains = "all" if cfg.mode == "all" else "-".join(cfg.train_spectrums)
    ckpt_name = (f"ckpt_source_{ds_name}_{cfg.mode}_{domains}_"
                 f"{cfg.aug_multiplier}x.pth")
    ckpt_path = os.path.join(cfg.output_dir, ckpt_name)
    print(f"  Source checkpoint → {ckpt_path}")
  
    # ── Build datasets ──
    train_loader, eval_dict, id_map, n_classes = build_datasets(cfg)
    img_size = (cfg.img_size, cfg.img_size)

    # ── Build models ──
    print(f"\n  Building models...")
    context_encoder = ContextEncoder(
        img_size, cfg.num_patches, cfg.embed_dim).to(cfg.device)
    target_encoder = TargetEncoder(
        img_size, cfg.num_patches, cfg.embed_dim).to(cfg.device)
    predictor = Predictor(
        cfg.num_patches, cfg.embed_dim).to(cfg.device)

    # Initialize target from context
    for pc, pt in zip(context_encoder.parameters(),
                      target_encoder.parameters()):
        pt.data.copy_(pc.data)
    for p in target_encoder.parameters():
        p.requires_grad = False

    n_ctx = sum(p.numel() for p in context_encoder.parameters())
    n_pred = sum(p.numel() for p in predictor.parameters())
    print(f"  Context encoder: {n_ctx/1e6:.2f}M params")
    print(f"  Predictor: {n_pred/1e6:.2f}M params")

    # ── Optimizer + schedulers ──
    opt = torch.optim.AdamW(
        list(context_encoder.parameters()) + list(predictor.parameters()),
        lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    total_steps = cfg.epochs * len(train_loader)

    # Warmup cosine LR
    warmup_steps = int(cfg.warmup_ratio * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return cfg.start_lr / cfg.learning_rate + \
                   (1 - cfg.start_lr / cfg.learning_rate) * step / warmup_steps
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return cfg.final_lr / cfg.learning_rate + \
                   (1 - cfg.final_lr / cfg.learning_rate) * \
                   0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # Momentum schedule
    def get_momentum(step):
        return cfg.ema_start + (cfg.ema_end - cfg.ema_start) * \
               step / max(1, total_steps)

    # ── Training ──
    print(f"\n{'─'*70}")
    print(f"  Training JEPA ({total_steps} steps)")
    print(f"{'─'*70}")

    feature_extractor = FeatureExtractor(context_encoder)
    global_step = 0
    eval_history = []
    best_eval = {"epoch": 0, "mean_rank1": 0.0, "mean_eer": float("inf")}
  
    for epoch in range(1, cfg.epochs + 1):
        context_encoder.train()
        predictor.train()
        target_encoder.eval()

        ep_loss = 0.0
        ep_var = 0.0
        n_bat = 0
        t0 = time.time()

        for images, _ in train_loader:
            images = images.to(cfg.device)
            B = images.size(0)

            ctx_masks, tgt_masks = patchify(
                B, cfg.num_patches, cfg.num_blocks,
                trg_ratio=tuple(cfg.trg_ratio),
                ctx_ratio=tuple(cfg.ctx_ratio),
                device=cfg.device)

            # Context encoder (masked)
            ctx_embeds = context_encoder(images, ctx_masks)

            # Feature variance monitoring
            with torch.no_grad():
                z_flat = ctx_embeds.reshape(-1, ctx_embeds.size(-1))
                ep_var += z_flat.var(dim=0).mean().item()

            # Target encoder (full, no grad)
            with torch.no_grad():
                tgt_full = target_encoder(images)
                tgt_embeds = apply_masks(tgt_full, tgt_masks)
                tgt_embeds = repeat_interleave_batch(
                    tgt_embeds, B, repeat=len(ctx_masks))

            # Predictor
            pred_embeds = predictor(ctx_embeds, ctx_masks, tgt_masks)

            # Loss
            loss = F.smooth_l1_loss(pred_embeds, tgt_embeds)

            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()

            # EMA update
            momentum = get_momentum(global_step)
            update_ema(context_encoder, target_encoder, momentum)

            global_step += 1
            ep_loss += loss.item()
            n_bat += 1

        ep_loss /= max(n_bat, 1)
        ep_var /= max(n_bat, 1)
        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]

        with torch.no_grad():
            sim = F.cosine_similarity(
                pred_embeds.reshape(-1, cfg.embed_dim),
                tgt_embeds.reshape(-1, cfg.embed_dim),
                dim=-1).mean().item()

        if epoch % 5 == 0 or epoch == cfg.epochs or epoch == 1:
            print(f"  ep {epoch:03d}/{cfg.epochs}  "
                  f"loss={ep_loss:.4f}  sim={sim:.3f}  "
                  f"var={ep_var:.4f}  lr={lr_now:.2e}  "
                  f"mom={momentum:.4f}  [{elapsed:.1f}s]")

        # ── Periodic evaluation ──
        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            print(f"\n  ── Eval at epoch {epoch} ──")
            context_encoder.eval()
            eval_results = run_full_eval(
                feature_extractor, eval_dict, cfg,
                tag=f"[ep{epoch}] ")

            eval_entry = {"epoch": epoch, "loss": ep_loss, "sim": sim}
            mean_r1 = np.mean([r["rank1"] for r in eval_results.values()])
            mean_eer = np.mean([r["eer"] for r in eval_results.values()])
            eval_entry["mean_rank1"] = mean_r1
            eval_entry["mean_eer"] = mean_eer
            for name, r in eval_results.items():
                eval_entry[name] = r
            eval_history.append(eval_entry)

            if mean_eer < best_eval["mean_eer"]:          # ← min EER now
                best_eval = {"epoch": epoch, "mean_rank1": mean_r1,
                             "mean_eer": mean_eer}
                torch.save({
                    "epoch": epoch,
                    "context_encoder": context_encoder.state_dict(),
                    "target_encoder": target_encoder.state_dict(),
                    "predictor": predictor.state_dict(),
                    "mean_rank1": mean_r1,
                    "mean_eer": mean_eer,                 # ← best EER saved
                    "arch": {                             # ← needed to rebuild
                        "img_size":    cfg.img_size,
                        "num_patches": cfg.num_patches,
                        "embed_dim":   cfg.embed_dim,
                    },
                    "mode": cfg.mode,
                    "train_spectrums": cfg.train_spectrums,
                    "aug_multiplier": cfg.aug_multiplier,
                    "seed": cfg.seed,
                }, ckpt_path)
                print(f"     New best EER={mean_eer:.2f}% "
                      f"(R1={mean_r1:.2f}%) → saved")

  
    # ══════════════════════════════════════════════════════════════
    #  Source subspace C0 — BEST weights, clean data, single pass
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print(f"  Computing C0 with best model (epoch {best_eval['epoch']}, "
          f"EER={best_eval['mean_eer']:.2f}%)")
    print(f"{'─'*70}")

    # 1. restore BEST weights — final-epoch weights are not the ones we ship
    ckpt = torch.load(ckpt_path, map_location=cfg.device, weights_only=False)
    context_encoder.load_state_dict(ckpt["context_encoder"])
    context_encoder.eval()
    feature_extractor = FeatureExtractor(context_encoder)

    # 2. clean loader: no augmentation, no shuffle, no dropped samples
    src_ds = train_loader.dataset                    # CASIADataset(augment=True)
    c0_ds = CASIADataset(src_ds.samples, src_ds.id_map, cfg.img_size,
                         augment=False, aug_multiplier=1)
    c0_loader = DataLoader(c0_ds, batch_size=cfg.batch_size, shuffle=False,
                           num_workers=cfg.num_workers, drop_last=False)

    # 3. accumulate the uncentered scatter (exact: batching is just reordering)
    d = cfg.embed_dim
    C0 = torch.zeros(d, d, dtype=torch.float64, device=cfg.device)
    n_feat = 0
    with torch.no_grad():
        for images, _ in c0_loader:
            x = feature_extractor(images.to(cfg.device))      # [B, d]
            # x = F.normalize(x, dim=-1)    # ← ONLY if NS-CTTA uses normalized x
            x = x.double()
            C0 += x.T @ x
            n_feat += x.size(0)
    C0 /= max(n_feat, 1)
    print(f"  C0 from {n_feat} clean source samples")

    # 4. diagnostic: how much of the d-dim space does the source occupy?
    ev = torch.linalg.eigvalsh(C0.cpu()).flip(0)     # descending
    for tau in [0.5, 0.1, 0.05, 0.01, 1e-3]:
        r0 = int((ev > tau * ev[0]).sum())
        print(f"    tau_eig={tau:<6} → r_0={r0:3d}/{d}   free={d-r0:3d}")

    # 5. attach C0 to the same checkpoint
    ckpt["C0"] = C0.cpu().float()
    ckpt["n_feat"] = n_feat
    ckpt["c0_augment"] = False
    ckpt["c0_normalized"] = False       # keep in sync with the F.normalize line
    torch.save(ckpt, ckpt_path)
    print(f"\n  Saved source model + C0 → {ckpt_path}")

  
    # ── Final summary ──
    print(f"\n{'='*80}")
    print(f"  TRAINING COMPLETE")
    print(f"  Best epoch: {best_eval['epoch']} "
          f"(R1={best_eval['mean_rank1']:.2f}%)")
    print(f"{'='*80}")

    # Print eval history table
    print(f"\n  {'Epoch':>6} {'Loss':>8} {'Sim':>6}", end="")
    eval_names = list(eval_dict.keys())
    for name in eval_names:
        print(f" │ {name[:12]:>12} R1   EER", end="")
    print()
    print(f"  {'─'*8}{'─'*8}{'─'*6}", end="")
    for _ in eval_names:
        print(f"─┼─{'─'*24}", end="")
    print()

    for entry in eval_history:
        print(f"  {entry['epoch']:>6} {entry['loss']:>8.4f} "
              f"{entry['sim']:>6.3f}", end="")
        for name in eval_names:
            if name in entry:
                r = entry[name]
                print(f" │ {r['rank1']:>6.2f} {r['eer']:>6.2f}", end="")
            else:
                print(f" │ {'---':>6} {'---':>6}", end="")
        print()

    # Save results
    save_path = os.path.join(cfg.output_dir,
                              f"jepa_{cfg.mode}_seed{cfg.seed}.json")
    with open(save_path, "w") as f:
        json.dump({
            "mode": cfg.mode,
            "config": {
                "embed_dim": cfg.embed_dim,
                "num_patches": cfg.num_patches,
                "epochs": cfg.epochs,
                "train_spectrums": cfg.train_spectrums,
                "aug_multiplier": cfg.aug_multiplier,
            },
            "best": best_eval,
            "history": eval_history,
        }, f, indent=2)
    print(f"\n  Saved: {save_path}")


if __name__ == "__main__":
    main()
