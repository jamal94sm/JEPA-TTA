"""
source_pretraining.py — Source-model pretraining with a method toggle.

  --method jepa     : transformer + self-supervised (I-JEPA)   [original path]
  --method compnet  : CompNet CNN + supervised cross-entropy on training IDs

Both paths share the same dataset pipeline and the same evaluation
(run_full_eval on the eval_dict), and both save a checkpoint whose backbone
produces [B, embed_dim] features — so all downstream subspace tooling works
unchanged. Point --output_dir somewhere method-specific so checkpoints do not
collide (e.g. ./output_jepa vs ./output_compnet).


python source_pretraining.py --method compnet --data_dir /home/pai-ng/Jamal/CASIA-MS-ROI --mode cross_domain_openset --train_spectrums WHT --output_dir ./output_compnet


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
                    repeat_interleave_batch, update_ema, CompNet)
from evaluate import run_full_eval


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def ckpt_name(cfg):
    """ckpt_{dataset}_{method}_{source_domain}.pth"""
    dataset = os.path.basename(os.path.normpath(cfg.data_dir)).lower()
    dataset = "casiams" if "casia" in dataset else ("xjtu" if "xjtu" in dataset
                                                    else dataset)
    domain = "-".join(cfg.train_spectrums) if cfg.train_spectrums else "all"
    return f"ckpt_{dataset}_{cfg.method}_{domain}.pth"


# ══════════════════════════════════════════════════════════════
#  Shared warmup-cosine LR schedule
# ══════════════════════════════════════════════════════════════

def make_scheduler(opt, cfg, total_steps):
    warmup_steps = int(cfg.warmup_ratio * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return cfg.start_lr / cfg.learning_rate + \
                   (1 - cfg.start_lr / cfg.learning_rate) * step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return cfg.final_lr / cfg.learning_rate + \
               (1 - cfg.final_lr / cfg.learning_rate) * \
               0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


# ══════════════════════════════════════════════════════════════
#  JEPA (self-supervised) — original training path, unchanged
# ══════════════════════════════════════════════════════════════

def train_jepa(cfg, train_loader, eval_dict, id_map, n_classes):
    img_size = (cfg.img_size, cfg.img_size)

    print(f"\n  Building JEPA models...")
    context_encoder = ContextEncoder(
        img_size, cfg.num_patches, cfg.embed_dim).to(cfg.device)
    target_encoder = TargetEncoder(
        img_size, cfg.num_patches, cfg.embed_dim).to(cfg.device)
    predictor = Predictor(
        cfg.num_patches, cfg.embed_dim).to(cfg.device)

    for pc, pt in zip(context_encoder.parameters(),
                      target_encoder.parameters()):
        pt.data.copy_(pc.data)
    for p in target_encoder.parameters():
        p.requires_grad = False

    n_ctx = sum(p.numel() for p in context_encoder.parameters())
    n_pred = sum(p.numel() for p in predictor.parameters())
    print(f"  Context encoder: {n_ctx/1e6:.2f}M params")
    print(f"  Predictor: {n_pred/1e6:.2f}M params")

    opt = torch.optim.AdamW(
        list(context_encoder.parameters()) + list(predictor.parameters()),
        lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * len(train_loader)
    scheduler = make_scheduler(opt, cfg, total_steps)

    def get_momentum(step):
        return cfg.ema_start + (cfg.ema_end - cfg.ema_start) * \
               step / max(1, total_steps)

    print(f"\n{'─'*70}")
    print(f"  Training JEPA ({total_steps} steps)")
    print(f"{'─'*70}")

    feature_extractor = FeatureExtractor(context_encoder)
    global_step = 0
    eval_history = []
    best_eval = {"epoch": 0, "mean_rank1": 0}

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

            ctx_embeds = context_encoder(images, ctx_masks)

            with torch.no_grad():
                z_flat = ctx_embeds.reshape(-1, ctx_embeds.size(-1))
                ep_var += z_flat.var(dim=0).mean().item()

            with torch.no_grad():
                tgt_full = target_encoder(images)
                tgt_embeds = apply_masks(tgt_full, tgt_masks)
                tgt_embeds = repeat_interleave_batch(
                    tgt_embeds, B, repeat=len(ctx_masks))

            pred_embeds = predictor(ctx_embeds, ctx_masks, tgt_masks)

            loss = F.smooth_l1_loss(pred_embeds, tgt_embeds)

            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()

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

            if mean_r1 > best_eval["mean_rank1"]:
                best_eval = {"epoch": epoch, "mean_rank1": mean_r1,
                             "mean_eer": mean_eer}
                ckpt_path = os.path.join(cfg.output_dir, ckpt_name(cfg))
                torch.save({
                    "epoch": epoch,
                    "method": "jepa",
                    "context_encoder": context_encoder.state_dict(),
                    "target_encoder": target_encoder.state_dict(),
                    "predictor": predictor.state_dict(),
                    "arch": {"embed_dim": cfg.embed_dim,
                             "num_patches": cfg.num_patches,
                             "img_size": cfg.img_size},
                    "mean_rank1": mean_r1,
                }, ckpt_path)
                print(f"    ★ New best R1={mean_r1:.2f}% "
                      f"EER={mean_eer:.2f}% → saved")

            print(f"    Summary: Mean R1={mean_r1:.2f}% | "
                  f"Mean EER={mean_eer:.2f}%\n")

    _print_history_jepa(eval_history, eval_dict)
    _print_footer(cfg, best_eval)

    save_path = os.path.join(cfg.output_dir,
                             f"jepa_{cfg.mode}_seed{cfg.seed}.json")
    with open(save_path, "w") as f:
        json.dump({
            "mode": cfg.mode, "method": "jepa",
            "config": {
                "embed_dim": cfg.embed_dim,
                "num_patches": cfg.num_patches,
                "epochs": cfg.epochs,
                "train_spectrums": cfg.train_spectrums,
                "aug_multiplier": cfg.aug_multiplier,
            },
            "best": best_eval, "history": eval_history,
        }, f, indent=2)
    print(f"\n  Saved: {save_path}")


# ══════════════════════════════════════════════════════════════
#  CompNet (supervised cross-entropy on training IDs)
# ══════════════════════════════════════════════════════════════

def train_compnet(cfg, train_loader, eval_dict, id_map, n_train_ids, train_id_map):
    print(f"\n  Building CompNet (supervised)...")
    model = CompNet(cfg.embed_dim, n_train_ids, base=cfg.compnet_channels).to(cfg.device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"  CompNet: {n_par/1e6:.2f}M params   n_classes={n_train_ids}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                            weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * len(train_loader)
    scheduler = make_scheduler(opt, cfg, total_steps)
    ce = torch.nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    # run_full_eval needs an object whose forward(x) -> [B, embed_dim];
    # for CompNet that is exactly the backbone (no FeatureExtractor wrapper).
    feature_extractor = model.backbone

    print(f"\n{'─'*70}")
    print(f"  Training CompNet ({total_steps} steps, CE on IDs)")
    print(f"{'─'*70}")

    global_step = 0
    eval_history = []
    best_eval = {"epoch": 0, "mean_rank1": 0.0, "mean_eer": float("inf")}

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        ep_loss, ep_correct, seen, n_bat = 0.0, 0, 0, 0
        t0 = time.time()

        for images, labels in train_loader:
            images = images.to(cfg.device)
            labels = labels.to(cfg.device)

            logits, _feat = model(images)
            loss = ce(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()

            global_step += 1
            ep_loss += loss.item()
            ep_correct += (logits.argmax(1) == labels).sum().item()
            seen += labels.size(0)
            n_bat += 1

        ep_loss /= max(n_bat, 1)
        ep_acc = 100.0 * ep_correct / max(seen, 1)
        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]

        if epoch % 5 == 0 or epoch == cfg.epochs or epoch == 1:
            print(f"  ep {epoch:03d}/{cfg.epochs}  CE={ep_loss:.4f}  "
                  f"train_acc={ep_acc:.2f}%  lr={lr_now:.2e}  [{elapsed:.1f}s]")

        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            print(f"\n  ── Eval at epoch {epoch} ──")
            model.eval()
            eval_results = run_full_eval(
                feature_extractor, eval_dict, cfg, tag=f"[ep{epoch}] ")

            eval_entry = {"epoch": epoch, "ce": ep_loss, "train_acc": ep_acc}
            mean_r1 = np.mean([r["rank1"] for r in eval_results.values()])
            mean_eer = np.mean([r["eer"] for r in eval_results.values()])
            eval_entry["mean_rank1"] = mean_r1
            eval_entry["mean_eer"] = mean_eer
            for name, r in eval_results.items():
                eval_entry[name] = r
            eval_history.append(eval_entry)

            if mean_eer < best_eval["mean_eer"]:        # save on MIN EER
                best_eval = {"epoch": epoch, "mean_rank1": mean_r1,
                             "mean_eer": mean_eer}
                ckpt_path = os.path.join(cfg.output_dir, ckpt_name(cfg))
                torch.save({
                    "epoch": epoch,
                    "method": "compnet",
                    "backbone": model.backbone.state_dict(),
                    "classifier": model.classifier.state_dict(),
                    "arch": {"embed_dim": cfg.embed_dim,
                             "compnet_channels": cfg.compnet_channels,
                             "img_size": cfg.img_size},
                    "train_id_map": train_id_map,        # ← identity str -> class idx
                    "n_train_ids": n_train_ids,          # ← convenient, redundant but explicit
                    "mean_rank1": mean_r1, "mean_eer": mean_eer,
                }, ckpt_path)
                print(f"    ★ New best EER={mean_eer:.2f}% "
                      f"(R1={mean_r1:.2f}%) → saved")

            print(f"    Summary: Mean R1={mean_r1:.2f}% | "
                  f"Mean EER={mean_eer:.2f}%\n")

    _print_history_compnet(eval_history, eval_dict)
    _print_footer(cfg, best_eval)

    save_path = os.path.join(cfg.output_dir,
                             f"compnet_{cfg.mode}_seed{cfg.seed}.json")
    with open(save_path, "w") as f:
        json.dump({
            "mode": cfg.mode, "method": "compnet",
            "config": {
                "embed_dim": cfg.embed_dim,
                "compnet_channels": cfg.compnet_channels,
                "epochs": cfg.epochs,
                "train_spectrums": cfg.train_spectrums,
                "aug_multiplier": cfg.aug_multiplier,
            },
            "best": best_eval, "history": eval_history,
        }, f, indent=2)
    print(f"\n  Saved: {save_path}")


# ══════════════════════════════════════════════════════════════
#  History / footer printers
# ══════════════════════════════════════════════════════════════

def _print_history_jepa(eval_history, eval_dict):
    eval_names = list(eval_dict.keys())
    print(f"\n  {'Epoch':>6} {'Loss':>8} {'Sim':>6}", end="")
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


def _print_history_compnet(eval_history, eval_dict):
    eval_names = list(eval_dict.keys())
    print(f"\n  {'Epoch':>6} {'CE':>8} {'Acc%':>6}", end="")
    for name in eval_names:
        print(f" │ {name[:12]:>12} R1   EER", end="")
    print()
    print(f"  {'─'*8}{'─'*8}{'─'*6}", end="")
    for _ in eval_names:
        print(f"─┼─{'─'*24}", end="")
    print()
    for entry in eval_history:
        print(f"  {entry['epoch']:>6} {entry['ce']:>8.4f} "
              f"{entry['train_acc']:>6.2f}", end="")
        for name in eval_names:
            if name in entry:
                r = entry[name]
                print(f" │ {r['rank1']:>6.2f} {r['eer']:>6.2f}", end="")
            else:
                print(f" │ {'---':>6} {'---':>6}", end="")
        print()


def _print_footer(cfg, best_eval):
    print(f"\n{'='*80}")
    print(f"  TRAINING COMPLETE  ({cfg.method})")
    print(f"  Best epoch: {best_eval['epoch']} "
          f"(R1={best_eval['mean_rank1']:.2f}%, "
          f"EER={best_eval.get('mean_eer', float('nan')):.2f}%)")
    print(f"{'='*80}")


# ══════════════════════════════════════════════════════════════
#  Dispatcher
# ══════════════════════════════════════════════════════════════

def main():
    cfg = get_cfg()
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"  SOURCE PRETRAINING  —  method: {cfg.method.upper()}")
    print(f"  Mode: {cfg.mode}   embed_dim={cfg.embed_dim}   "
          f"epochs={cfg.epochs}   aug={cfg.aug_multiplier}×")
    print(f"{'='*80}\n")

    train_loader, eval_dict, id_map, n_train_ids, train_id_map = build_datasets(cfg)

    if cfg.method == "jepa":
        train_jepa(cfg, train_loader, eval_dict, id_map, n_train_ids)
    elif cfg.method == "compnet":
        train_compnet(cfg, train_loader, eval_dict, id_map, n_train_ids, train_id_map)
    else:
        raise SystemExit(f"unknown method: {cfg.method}")


if __name__ == "__main__":
    main()
