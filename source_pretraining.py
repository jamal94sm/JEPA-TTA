"""
source_pretraining.py — Source-model pretraining with a method toggle.

  --method jepa     : transformer + self-supervised (I-JEPA)   [original path]
  --method compnet  : CompNet CNN + supervised cross-entropy on training IDs
  --method vit_sup  : plain ViT + supervised cross-entropy on training IDs

Both paths share the same dataset pipeline and the same evaluation
(run_full_eval on the eval_dict), and both save a checkpoint whose backbone
produces [B, embed_dim] features — so all downstream subspace tooling works
unchanged. Point --output_dir somewhere method-specific so checkpoints do not
collide (e.g. ./output_jepa vs ./output_compnet).

JEPA add-ons (both default OFF, so --use_corruption 0 --use_gabor 0 reproduces
the original baseline exactly):
  --use_corruption 1 : domain-shift corruption applied to the CONTEXT view only
  --use_gabor 1      : Gabor line-structure auxiliary loss (design A1)


python source_pretraining.py --method compnet --data_dir /home/pai-ng/Jamal/CASIA-MS-ROI --mode cross_domain_openset --train_spectrums WHT --output_dir ./output_compnet


nohup python source_pretraining.py --method vit_sup \
  --data_dir /home/pai-ng/Jamal/CASIA-MS-ROI \
  --mode cross_domain_openset --train_spectrums WHT \
  --patch_size 14 --vit_depth 6 --vit_heads 8 \
  --output_dir ./output_vitsup > SupViT.log 2>&1 &

nohup python source_pretraining.py --method jepa \
  --data_dir /home/pai-ng/Jamal/CASIA-MS-ROI \
  --mode cross_domain_openset --train_spectrums WHT \
  --use_gabor 1 --gabor_weight 0.3 \
  --output_dir ./output_jepa_gabor > JepaGabor.log 2>&1 &

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
                    repeat_interleave_batch, update_ema, CompNet, PlainViT,
                    FeatModule, GaborHead)

from evaluate import run_full_eval
from corruption import corrupt_images
from gabor import GaborBank, patch_energy_descriptor, sanity_report

CASIA_MEAN = [0.5, 0.5, 0.5]                    # matches dataset.py's Normalize()
CASIA_STD  = [0.5, 0.5, 0.5]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def ckpt_name(cfg):
    """ckpt_{dataset}_{method}_{source_domain}.pth"""
    dataset = os.path.basename(os.path.normpath(cfg.data_dir)).lower()

    if "casia" in dataset:
        dataset = "casiams"
    elif "xjtu" in dataset:
        dataset = "xjtu"
    elif "xpalm" in dataset:
        dataset = "xpalm"

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
#  JEPA (self-supervised)
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

    # ─── Gabor structural auxiliary head (design A1) ──────────
    use_gabor = bool(getattr(cfg, "use_gabor", 0)) and \
                float(getattr(cfg, "gabor_weight", 0.0)) > 0.0
    gabor_bank = gabor_head = None
    if use_gabor:
        gabor_bank = GaborBank(n_orient=cfg.gabor_orient).to(cfg.device)
        gabor_head = GaborHead(cfg.embed_dim, gabor_bank.K).to(cfg.device)
        n_gab = sum(p.numel() for p in gabor_head.parameters())
        print(f"  Gabor bank: K={gabor_bank.K} channels "
              f"({cfg.gabor_orient} orient x {gabor_bank.n_scales} scales), "
              f"trainable={gabor_bank.weight.requires_grad}")
        print(f"  Gabor head: {n_gab/1e6:.3f}M params   "
              f"weight={cfg.gabor_weight}")

    print(f"  Corruption: {'ON' if getattr(cfg, 'use_corruption', 0) else 'OFF'}"
          f"   Gabor aux: {'ON' if use_gabor else 'OFF'}")

    train_params = list(context_encoder.parameters()) + list(predictor.parameters())
    if use_gabor:
        train_params += list(gabor_head.parameters())
    opt = torch.optim.AdamW(train_params, lr=cfg.learning_rate,
                            weight_decay=cfg.weight_decay)

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
        if use_gabor:
            gabor_head.train()

        ep_loss = 0.0          # raw JEPA term only — comparable across runs
        ep_var = 0.0
        ep_gab = 0.0           # Gabor auxiliary loss
        ep_gcos = 0.0          # cosine(pred, target) for the Gabor branch
        ep_gpvar = 0.0         # variance of Gabor head outputs (collapse check)
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

            if cfg.use_corruption:
                images_ctx = corrupt_images(images, cfg, CASIA_MEAN, CASIA_STD)
            else:
                images_ctx = images
            ctx_embeds = context_encoder(images_ctx, ctx_masks)

            with torch.no_grad():
                z_flat = ctx_embeds.reshape(-1, ctx_embeds.size(-1))
                ep_var += z_flat.var(dim=0).mean().item()

            with torch.no_grad():
                tgt_full = target_encoder(images)
                tgt_embeds = apply_masks(tgt_full, tgt_masks)
                tgt_embeds = repeat_interleave_batch(
                    tgt_embeds, B, repeat=len(ctx_masks))

            pred_embeds = predictor(ctx_embeds, ctx_masks, tgt_masks)

            loss_jepa = F.smooth_l1_loss(pred_embeds, tgt_embeds)
            loss = loss_jepa

            # ─── Gabor auxiliary (A1): visible patches, clean-image target ───
            if use_gabor:
                with torch.no_grad():
                    desc = patch_energy_descriptor(
                        gabor_bank(images), cfg.num_patches)     # CLEAN image
                    g_tgt = apply_masks(desc, ctx_masks)         # CONTEXT masks

                if epoch == 1 and n_bat == 0:
                    assert g_tgt.shape[:2] == ctx_embeds.shape[:2], (
                        f"Gabor/context mask misalignment: "
                        f"g_tgt {tuple(g_tgt.shape)} vs "
                        f"ctx_embeds {tuple(ctx_embeds.shape)}")
                    rep = sanity_report(gabor_bank, images, cfg.num_patches)
                    print("\n  ── Gabor sanity check (epoch 1, batch 0) ──")
                    for k, v in rep.items():
                        print(f"      {k}: {v}")
                    print(f"      ctx_embeds: {tuple(ctx_embeds.shape)}   "
                          f"g_tgt: {tuple(g_tgt.shape)}")
                    if rep["desc_pair_cos"] > 0.95:
                        print("      !! WARNING: descriptors nearly identical "
                              "across patches — target carries little signal.")
                    if not (0.99 < rep["desc_norm_mean"] < 1.01):
                        print("      !! WARNING: descriptor norms != 1.0 "
                              "— normalization broken.")
                    if rep["resp_absmean"] < 1e-6:
                        print("      !! WARNING: Gabor responses ~0 "
                              "— filter construction bug.")
                    print()

                g_pred = F.normalize(gabor_head(ctx_embeds), dim=-1)
                g_cos = F.cosine_similarity(g_pred, g_tgt, dim=-1)
                l_gab = (1.0 - g_cos).mean()
                loss = loss + cfg.gabor_weight * l_gab

                ep_gab += l_gab.item()
                ep_gcos += g_cos.mean().item()
                ep_gpvar += g_pred.reshape(
                    -1, g_pred.size(-1)).var(dim=0).mean().item()

            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()

            momentum = get_momentum(global_step)
            update_ema(context_encoder, target_encoder, momentum)

            global_step += 1
            ep_loss += loss_jepa.item()
            n_bat += 1

        ep_loss /= max(n_bat, 1)
        ep_var /= max(n_bat, 1)
        ep_gab /= max(n_bat, 1)
        ep_gcos /= max(n_bat, 1)
        ep_gpvar /= max(n_bat, 1)
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
            if use_gabor:
                print(f"           gabor: l_gab={ep_gab:.4f}  "
                      f"cos={ep_gcos:.3f}  pred_var={ep_gpvar:.5f}")

        # Gradient-norm diagnostic: grads from the final batch are still live
        # here (zero_grad happens at the start of the next iteration).
        if use_gabor and (epoch % cfg.gabor_log_every == 0 or epoch == 1):
            with torch.no_grad():
                head_gnorm = sum(
                    p.grad.norm().item() ** 2
                    for p in gabor_head.parameters()
                    if p.grad is not None) ** 0.5
                enc_gnorm = sum(
                    p.grad.norm().item() ** 2
                    for p in context_encoder.parameters()
                    if p.grad is not None) ** 0.5
            ratio = head_gnorm / max(enc_gnorm, 1e-12)
            print(f"           grad norms: gabor_head={head_gnorm:.4f}  "
                  f"context_encoder={enc_gnorm:.4f}  ratio={ratio:.2f}")

        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            print(f"\n  ── Eval at epoch {epoch} ──")
            context_encoder.eval()
            eval_results = run_full_eval(
                feature_extractor, eval_dict, cfg,
                tag=f"[ep{epoch}] ")

            eval_entry = {"epoch": epoch, "loss": ep_loss, "sim": sim}
            if use_gabor:
                eval_entry["l_gab"] = ep_gab
                eval_entry["gabor_cos"] = ep_gcos
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
                ckpt = {
                    "epoch": epoch,
                    "method": "jepa",
                    "context_encoder": context_encoder.state_dict(),
                    "target_encoder": target_encoder.state_dict(),
                    "predictor": predictor.state_dict(),
                    "arch": {"embed_dim": cfg.embed_dim,
                             "num_patches": cfg.num_patches,
                             "img_size": cfg.img_size},
                    "mean_rank1": mean_r1,
                }
                if use_gabor:
                    ckpt["gabor_head"] = gabor_head.state_dict()
                    ckpt["gabor_cfg"] = {"orient": cfg.gabor_orient,
                                         "K": gabor_bank.K,
                                         "weight": cfg.gabor_weight}
                torch.save(ckpt, ckpt_path)
                print(f"    ★ New best R1={mean_r1:.2f}% "
                      f"EER={mean_eer:.2f}% → saved")

            print(f"    Summary: Mean R1={mean_r1:.2f}% | "
                  f"Mean EER={mean_eer:.2f}%\n")

    _print_history_jepa(eval_history, eval_dict, use_gabor)
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
                "use_corruption": int(getattr(cfg, "use_corruption", 0)),
                "use_gabor": int(use_gabor),
                "gabor_weight": float(getattr(cfg, "gabor_weight", 0.0)),
                "gabor_orient": int(getattr(cfg, "gabor_orient", 0)),
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
                    "train_id_map": train_id_map,        # identity str -> class idx
                    "n_train_ids": n_train_ids,
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
#  Supervised ViT (JEPA encoder + CE head on training IDs)
# ══════════════════════════════════════════════════════════════

def train_vit_sup(cfg, train_loader, eval_dict, id_map, n_train_ids, train_id_map):
    print(f"\n  Building Supervised ViT (JEPA encoder + CE head)...")
    model = PlainViT(img_size=cfg.img_size, patch_size=cfg.patch_size,
                     embed_dim=cfg.embed_dim, depth=cfg.vit_depth,
                     n_heads=cfg.vit_heads, n_classes=n_train_ids).to(cfg.device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"  SupervisedViT: {n_par/1e6:.2f}M params   n_classes={n_train_ids}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                            weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * len(train_loader)
    scheduler = make_scheduler(opt, cfg, total_steps)
    ce = torch.nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    # run_full_eval needs an object whose forward(x) -> [B, embed_dim];
    # for the ViT that is model.backbone = FeatureExtractor(encoder).
    feature_extractor = FeatModule(model)

    print(f"\n{'─'*70}")
    print(f"  Training Supervised ViT ({total_steps} steps, CE on IDs)")
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
                    "method": "vit_sup",
                    "full_state": model.state_dict(),
                    "classifier": model.classifier.state_dict(),
                    "arch": {"embed_dim": cfg.embed_dim,
                             "patch_size": cfg.patch_size,
                             "vit_depth": cfg.vit_depth,
                             "vit_heads": cfg.vit_heads,
                             "img_size": cfg.img_size},
                    "train_id_map": train_id_map,
                    "n_train_ids": n_train_ids,
                    "mean_rank1": mean_r1, "mean_eer": mean_eer,
                }, ckpt_path)
                print(f"    ★ New best EER={mean_eer:.2f}% "
                      f"(R1={mean_r1:.2f}%) → saved")

            print(f"    Summary: Mean R1={mean_r1:.2f}% | "
                  f"Mean EER={mean_eer:.2f}%\n")

    _print_history_compnet(eval_history, eval_dict)
    _print_footer(cfg, best_eval)

    save_path = os.path.join(cfg.output_dir,
                             f"vitsup_{cfg.mode}_seed{cfg.seed}.json")
    with open(save_path, "w") as f:
        json.dump({
            "mode": cfg.mode, "method": "vit_sup",
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
#  History / footer printers
# ══════════════════════════════════════════════════════════════

def _print_history_jepa(eval_history, eval_dict, use_gabor=False):
    eval_names = list(eval_dict.keys())
    if use_gabor:
        print(f"\n  {'Epoch':>6} {'Loss':>8} {'Sim':>6} {'l_gab':>7} {'gcos':>6}", end="")
    else:
        print(f"\n  {'Epoch':>6} {'Loss':>8} {'Sim':>6}", end="")
    for name in eval_names:
        print(f" │ {name[:12]:>12} R1   EER", end="")
    print()
    print(f"  {'─'*8}{'─'*8}{'─'*6}", end="")
    if use_gabor:
        print(f"{'─'*7}{'─'*6}", end="")
    for _ in eval_names:
        print(f"─┼─{'─'*24}", end="")
    print()
    for entry in eval_history:
        print(f"  {entry['epoch']:>6} {entry['loss']:>8.4f} "
              f"{entry['sim']:>6.3f}", end="")
        if use_gabor:
            print(f" {entry.get('l_gab', float('nan')):>7.4f} "
                  f"{entry.get('gabor_cos', float('nan')):>6.3f}", end="")
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
    elif cfg.method == "vit_sup":
        train_vit_sup(cfg, train_loader, eval_dict, id_map, n_train_ids, train_id_map)
    else:
        raise SystemExit(f"unknown method: {cfg.method}")


if __name__ == "__main__":
    main()
