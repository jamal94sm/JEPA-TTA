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

JEPA add-ons (all default OFF, so the plain flags reproduce the original
baseline exactly):
  --use_corruption 1  : domain-shift corruption applied to the CONTEXT view only
  --struct_mode a1/a2/both : Gabor line-structure auxiliary loss(es)
  --use_supervision 1 : supervised identity term (SupCon / ArcFace / CE) on
                        pooled context embeddings, using source-domain labels


python source_pretraining.py --method compnet --data_dir /home/pai-ng/Jamal/CASIA-MS-ROI --mode cross_domain_openset --train_spectrums WHT --output_dir ./output_compnet


nohup python source_pretraining.py --method vit_sup \
  --data_dir /home/pai-ng/Jamal/CASIA-MS-ROI \
  --mode cross_domain_openset --train_spectrums WHT \
  --patch_size 14 --vit_depth 6 --vit_heads 8 \
  --output_dir ./output_vitsup > SupViT.log 2>&1 &

nohup python source_pretraining.py --method jepa \
  --data_dir /home/pai-ng/Jamal/CASIA-MS-ROI \
  --mode cross_domain_openset --train_spectrums WHT \
  --struct_mode a1 --w_a1 0.3 \
  --output_dir ./output_jepa_gabor > JepaGabor.log 2>&1 &

nohup python source_pretraining.py --method jepa \
  --data_dir /home/pai-ng/Jamal/XJTU-UP \
  --mode cross_domain_openset --train_spectrums iPhone_Nature \
  --use_corruption 1 --gabor_gray 0 --struct_mode a2 --struct_loss infonce --w_a2 0.3 \
  --use_supervision 1 --sup_loss supcon --w_sup 0.1 --batch_size 256 \
  --output_dir ./output_jepa_a2_supcon > JepaA2Supcon.log 2>&1 &

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
                    FeatModule, StructureHead, UncertaintyWeighting)

from evaluate import run_full_eval
from corruption import corrupt_images
from gabor import GaborBank, patch_energy_descriptor, sanity_report
from struct_loss import structure_loss, grad_conflict_cosine
from sup_loss import supcon_loss, build_sup_head

CASIA_MEAN = [0.5, 0.5, 0.5]                    # matches dataset.py's Normalize()
CASIA_STD  = [0.5, 0.5, 0.5]


def gabor_weight_at(epoch, cfg):
    """Auxiliary-loss weight for this epoch (1-indexed)."""
    w0 = float(cfg.gabor_weight)
    wf = float(getattr(cfg, "gabor_weight_final", 0.0))
    s = getattr(cfg, "gabor_schedule", "constant")

    if s == "constant":
        return w0
    if s == "cosine":
        p = (epoch - 1) / max(1, cfg.epochs - 1)
        return wf + (w0 - wf) * 0.5 * (1 + math.cos(math.pi * p))

    T = max(1, int(cfg.gabor_schedule_end * cfg.epochs))
    t = min(1.0, (epoch - 1) / T)
    if s == "decay":
        return w0 + (wf - w0) * t
    if s == "ramp":
        return w0 * t
    raise ValueError(f"unknown gabor_schedule: {s}")


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
#  JEPA (self-supervised, with optional structural + supervised terms)
# ══════════════════════════════════════════════════════════════

def train_jepa(cfg, train_loader, eval_dict, id_map, n_classes):
    img_size = (cfg.img_size, cfg.img_size)

    print(f"\n  Building JEPA models...")
    context_encoder = ContextEncoder(
        img_size, cfg.num_patches, cfg.embed_dim).to(cfg.device)
    target_encoder = TargetEncoder(
        img_size, cfg.num_patches, cfg.embed_dim).to(cfg.device)
    predictor = Predictor(
        cfg.num_patches, cfg.embed_dim,
        norm_struct_out=bool(cfg.norm_struct_out)).to(cfg.device)

    for pc, pt in zip(context_encoder.parameters(),
                      target_encoder.parameters()):
        pt.data.copy_(pc.data)
    for p in target_encoder.parameters():
        p.requires_grad = False

    n_ctx = sum(p.numel() for p in context_encoder.parameters())
    n_pred = sum(p.numel() for p in predictor.parameters())
    print(f"  Context encoder: {n_ctx/1e6:.2f}M params")
    print(f"  Predictor: {n_pred/1e6:.2f}M params")

    # ─── Structural auxiliary tasks (A1 = visible, A2 = hidden) ───
    use_a1 = bool(getattr(cfg, "use_a1", False))
    use_a2 = bool(getattr(cfg, "use_a2", False))
    use_struct = use_a1 or use_a2
    gabor_bank = struct_head = struct_head_a2 = task_weighter = None

    if use_struct:
        gabor_bank = GaborBank(
            n_orient=cfg.gabor_orient,
            scales=cfg.gabor_scales,
            per_channel=not bool(getattr(cfg, "gabor_gray", 1)),
        ).to(cfg.device)
        struct_head = StructureHead(
            cfg.embed_dim, gabor_bank.K,
            hidden=cfg.struct_head_hidden).to(cfg.device)

        # Separate head only means anything when BOTH tasks are active.
        if cfg.struct_head_mode == "separate" and use_a1 and use_a2:
            struct_head_a2 = StructureHead(
                cfg.embed_dim, gabor_bank.K,
                hidden=cfg.struct_head_hidden).to(cfg.device)
        else:
            struct_head_a2 = struct_head          # alias -> shared behaviour

        n_sh = sum(p.numel() for p in struct_head.parameters())
        if struct_head_a2 is not struct_head:
            n_sh += sum(p.numel() for p in struct_head_a2.parameters())

    # ─── Supervised identity term (uses source-domain labels) ─────
    use_sup = bool(getattr(cfg, "use_supervision", 0)) and cfg.w_sup > 0
    sup_head = None
    if use_sup:
        sup_head = build_sup_head(cfg.sup_loss, cfg.embed_dim, n_classes, cfg)
        if sup_head is not None:
            sup_head = sup_head.to(cfg.device)

    n_tasks = 1 + int(use_a1) + int(use_a2) + int(use_sup)
    if use_struct or use_sup:
        if cfg.task_weighting == "uncertainty":
            task_weighter = UncertaintyWeighting(n_tasks).to(cfg.device)

    if use_struct:
        mode = "per-channel RGB" if gabor_bank.per_channel else "grayscale"
        head_mode = ("separate" if struct_head_a2 is not struct_head
                     else "shared")
        print(f"  Gabor bank: K={gabor_bank.K} "
              f"({cfg.gabor_orient} orient x {gabor_bank.n_scales} scales, {mode})")
        print(f"  Structure head: {n_sh/1e6:.3f}M params  "
              f"(hidden={cfg.struct_head_hidden} -> {gabor_bank.K}, {head_mode})")
        print(f"  Struct out norm: {'ON' if cfg.norm_struct_out else 'OFF'}")
        print(f"  Struct mode: {cfg.struct_mode}   "
              f"loss_a1={cfg.loss_a1 if use_a1 else '—'}  "
              f"loss_a2={cfg.loss_a2 if use_a2 else '—'}   "
              f"weighting={cfg.task_weighting}   "
              f"w_a1={cfg.w_a1 if use_a1 else '—'}  w_a2={cfg.w_a2 if use_a2 else '—'}")

    if use_sup:
        n_sp = sum(p.numel() for p in sup_head.parameters()) if sup_head else 0
        print(f"  Supervision: {cfg.sup_loss}  w_sup={cfg.w_sup}  "
              f"n_classes={n_classes}  head={n_sp/1e6:.3f}M params")

    print(f"  Corruption: {'ON' if getattr(cfg, 'use_corruption', 0) else 'OFF'}"
          f"   Structural: {'ON' if use_struct else 'OFF'}"
          f"   Supervision: {'ON' if use_sup else 'OFF'}")

    train_params = list(context_encoder.parameters()) + list(predictor.parameters())
    if use_struct:
        train_params += list(struct_head.parameters())
        if struct_head_a2 is not struct_head:      # avoid double-registering
            train_params += list(struct_head_a2.parameters())
    if sup_head is not None:
        train_params += list(sup_head.parameters())
    if task_weighter is not None:
        train_params += list(task_weighter.parameters())
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
        if use_struct:
            struct_head.train()
            if struct_head_a2 is not struct_head:
                struct_head_a2.train()
        if sup_head is not None:
            sup_head.train()

        ep_loss = 0.0          # raw JEPA term only — comparable across runs
        ep_var = 0.0
        ep_a1 = ep_a1_cos = ep_a1_top1 = 0.0
        ep_a2 = ep_a2_cos = ep_a2_top1 = 0.0
        n_a1 = n_a2 = 0
        ep_sup = ep_sup_aux = 0.0
        n_sup = 0
        ep_conflict = float("nan")
        n_bat = 0
        t0 = time.time()

        for images, labels in train_loader:
            images = images.to(cfg.device)
            labels = labels.to(cfg.device)
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

            # A2 requests the extra structure query from the shared predictor.
            if use_a2:
                pred_embeds, struct_hidden = predictor(
                    ctx_embeds, ctx_masks, tgt_masks, predict_structure=True)
            else:
                pred_embeds = predictor(ctx_embeds, ctx_masks, tgt_masks)

            loss_jepa = F.smooth_l1_loss(pred_embeds, tgt_embeds)

            l_a1 = l_a2 = None
            if use_struct:
                with torch.no_grad():
                    desc = patch_energy_descriptor(
                        gabor_bank(images), cfg.num_patches)   # CLEAN image

                if use_a1:
                    t_a1 = apply_masks(desc, ctx_masks)        # visible patches
                    if epoch == 1 and n_bat == 0:
                        assert t_a1.shape[:2] == ctx_embeds.shape[:2], (
                            f"A1 misalignment: t_a1 {tuple(t_a1.shape)} vs "
                            f"ctx_embeds {tuple(ctx_embeds.shape)}")
                    l_a1, s_a1 = structure_loss(
                        struct_head(ctx_embeds), t_a1,
                        kind=cfg.loss_a1,
                        temperature=cfg.infonce_temp,
                        max_n=cfg.infonce_max_n)
                    ep_a1 += l_a1.item()
                    ep_a1_cos += s_a1["cos"]
                    ep_a1_top1 += s_a1["top1"]
                    n_a1 += 1

                if use_a2:
                    t_a2 = repeat_interleave_batch(
                        apply_masks(desc, tgt_masks), B,
                        repeat=len(ctx_masks))                 # hidden patches
                    if epoch == 1 and n_bat == 0:
                        assert t_a2.shape[:2] == struct_hidden.shape[:2], (
                            f"A2 misalignment: t_a2 {tuple(t_a2.shape)} vs "
                            f"struct_hidden {tuple(struct_hidden.shape)}")
                    l_a2, s_a2 = structure_loss(
                        struct_head_a2(struct_hidden), t_a2,
                        kind=cfg.loss_a2,
                        temperature=cfg.infonce_temp,
                        max_n=cfg.infonce_max_n)
                    ep_a2 += l_a2.item()
                    ep_a2_cos += s_a2["cos"]
                    ep_a2_top1 += s_a2["top1"]
                    n_a2 += 1

                if epoch == 1 and n_bat == 0:
                    rep = sanity_report(gabor_bank, images, cfg.num_patches)
                    print("\n  ── Structural sanity check (epoch 1, batch 0) ──")
                    for k, v in rep.items():
                        print(f"      {k}: {v}")
                    if rep["desc_pair_cos"] > 0.95:
                        print("      !! WARNING: descriptors nearly identical "
                              "across patches — target carries little signal.")
                    if not (0.99 < rep["desc_norm_mean"] < 1.01):
                        print("      !! WARNING: descriptor norms != 1.0.")
                    if rep["resp_absmean"] < 1e-6:
                        print("      !! WARNING: Gabor responses ~0.")
                    print()

            # ─── Supervised identity term ───
            # Pooled the same way FeatureExtractor pools at eval time, but
            # over the corrupted/visible context (a harder, related target).
            l_sup = None
            if use_sup:
                z_sup = ctx_embeds.mean(dim=1)             # (B, embed_dim)
                if cfg.sup_loss == "supcon":
                    l_sup, s_sup = supcon_loss(
                        z_sup, labels, temperature=cfg.supcon_temp)
                else:
                    l_sup, s_sup = sup_head(z_sup, labels)
                ep_sup += l_sup.item()
                ep_sup_aux += s_sup.get("acc", s_sup.get("pos_per_anchor", 0.0))
                n_sup += 1

                if epoch == 1 and n_bat == 0 and cfg.sup_loss == "supcon":
                    print(f"\n  ── SupCon sanity check (epoch 1, batch 0) ──")
                    print(f"      pos_per_anchor: {s_sup['pos_per_anchor']:.3f}")
                    print(f"      frac_anchors_with_pos: "
                          f"{s_sup['frac_anchors_with_pos']:.3f}")
                    if s_sup["pos_per_anchor"] < 1.0:
                        print("      !! WARNING: fewer than 1 positive/anchor "
                              "on average — increase --batch_size or use a "
                              "sampler that guarantees same-identity pairs.")
                    print()

            # ─── Combine task losses ───
            if task_weighter is not None:
                terms = [loss_jepa]
                if l_a1 is not None:
                    terms.append(l_a1)
                if l_a2 is not None:
                    terms.append(l_a2)
                if l_sup is not None:
                    terms.append(l_sup)
                loss = task_weighter(terms)
            else:
                loss = loss_jepa
                if l_a1 is not None:
                    loss = loss + cfg.w_a1 * l_a1
                if l_a2 is not None:
                    loss = loss + cfg.w_a2 * l_a2
                if l_sup is not None:
                    loss = loss + cfg.w_sup * l_sup

            # ─── Gradient-conflict diagnostic on shared params ───
            # >0 complementary, ~0 orthogonal, <0 conflicting.
            if ((use_struct or use_sup) and cfg.log_conflict and n_bat == 0
                    and (epoch % cfg.gabor_log_every == 0 or epoch == 1)):
                l_aux_tot = 0.0
                if l_a1 is not None:
                    l_aux_tot = l_aux_tot + l_a1
                if l_a2 is not None:
                    l_aux_tot = l_aux_tot + l_a2
                if l_sup is not None:
                    l_aux_tot = l_aux_tot + l_sup
                ep_conflict = grad_conflict_cosine(
                    loss_jepa, l_aux_tot,
                    list(context_encoder.norm.parameters()))

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
        ep_a1 /= max(n_a1, 1)
        ep_a1_cos /= max(n_a1, 1)
        ep_a1_top1 /= max(n_a1, 1)
        ep_a2 /= max(n_a2, 1)
        ep_a2_cos /= max(n_a2, 1)
        ep_a2_top1 /= max(n_a2, 1)
        ep_sup /= max(n_sup, 1)
        ep_sup_aux /= max(n_sup, 1)
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
            if use_a1:
                print(f"           A1 (visible): loss={ep_a1:.4f}  "
                      f"cos={ep_a1_cos:.3f}  top1={ep_a1_top1:.3f}")
            if use_a2:
                print(f"           A2 (hidden):  loss={ep_a2:.4f}  "
                      f"cos={ep_a2_cos:.3f}  top1={ep_a2_top1:.3f}")
            if use_sup:
                aux = "acc" if cfg.sup_loss in ("arcface", "ce") else "pos/anchor"
                print(f"           SUP ({cfg.sup_loss}): loss={ep_sup:.4f}  "
                      f"{aux}={ep_sup_aux:.3f}")
            if (use_struct or use_sup) and cfg.log_conflict:
                msg = f"           conflict_cos={ep_conflict:+.4f}"
                if task_weighter is not None:
                    ws = "  ".join(f"{w:.3f}" for w in task_weighter.weights())
                    msg += f"   learned_w=[{ws}]"
                print(msg)

        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            print(f"\n  ── Eval at epoch {epoch} ──")
            context_encoder.eval()
            eval_results = run_full_eval(
                feature_extractor, eval_dict, cfg,
                tag=f"[ep{epoch}] ")

            eval_entry = {"epoch": epoch, "loss": ep_loss, "sim": sim,
                          "var": ep_var}
            if use_a1:
                eval_entry["l_a1"] = ep_a1
                eval_entry["a1_cos"] = ep_a1_cos
                eval_entry["a1_top1"] = ep_a1_top1
            if use_a2:
                eval_entry["l_a2"] = ep_a2
                eval_entry["a2_cos"] = ep_a2_cos
                eval_entry["a2_top1"] = ep_a2_top1
            if use_sup:
                eval_entry["l_sup"] = ep_sup
                eval_entry["sup_aux"] = ep_sup_aux
            if use_struct or use_sup:
                eval_entry["conflict_cos"] = ep_conflict
                if task_weighter is not None:
                    eval_entry["learned_w"] = task_weighter.weights()

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
                if use_struct:
                    ckpt["struct_head"] = struct_head.state_dict()
                    if struct_head_a2 is not struct_head:
                        ckpt["struct_head_a2"] = struct_head_a2.state_dict()
                    ckpt["struct_cfg"] = {
                        "struct_mode": cfg.struct_mode,
                        "struct_head_mode": cfg.struct_head_mode,
                        "norm_struct_out": int(cfg.norm_struct_out),
                        "loss_a1": cfg.loss_a1,
                        "loss_a2": cfg.loss_a2,
                        "w_a1": cfg.w_a1, "w_a2": cfg.w_a2,
                        "task_weighting": cfg.task_weighting,
                        "gabor_orient": cfg.gabor_orient,
                        "gabor_K": gabor_bank.K,
                        "per_channel": gabor_bank.per_channel,
                    }
                    if task_weighter is not None:
                        ckpt["task_weighter"] = task_weighter.state_dict()
                if sup_head is not None:
                    ckpt["sup_head"] = sup_head.state_dict()
                if use_sup:
                    ckpt["sup_cfg"] = {
                        "use_supervision": int(use_sup),
                        "sup_loss": cfg.sup_loss,
                        "w_sup": cfg.w_sup,
                    }
                torch.save(ckpt, ckpt_path)
                print(f"    ★ New best R1={mean_r1:.2f}% "
                      f"EER={mean_eer:.2f}% → saved")

            print(f"    Summary: Mean R1={mean_r1:.2f}% | "
                  f"Mean EER={mean_eer:.2f}%\n")

    _print_history_jepa(eval_history, eval_dict, use_a1, use_a2, use_sup)
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
                "struct_mode": cfg.struct_mode,
                "loss_a1": cfg.loss_a1,
                "loss_a2": cfg.loss_a2,
                "struct_head_mode": cfg.struct_head_mode,
                "norm_struct_out": int(cfg.norm_struct_out),
                "w_a1": float(cfg.w_a1),
                "w_a2": float(cfg.w_a2),
                "struct_head_hidden": int(cfg.struct_head_hidden),
                "task_weighting": cfg.task_weighting,
                "infonce_temp": float(cfg.infonce_temp),
                "gabor_orient": int(cfg.gabor_orient),
                "gabor_gray": int(getattr(cfg, "gabor_gray", 1)),
                "use_supervision": int(use_sup),
                "sup_loss": cfg.sup_loss,
                "w_sup": float(cfg.w_sup),
            },
            "best": best_eval, "history": eval_history,
        }, f, indent=2)
    print(f"\n  Saved: {save_path}")


def _print_history_jepa(eval_history, eval_dict, use_a1=False, use_a2=False,
                         use_sup=False):
    eval_names = list(eval_dict.keys())

    print(f"\n  {'Epoch':>6} {'Loss':>8} {'Sim':>6}", end="")
    if use_a1:
        print(f" {'l_a1':>7} {'a1cos':>6} {'a1top1':>7}", end="")
    if use_a2:
        print(f" {'l_a2':>7} {'a2cos':>6} {'a2top1':>7}", end="")
    if use_sup:
        print(f" {'l_sup':>7} {'supaux':>7}", end="")
    for name in eval_names:
        print(f" │ {name[:12]:>12} R1   EER", end="")
    print()

    print(f"  {'─'*8}{'─'*8}{'─'*6}", end="")
    if use_a1:
        print(f"{'─'*7}{'─'*6}{'─'*7}", end="")
    if use_a2:
        print(f"{'─'*7}{'─'*6}{'─'*7}", end="")
    if use_sup:
        print(f"{'─'*7}{'─'*7}", end="")
    for _ in eval_names:
        print(f"─┼─{'─'*24}", end="")
    print()

    for entry in eval_history:
        print(f"  {entry['epoch']:>6} {entry['loss']:>8.4f} "
              f"{entry['sim']:>6.3f}", end="")
        if use_a1:
            print(f" {entry.get('l_a1', float('nan')):>7.4f} "
                  f"{entry.get('a1_cos', float('nan')):>6.3f} "
                  f"{entry.get('a1_top1', float('nan')):>7.3f}", end="")
        if use_a2:
            print(f" {entry.get('l_a2', float('nan')):>7.4f} "
                  f"{entry.get('a2_cos', float('nan')):>6.3f} "
                  f"{entry.get('a2_top1', float('nan')):>7.3f}", end="")
        if use_sup:
            print(f" {entry.get('l_sup', float('nan')):>7.4f} "
                  f"{entry.get('sup_aux', float('nan')):>7.3f}", end="")
        for name in eval_names:
            if name in entry:
                r = entry[name]
                print(f" │ {r['rank1']:>6.2f} {r['eer']:>6.2f}", end="")
            else:
                print(f" │ {'---':>6} {'---':>6}", end="")
        print()


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
#  History / footer printers (CompNet / ViT-sup)
# ══════════════════════════════════════════════════════════════

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
