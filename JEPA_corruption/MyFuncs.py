import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import Datasets as datasets_root
import MyModels
import MyUtils
import Utils as root_utils
import periodic_eval
from gabor import GaborBank, patch_energy_descriptor, sanity_report
from struct_loss import structure_loss, grad_conflict_cosine


CORRUPT_BLUE, CORRUPT_RED, CORRUPT_GREEN, CORRUPT_JITTER, CORRUPT_NOISE = 0, 1, 2, 3, 4
N_CORRUPT = 5

_DATASET_NORM = {
    "stl10": ((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "tiny-imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}


def _extract_eval_features(encoder, x):
    B = x.size(0)
    P = encoder.pos_embed.size(1)
    full_mask = [torch.arange(P, device=x.device).unsqueeze(0).expand(B, -1)]
    z = encoder(x, full_mask)
    return z.mean(dim=1)


def MSE_loss(preds, targets):
    return F.mse_loss(preds, targets)


def _norm_stats(args):
    key = str(getattr(args, "dataset", "stl10")).lower().replace("_", "-")
    if "tiny" in key:
        key = "tiny-imagenet"
    elif "stl" in key:
        key = "stl10"
    return _DATASET_NORM.get(key, _DATASET_NORM["stl10"])


def _denorm(x, mean, std):
    mean = x.new_tensor(mean).view(1, -1, 1, 1)
    std = x.new_tensor(std).view(1, -1, 1, 1)
    return (x * std + mean).clamp(0.0, 1.0)


def _renorm(x, mean, std):
    mean = x.new_tensor(mean).view(1, -1, 1, 1)
    std = x.new_tensor(std).view(1, -1, 1, 1)
    return (x - mean) / std

'''
def corrupt_visible_patches(images, context_masks, args):
    """
    For each visible patch, independently sample one of:
    blue, red, green, jitter, noise. Target patches are left clean.
    """
    B, _, H, W = images.shape
    G = int(args.num_patches)
    ph, pw = H // G, W // G
    P = G * G
    mean, std = _norm_stats(args)
    x = _denorm(images, mean, std)

    ctx = context_masks[0]  # (B, N_ctx)
    kinds = torch.randint(0, N_CORRUPT, ctx.shape, device=images.device)
    patch_kind = torch.full((B, P), -1, device=images.device, dtype=torch.long)
    patch_kind.scatter_(1, ctx, kinds)
    tmap = patch_kind.view(B, G, G)

    # (B, 3, G, ph, G, pw)
    x = x.reshape(B, 3, G, ph, G, pw)

    def _mask(kind):
        m = tmap == kind
        return m[:, None, :, None, :, None]

    m_blue = _mask(CORRUPT_BLUE)
    m_red = _mask(CORRUPT_RED)
    m_green = _mask(CORRUPT_GREEN)
    m_jit = _mask(CORRUPT_JITTER)
    m_noise = _mask(CORRUPT_NOISE)

    # Isolate the named channel (zero the other two) in [0, 1].
    x = torch.where(m_blue.expand_as(x), x * x.new_tensor([0.0, 0.0, 1.0]).view(1, 3, 1, 1, 1, 1), x)
    x = torch.where(m_red.expand_as(x), x * x.new_tensor([1.0, 0.0, 0.0]).view(1, 3, 1, 1, 1, 1), x)
    x = torch.where(m_green.expand_as(x), x * x.new_tensor([0.0, 1.0, 0.0]).view(1, 3, 1, 1, 1, 1), x)

    strength = float(getattr(args, "jitter_strength", 0.4))
    b = 1.0 + (torch.rand(B, 1, G, 1, G, 1, device=x.device) * 2.0 - 1.0) * strength
    c = 1.0 + (torch.rand(B, 1, G, 1, G, 1, device=x.device) * 2.0 - 1.0) * strength
    jittered = ((x * b - 0.5) * c + 0.5).clamp(0.0, 1.0)
    x = torch.where(m_jit.expand_as(x), jittered, x)

    sigma = float(getattr(args, "corruption_std", 0.1))
    noisy = (x + sigma * torch.randn_like(x)).clamp(0.0, 1.0)
    x = torch.where(m_noise.expand_as(x), noisy, x)

    x = x.reshape(B, 3, H, W)
    return _renorm(x, mean, std)
'''


import torchvision.transforms.functional as TF

CORRUPTION_NAMES = [
    "color_temp", "gamma", "channel_mix", "desaturate", "blur", "noise", "vignette",
]


def _severity(mask_1d, max_strength, device):
    """Per-sample severity in [0, max_strength], only meaningful where mask_1d is True."""
    return torch.rand(mask_1d.shape[0], 1, 1, 1, device=device) * max_strength


def _apply_color_temp(xs, mask, max_strength):
    # Simulates illuminant / spectral-band shift (e.g. CASIA WHT vs 940nm, or
    # different device white-balance pipelines): push R up / B down or vice versa.
    shift = (torch.rand(xs.size(0), 1, 1, 1, device=xs.device) * 2 - 1) * max_strength
    r = (xs[:, 0:1] * (1 + shift)).clamp(0, 1)
    b = (xs[:, 2:3] * (1 - shift)).clamp(0, 1)
    out = torch.cat([r, xs[:, 1:2], b], dim=1)
    return torch.where(mask.view(-1, 1, 1, 1), out, xs)


def _apply_gamma(xs, mask, max_strength):
    # Simulates sensor exposure / tone-curve differences across devices.
    gamma = 1.0 + (torch.rand(xs.size(0), 1, 1, 1, device=xs.device) * 2 - 1) * max_strength
    gamma = gamma.clamp(min=0.2)
    out = xs.clamp(min=1e-3).pow(gamma)
    return torch.where(mask.view(-1, 1, 1, 1), out, xs)


def _apply_channel_mix(xs, mask, max_strength):
    # Simulates spectral-response crosstalk between bands/sensors (e.g. multispectral
    # NIR bands vs. RGB smartphone sensor having different channel sensitivities).
    B = xs.size(0)
    eye = torch.eye(3, device=xs.device).unsqueeze(0).expand(B, -1, -1)
    off = (torch.rand(B, 3, 3, device=xs.device) * 2 - 1) * max_strength
    M = eye + off
    out = torch.einsum('bij,bjhw->bihw', M, xs).clamp(0, 1)
    return torch.where(mask.view(-1, 1, 1, 1), out, xs)


def _apply_desaturate(xs, mask, max_strength):
    # Simulates spectrum change toward near-IR-like capture (low/no color information),
    # relevant for CASIA's 850/940nm bands vs. WHT / XJTU's RGB captures.
    alpha = torch.rand(xs.size(0), 1, 1, 1, device=xs.device) * max_strength
    gray = xs.mean(dim=1, keepdim=True).expand_as(xs)
    out = xs * (1 - alpha) + gray * alpha
    return torch.where(mask.view(-1, 1, 1, 1), out, xs)


def _apply_blur(xs, mask, max_sigma):
    # Simulates device/optics resolution differences (e.g. fixed NIR camera vs.
    # handheld smartphone focus/motion blur in XJTU-UP).
    out = xs.clone()
    idx = mask.nonzero(as_tuple=True)[0]
    for i in idx.tolist():
        sigma = float(torch.rand(1).item()) * max_sigma
        if sigma < 0.05:
            continue
        k = max(3, int(2 * round(3 * sigma) + 1))
        out[i:i+1] = TF.gaussian_blur(xs[i:i+1], kernel_size=k, sigma=sigma)
    return out


def _apply_noise(xs, mask, max_std):
    # Sensor noise floor differences across devices.
    std = torch.rand(xs.size(0), 1, 1, 1, device=xs.device) * max_std
    out = (xs + std * torch.randn_like(xs)).clamp(0, 1)
    return torch.where(mask.view(-1, 1, 1, 1), out, xs)


def _apply_vignette(xs, mask, max_strength):
    # Simulates acquisition-geometry / lens falloff differences (contact vs.
    # contactless capture, different lens systems).
    H, W = xs.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=xs.device),
        torch.linspace(-1, 1, W, device=xs.device),
        indexing="ij",
    )
    r = torch.sqrt(xx ** 2 + yy ** 2)
    strength = torch.rand(xs.size(0), 1, 1, 1, device=xs.device) * max_strength
    falloff = 1.0 - strength * r.unsqueeze(0).unsqueeze(0).clamp(0, 1)
    out = (xs * falloff).clamp(0, 1)
    return torch.where(mask.view(-1, 1, 1, 1), out, xs)


_CORRUPTION_FN = {
    "color_temp":  lambda xs, m, args: _apply_color_temp(xs, m, args.color_temp_strength),
    "gamma":       lambda xs, m, args: _apply_gamma(xs, m, args.gamma_strength),
    "channel_mix": lambda xs, m, args: _apply_channel_mix(xs, m, args.channel_mix_strength),
    "desaturate":  lambda xs, m, args: _apply_desaturate(xs, m, args.desaturate_strength),
    "blur":        lambda xs, m, args: _apply_blur(xs, m, args.blur_sigma_max),
    "noise":       lambda xs, m, args: _apply_noise(xs, m, args.corruption_std),
    "vignette":    lambda xs, m, args: _apply_vignette(xs, m, args.vignette_strength),
}


def corrupt_visible_patches(images, context_masks, args):
    """
    Domain-shift-style corruption, applied UNIFORMLY across the whole image
    (so every visible patch shares the same corruption) rather than per-patch.

    - Which samples get corrupted at all: Bernoulli(args.corruption_prob) per sample
      -> the number of corrupted samples per batch is itself random, controlled by
         the corruption_prob hyperparameter.
    - Which corruption(s) each corrupted sample gets:
        args.corruption_mode == 'single' -> exactly one type, chosen uniformly.
        args.corruption_mode == 'mixed'  -> each type included independently with
                                             probability args.mix_prob (at least one
                                             type is forced in so 'corrupted' is never a no-op).
    - Severity of each applied corruption is randomized per-sample up to that
      corruption's --*_strength / --corruption_std / --blur_sigma_max hyperparameter.
    """
    B = images.size(0)
    device = images.device
    mean, std = _norm_stats(args)
    x = _denorm(images, mean, std)

    corrupt_mask = torch.rand(B, device=device) < args.corruption_prob
    if not corrupt_mask.any():
        return images

    n_types = len(CORRUPTION_NAMES)

    if args.corruption_mode == "single":
        chosen = torch.randint(0, n_types, (B,), device=device)
        type_masks = {
            name: corrupt_mask & (chosen == i)
            for i, name in enumerate(CORRUPTION_NAMES)
        }
    else:  # mixed
        include = torch.rand(B, n_types, device=device) < args.mix_prob
        # Force at least one type per corrupted sample so it's never a silent no-op.
        none_selected = ~include.any(dim=1)
        if none_selected.any():
            forced = torch.randint(0, n_types, (int(none_selected.sum()),), device=device)
            include[none_selected, forced] = True
        type_masks = {
            name: corrupt_mask & include[:, i]
            for i, name in enumerate(CORRUPTION_NAMES)
        }

    for name in CORRUPTION_NAMES:
        m = type_masks[name]
        if m.any():
            x = _CORRUPTION_FN[name](x, m, args)

    return _renorm(x, mean, std)


###############################################


def Train(
    dataloader,
    context_encoder,
    target_encoder,
    predictor,
    opt,
    lr_scheduler,
    wd_scheduler,
    momentum_schedule,
    checkpoint_state,
    args,
    gabor_bank=None,
    struct_head=None,
    task_weighter=None,
):
    device = args.device
    epoch_losses = []
    global_step = checkpoint_state["global_step"]
    start_epoch = checkpoint_state["start_epoch"]
    run_dir = checkpoint_state["run_dir"]
    best_acc = checkpoint_state.get("best_acc", checkpoint_state.get("best_loss", float("-inf")))
    eval_history = list(checkpoint_state.get("eval_history", []))

    use_a1 = struct_head is not None and bool(getattr(args, "use_a1", False))
    use_a2 = struct_head is not None and bool(getattr(args, "use_a2", False))
    use_struct = use_a1 or use_a2


    # ─── Startup summary — mirrors source_pretraining.py's transparency ───
    n_ctx = sum(p.numel() for p in context_encoder.parameters())
    n_tgt = sum(p.numel() for p in target_encoder.parameters())
    n_pred = sum(p.numel() for p in predictor.parameters())
    print(f"\n{'='*70}")
    print(f"  TRAINING SETUP")
    print(f"{'='*70}")
    print(f"  Context encoder: {n_ctx/1e6:.2f}M params (trainable)")
    print(f"  Target encoder:  {n_tgt/1e6:.2f}M params (EMA, frozen)")
    print(f"  Predictor:       {n_pred/1e6:.2f}M params")
    print(f"  Corruption: {'ON' if getattr(args, 'use_corruption', 1) else 'OFF'}", end="")
    if getattr(args, "use_corruption", 1):
        print(f"  (prob={args.corruption_prob}, mode={args.corruption_mode})")
    else:
        print()
    print(f"  Structural (Gabor): {'ON' if use_struct else 'OFF'}", end="")
    if use_struct:
        n_sh = sum(p.numel() for p in struct_head.parameters())
        mode = getattr(args, "struct_mode", "?")
        print(f"  (mode={mode}, A1={'on' if use_a1 else 'off'}, "
              f"A2={'on' if use_a2 else 'off'}, "
              f"gabor_bank K={gabor_bank.K}, head={n_sh/1e6:.3f}M params)")
        print(f"    w_a1={getattr(args,'w_a1','—') if use_a1 else '—'}  "
              f"w_a2={getattr(args,'w_a2','—') if use_a2 else '—'}  "
              f"loss_a1={getattr(args,'loss_a1','—') if use_a1 else '—'}  "
              f"loss_a2={getattr(args,'loss_a2','—') if use_a2 else '—'}  "
              f"weighting={getattr(args,'task_weighting','fixed')}")
    else:
        print()
    print(f"{'='*70}\n")

    

    for _ in range(global_step):
        lr_scheduler.step()
        wd_scheduler.step()
        next(momentum_schedule)

    for epoch in range(start_epoch, args.epochs):
        context_encoder.train()
        predictor.train()
        target_encoder.eval()
        if use_struct:
            struct_head.train()

        pbar = root_utils.make_epoch_progress_bar(dataloader, epoch, args)
        epoch_loss = 0.0
        n_batches = 0
        epoch_var_sum = 0.0
        epoch_var_count = 0

        ep_a1 = ep_a1_cos = ep_a1_top1 = 0.0
        ep_a2 = ep_a2_cos = ep_a2_top1 = 0.0
        n_a1 = n_a2 = 0
        ep_conflict = float("nan")

        for images, _ in pbar:
            images = images.to(device)
            B = images.size(0)

            context_masks, target_masks = MyUtils.Patchify(
                image_shape=(B, 3, images.size(2), images.size(3)),
                num_blocks=args.num_blocks,
                num_patches=args.num_patches,
                device=device,
            )

            if getattr(args, "use_corruption", 1):
                images_ctx = corrupt_visible_patches(images, context_masks, args)
            else:
                images_ctx = images
            context_embeddings = context_encoder(images_ctx, context_masks)

            if epoch == start_epoch and n_batches == 0:
                identical = torch.equal(images_ctx, images)
                print(f"[sanity] use_corruption={getattr(args,'use_corruption',1)}  "
                      f"images_ctx == images: {identical}  "
                      f"max|diff|={ (images_ctx - images).abs().max().item():.4f}")
            
            with torch.no_grad():
                z = context_embeddings.reshape(-1, context_embeddings.size(-1))
                if z.size(0) > 0:
                    batch_var = z.var(dim=0, unbiased=False).mean().item()
                    if batch_var == batch_var:
                        epoch_var_sum += batch_var
                        epoch_var_count += 1

                full_targets = target_encoder(images)
                target_embeddings = MyUtils.apply_masks(full_targets, target_masks)
                target_embeddings = MyUtils._repeat_interleave_batch(
                    target_embeddings, B, repeat=len(context_masks)
                )

            # A2 requests the extra structure query from the shared predictor.
            if use_a2:
                pred_embeddings, struct_hidden = predictor(
                    context_embeddings, context_masks, target_masks,
                    predict_structure=True)
            else:
                pred_embeddings = predictor(
                    context_embeddings, context_masks, target_masks)

            loss_jepa = MSE_loss(pred_embeddings, target_embeddings)
            loss = loss_jepa

            l_a1 = l_a2 = None
            if use_struct:
                with torch.no_grad():
                    desc = patch_energy_descriptor(
                        gabor_bank(images), args.num_patches)   # CLEAN image

                if use_a1:
                    t_a1 = MyUtils.apply_masks(desc, context_masks)   # visible patches
                    if epoch == start_epoch and n_batches == 0:
                        assert t_a1.shape[:2] == context_embeddings.shape[:2], (
                            f"A1 misalignment: t_a1 {tuple(t_a1.shape)} vs "
                            f"context_embeddings {tuple(context_embeddings.shape)}")
                    l_a1, s_a1 = structure_loss(
                        struct_head(context_embeddings), t_a1,
                        kind=args.loss_a1,
                        temperature=args.infonce_temp,
                        max_n=args.infonce_max_n)
                    ep_a1 += l_a1.item()
                    ep_a1_cos += s_a1["cos"]
                    if s_a1["top1"] == s_a1["top1"]:   # skip NaN (non-InfoNCE)
                        ep_a1_top1 += s_a1["top1"]
                    n_a1 += 1

                if use_a2:
                    t_a2 = MyUtils._repeat_interleave_batch(
                        MyUtils.apply_masks(desc, target_masks), B,
                        repeat=len(context_masks))               # hidden patches
                    if epoch == start_epoch and n_batches == 0:
                        assert t_a2.shape[:2] == struct_hidden.shape[:2], (
                            f"A2 misalignment: t_a2 {tuple(t_a2.shape)} vs "
                            f"struct_hidden {tuple(struct_hidden.shape)}")
                    l_a2, s_a2 = structure_loss(
                        struct_head(struct_hidden), t_a2,
                        kind=args.loss_a2,
                        temperature=args.infonce_temp,
                        max_n=args.infonce_max_n)
                    ep_a2 += l_a2.item()
                    ep_a2_cos += s_a2["cos"]
                    if s_a2["top1"] == s_a2["top1"]:
                        ep_a2_top1 += s_a2["top1"]
                    n_a2 += 1

                if epoch == start_epoch and n_batches == 0:
                    rep = sanity_report(gabor_bank, images, args.num_patches)
                    print("\n── Structural sanity check (epoch 1, batch 0) ──")
                    for k, v in rep.items():
                        print(f"    {k}: {v}")
                    if rep["desc_pair_cos"] > 0.95:
                        print("    !! WARNING: descriptors nearly identical "
                              "across patches — target carries little signal.")
                    if not (0.99 < rep["desc_norm_mean"] < 1.01):
                        print("    !! WARNING: descriptor norms != 1.0.")
                    if rep["resp_absmean"] < 1e-6:
                        print("    !! WARNING: Gabor responses ~0.")
                    print()

            # ─── Combine task losses ───
            if task_weighter is not None:
                terms = [loss_jepa]
                if l_a1 is not None:
                    terms.append(l_a1)
                if l_a2 is not None:
                    terms.append(l_a2)
                loss = task_weighter(terms)
            else:
                if l_a1 is not None:
                    loss = loss + args.w_a1 * l_a1
                if l_a2 is not None:
                    loss = loss + args.w_a2 * l_a2

            # ─── Gradient-conflict diagnostic on shared params ───
            # >0 complementary, ~0 orthogonal, <0 conflicting.
            if (use_struct and args.log_conflict and n_batches == 0
                    and ((epoch + 1) % args.gabor_log_every == 0 or epoch == start_epoch)):
                l_struct_tot = 0.0
                if l_a1 is not None:
                    l_struct_tot = l_struct_tot + l_a1
                if l_a2 is not None:
                    l_struct_tot = l_struct_tot + l_a2
                ep_conflict = grad_conflict_cosine(
                    loss_jepa, l_struct_tot,
                    list(context_encoder.norm.parameters()))

            _new_lr = lr_scheduler.step()
            _new_wd = wd_scheduler.step()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            momentum = next(momentum_schedule)
            MyUtils.update_ema(context_encoder, target_encoder, momentum=momentum)

            global_step += 1
            epoch_loss += loss_jepa.item()
            n_batches += 1
            pbar.set_postfix(
                loss=f"{epoch_loss / n_batches:.4f}",
                lr=f"{_new_lr:.2e}",
                wd=f"{_new_wd:.2e}",
                mom=f"{momentum:.4f}",
            )

        root_utils.finish_epoch_progress_bar(pbar)
        epoch_loss /= max(n_batches, 1)
        epoch_losses.append(epoch_loss)
        feat_var = epoch_var_sum / max(epoch_var_count, 1)
        print(
            f"Epoch {epoch+1} | loss={epoch_loss:.4f} | feature_var={feat_var:.6f} "
            f"| corruption={'ON' if getattr(args,'use_corruption',1) else 'OFF'} "
            f"| struct={'ON' if use_struct else 'OFF'} "
            f"| MSE(z_p, z_2)"
        )
        if use_a1:
            print(f"    A1 (visible): loss={ep_a1/max(n_a1,1):.4f}  "
                  f"cos={ep_a1_cos/max(n_a1,1):.3f}  "
                  f"top1={ep_a1_top1/max(n_a1,1):.3f}")
        if use_a2:
            print(f"    A2 (hidden):  loss={ep_a2/max(n_a2,1):.4f}  "
                  f"cos={ep_a2_cos/max(n_a2,1):.3f}  "
                  f"top1={ep_a2_top1/max(n_a2,1):.3f}")
        if use_struct and args.log_conflict:
            msg = f"    conflict_cos={ep_conflict:+.4f}"
            if task_weighter is not None:
                ws = "  ".join(f"{w:.3f}" for w in task_weighter.weights())
                msg += f"   learned_w=[{ws}]"
            print(msg)

        models = {
            "context": context_encoder,
            "target": target_encoder,
            "predictor": predictor,
        }
        if use_struct:
            models["struct_head"] = struct_head
            if task_weighter is not None:
                models["task_weighter"] = task_weighter

        eval_history, best_acc = periodic_eval.maybe_eval_epoch(
            epoch, context_encoder, args, eval_history, _extract_eval_features,
            best_acc=best_acc,
            models=models,
            opt=opt,
            run_dir=run_dir,
            global_step=global_step,
        )
        checkpoint_state["eval_history"] = eval_history
        checkpoint_state["best_acc"] = best_acc
        MyUtils.save_epoch(run_dir, models, opt, epoch, global_step, best_acc)

    return epoch_losses, eval_history


def run_linear_probing(folder_name, args):
    ckpt_path = MyUtils.resolve_ckpt_path(folder_name, args)
    train_set, test_set = datasets_root.load_eval_dataset(args)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)
    encoder = MyUtils.load_frozen_context_encoder(ckpt_path, args)
    feature_extractor = MyModels.FeatureExtractor(encoder)
    return MyUtils.linear_probe(
        feature_extractor, train_loader, test_loader,
        num_classes=datasets_root.eval_num_classes(args),
        lr=args.eval_lr, epochs=args.eval_epochs, device=args.device,
    )


def run_knn_evaluation(folder_name, args):
    ckpt_path = MyUtils.resolve_ckpt_path(folder_name, args)
    train_set, test_set = datasets_root.load_eval_dataset(args)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    encoder = MyUtils.load_frozen_context_encoder(ckpt_path, args)
    feature_extractor = MyModels.FeatureExtractor(encoder)
    return MyUtils.knn_evaluate(
        feature_extractor, train_loader, test_loader,
        k=args.K, num_classes=datasets_root.eval_num_classes(args), device=args.device,
    )
