import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import Datasets as datasets_root
import MyModels
import MyUtils
import Utils as root_utils
import periodic_eval


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
):
    device = args.device
    epoch_losses = []
    global_step = checkpoint_state["global_step"]
    start_epoch = checkpoint_state["start_epoch"]
    run_dir = checkpoint_state["run_dir"]
    best_acc = checkpoint_state.get("best_acc", checkpoint_state.get("best_loss", float("-inf")))
    eval_history = list(checkpoint_state.get("eval_history", []))

    for _ in range(global_step):
        lr_scheduler.step()
        wd_scheduler.step()
        next(momentum_schedule)

    for epoch in range(start_epoch, args.epochs):
        context_encoder.train()
        predictor.train()
        target_encoder.eval()

        pbar = root_utils.make_epoch_progress_bar(dataloader, epoch, args)
        epoch_loss = 0.0
        n_batches = 0
        epoch_var_sum = 0.0
        epoch_var_count = 0

        for images, _ in pbar:
            images = images.to(device)
            B = images.size(0)

            context_masks, target_masks = MyUtils.Patchify(
                image_shape=(B, 3, images.size(2), images.size(3)),
                num_blocks=args.num_blocks,
                num_patches=args.num_patches,
                device=device,
            )

            images_ctx = corrupt_visible_patches(images, context_masks, args)
            context_embeddings = context_encoder(images_ctx, context_masks)

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

            pred_embeddings = predictor(context_embeddings, context_masks, target_masks)
            loss = MSE_loss(pred_embeddings, target_embeddings)

            _new_lr = lr_scheduler.step()
            _new_wd = wd_scheduler.step()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            momentum = next(momentum_schedule)
            MyUtils.update_ema(context_encoder, target_encoder, momentum=momentum)

            global_step += 1
            epoch_loss += loss.item()
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
            f"| visible-patch corrupt (blue/red/green/jitter/noise) | MSE(z_p, z_2)"
        )

        models = {
            "context": context_encoder,
            "target": target_encoder,
            "predictor": predictor,
        }
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
