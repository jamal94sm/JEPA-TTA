
import torch
import sys
from tqdm import tqdm
import time
from . import MyUtils
import os
from Datasets.CIFAR10.loader import load_dataset 
from torch.utils.data import DataLoader
from . import MyModels
from Baselines.periodic_eval import maybe_eval_epoch
import Utils as root_utils


def _extract_eval_features(encoder, x):
    B = x.size(0)
    P = encoder.pos_embed.size(1)
    full_mask = [torch.arange(P, device=x.device).unsqueeze(0).expand(B, -1)]
    z = encoder(x, full_mask)
    return z.mean(dim=1)


def MSE_loss(preds, targets):
    return torch.nn.functional.smooth_l1_loss(preds, targets)



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
    args):

    device = args.device
    
    epoch_losses  = []
    global_step   = checkpoint_state["global_step"]
    start_epoch   = checkpoint_state["start_epoch"]
    run_dir       = checkpoint_state["run_dir"]
    best_loss     = checkpoint_state["best_loss"]
    eval_history  = list(checkpoint_state.get("eval_history", []))

    # fast-forward all schedules if resuming mid-training
    for _ in range(global_step):
        lr_scheduler.step()
        wd_scheduler.step()
        next(momentum_schedule)

    # ------------------------------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        context_encoder.train()
        predictor.train()
        target_encoder.eval()

        pbar = root_utils.make_epoch_progress_bar(dataloader, epoch, args)
        epoch_loss = 0.0
        n_batches  = 0

        # >>> NEW: feature variance accumulators (SCALARS ONLY)
        epoch_var_sum   = 0.0
        epoch_var_count = 0

        for images, _ in pbar:
            images = images.to(device)
            B      = images.size(0)

            context_masks, target_masks = MyUtils.Patchify(
                image_shape=(B, 3, images.size(2), images.size(3)),
                num_blocks=args.num_blocks,
                num_patches=args.num_patches,
                device=device,
            )

            # ---- context encoder ----
            context_embeddings = context_encoder(images, context_masks)  # (B, N_ctx, D)

            # >>> NEW: per-batch feature variance (handles variable N_ctx safely)
            with torch.no_grad():
                z = context_embeddings.reshape(-1, context_embeddings.size(-1))
                batch_var = z.var(dim=0, unbiased=False).mean().item()
                epoch_var_sum += batch_var
                epoch_var_count += 1

            # ---- target encoder (EMA, no grad) ----
            with torch.no_grad():
                full_targets_embeddings = target_encoder(images)  # (B, P, D)
                target_embeddings = MyUtils.apply_masks(full_targets_embeddings, target_masks)
                target_embeddings = MyUtils._repeat_interleave_batch(target_embeddings, B, repeat=len(context_masks))

            # ---- predictor ----
            pred_embeddings = predictor(context_embeddings, context_masks, target_masks)

            # ---- loss ----
            loss = MSE_loss(pred_embeddings, target_embeddings)

            # ---- optimise ----
            _new_lr = lr_scheduler.step()
            _new_wd = wd_scheduler.step()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            # ---- EMA update ----
            momentum = next(momentum_schedule)
            MyUtils.update_ema(context_encoder, target_encoder, momentum=momentum)

            global_step += 1
            epoch_loss  += loss.item()
            n_batches   += 1

            pbar.set_postfix(
                loss=f"{epoch_loss / n_batches:.4f}",
                lr=f"{_new_lr:.2e}",
                wd=f"{_new_wd:.2e}",
                mom=f"{momentum:.4f}",)

        root_utils.finish_epoch_progress_bar(pbar)
        epoch_loss /= max(n_batches, 1)
        epoch_losses.append(epoch_loss)

        # >>> NEW: epoch-level feature variance (single scalar)
        feat_var = epoch_var_sum / max(epoch_var_count, 1)
        print(
            f"Epoch {epoch+1} | loss={epoch_loss:.4f} | feature_var={feat_var:.6f}"
        )

        models = {
            "context"  : context_encoder,
            "target"   : target_encoder,
            "predictor": predictor,
        }
        MyUtils.save_epoch(run_dir, models, opt, epoch, global_step, best_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            MyUtils.save_best(
                run_dir, models, opt, epoch, global_step, best_loss
            )

        eval_history = maybe_eval_epoch(
            epoch, context_encoder, args, eval_history, _extract_eval_features
        )
        checkpoint_state["eval_history"] = eval_history

    return epoch_losses, eval_history



##############################################################################################
##############################################################################################


def run_linear_probing(folder_name, args):
    ckpt_path = MyUtils.resolve_ckpt_path(folder_name, args)
    
    if args.eval_dataset.lower() in {"cifar10"}:
        train_set, test_set = load_dataset(root=os.path.join("Datasets/data_bank", args.dataset), args=args)
    else: 
        raise ValueError(f"Unknown dataset: {args.dataset} for validation")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size)

    encoder = MyUtils.load_frozen_context_encoder(ckpt_path, args)
    feature_extractor = MyModels.FeatureExtractor(encoder)

    acc = MyUtils.linear_probe(
        feature_extractor,
        train_loader,
        test_loader,
        num_classes=10,
        lr=args.eval_lr,
        epochs=args.eval_epochs,
        device=args.device,
    )
    return acc


def run_knn_evaluation(folder_name, args):
    k = args.K
    ckpt_path = MyUtils.resolve_ckpt_path(folder_name, args)

    if args.eval_dataset.lower() in {"cifar10"}:
        train_set, test_set = load_dataset(
            root=os.path.join("Datasets/data_bank", args.dataset),
            args=args
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False)

    encoder = MyUtils.load_frozen_context_encoder(ckpt_path, args)
    feature_extractor = MyModels.FeatureExtractor(encoder)

    acc = MyUtils.knn_evaluate(
        feature_extractor,
        train_loader,
        test_loader,
        k=k,
        num_classes=10,
        device=args.device,
    )
    return acc