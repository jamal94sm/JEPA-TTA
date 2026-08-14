import os

import torch
from torch.utils.data import DataLoader

import checkpoint_init
import MyFuncs
import MyModels
import MyUtils
import Utils as root_utils

baseline_name = "JEPA_corruption_all"


def run(dataset, args):
    """
    Standard JEPA: loss = MSE(z_p, z_2) on clean EMA targets.
    Context encoder sees the same image with only *visible* patches corrupted
    (each visible patch independently: blue / red / green / jitter / noise).
    """
    args.baseline_name = baseline_name
    ckpt_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    images_shape = (args.batch_size, 3, args.image_size[0], args.image_size[1])

    context_encoder = MyModels.Context_Encoder(
        images_shape[-2:], args.num_patches, args.embed_dim,
        depth=args.encoder_depth, num_heads=args.heads,
    ).to(args.device)
    target_encoder = MyModels.Target_Encoder(
        images_shape[-2:], args.num_patches, args.embed_dim,
        depth=args.encoder_depth, num_heads=args.heads,
    ).to(args.device)
    predictor = MyModels.Predictor(args.num_patches, args.embed_dim).to(args.device)

    models = {"context": context_encoder, "target": target_encoder, "predictor": predictor}
    loaded_from_init = checkpoint_init.maybe_load_initialization(
        args, models, profile="jepa", method_name=baseline_name
    )

    if root_utils.maybe_eval_only(
        context_encoder, args, MyFuncs._extract_eval_features, models, ckpt_base
    ):
        return

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
    )

    if not loaded_from_init:
        for pc, pt in zip(context_encoder.parameters(), target_encoder.parameters()):
            pt.data.copy_(pc.data)
    for p in target_encoder.parameters():
        p.requires_grad = False

    opt = torch.optim.AdamW(
        list(context_encoder.parameters()) + list(predictor.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    total_steps = args.epochs * len(dataloader)
    lr_scheduler = MyUtils.WarmupCosineSchedule(
        optimizer=opt,
        warmup_steps=int(args.warmup_ratio * total_steps),
        start_lr=args.start_lr,
        ref_lr=args.learning_rate,
        total_steps=total_steps,
        final_lr=args.final_lr,
    )
    wd_scheduler = MyUtils.CosineWDSchedule(
        optimizer=opt,
        ref_wd=args.weight_decay,
        total_steps=total_steps,
        final_wd=args.final_weight_decay,
    )
    momentum_schedule = (
        args.ema_start + i * (args.ema_end - args.ema_start) / total_steps
        for i in range(total_steps + 1)
    )

    key = str(getattr(args, "dataset", "stl10")).lower()
    prefix = "TIN" if "tiny" in key else "STL"
    args.ckpt_run_name = getattr(args, "ckpt_run_name", None) or prefix

    checkpoint_state = MyUtils.prepare_checkpoint_state(models, opt, ckpt_base, args)
    print(f"Checkpoints → {checkpoint_state['run_dir']}")

    epoch_losses, eval_history = MyFuncs.Train(
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
    )

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")
    MyUtils.Plot(epoch_losses, plot_name=baseline_name, results_dir=results_dir)
    plot_path = root_utils.plot_eval_progress(
        eval_history, baseline_name, args.evaluation, results_dir=results_dir
    )
    if plot_path:
        print(f"Saved eval progress plot: {plot_path}")
    args.results_dir = results_dir
    tsne_path = root_utils.live_eval_tsne(
        context_encoder, args, MyFuncs._extract_eval_features, args.epochs
    )
    print(f"Saved t-SNE: {tsne_path}")
