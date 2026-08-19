import os
import torch
from torch.utils.data import DataLoader
import checkpoint_init
import MyFuncs
import MyModels
import MyUtils
import Utils as root_utils
from gabor import GaborBank

baseline_name = "JEPA_corruption_all"

def run(dataset, args):
    """
    Standard JEPA: loss = MSE(z_p, z_2) on clean EMA targets.
    Context encoder sees the same image with only *visible* patches corrupted
    (each visible patch independently: blue / red / green / jitter / noise).

    Optional structural auxiliary tasks (--struct_mode a1 / a2 / both):
    A1 predicts Gabor line-structure descriptors on VISIBLE patches directly
    from context_embeddings; A2 predicts them on HIDDEN patches via the
    predictor's structure task token. Both share one StructureHead.
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

    # ─── Structural auxiliary tasks (A1 / A2) ──────────────────
    use_a1 = bool(getattr(args, "use_a1", False))
    use_a2 = bool(getattr(args, "use_a2", False))
    use_struct = use_a1 or use_a2
    gabor_bank = struct_head = task_weighter = None

    if use_struct:
        gabor_bank = GaborBank(
            n_orient=args.gabor_orient,
            scales=args.gabor_scales,                          # NEW
            per_channel=not bool(args.gabor_gray),
        ).to(args.device)
        struct_head = MyModels.StructureHead(
            args.embed_dim, gabor_bank.K,
            hidden=args.struct_head_hidden).to(args.device)
        n_tasks = 1 + int(use_a1) + int(use_a2)
        if args.task_weighting == "uncertainty":
            task_weighter = MyModels.UncertaintyWeighting(n_tasks).to(args.device)

        n_sh = sum(p.numel() for p in struct_head.parameters())
        mode = "per-channel RGB" if gabor_bank.per_channel else "grayscale"
        print(f"Struct mode: {args.struct_mode}   loss_a1={args.loss_a1}  "
              f"loss_a2={args.loss_a2}   weighting={args.task_weighting}")
        print(f"Gabor bank: K={gabor_bank.K} "
              f"({args.gabor_orient} orient x {gabor_bank.n_scales} scales, {mode})")
        print(f"Structure head: {n_sh/1e6:.3f}M params "
              f"(hidden={args.struct_head_hidden} -> {gabor_bank.K})")

        models["struct_head"] = struct_head
        if task_weighter is not None:
            models["task_weighter"] = task_weighter

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

    train_params = list(context_encoder.parameters()) + list(predictor.parameters())
    if use_struct:
        train_params += list(struct_head.parameters())
    if task_weighter is not None:
        train_params += list(task_weighter.parameters())

    opt = torch.optim.AdamW(
        train_params,
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
        gabor_bank=gabor_bank,
        struct_head=struct_head,
        task_weighter=task_weighter,
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
