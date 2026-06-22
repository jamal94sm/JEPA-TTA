from . import MyUtils
from . import MyModels
from . import MyFuncs
import torch
from torch.utils.data import DataLoader
import os
import Utils as root_utils
from Baselines import checkpoint_init

folder_name = os.path.join( "Baselines", os.path.basename(os.path.dirname(__file__)) )
baseline_name = os.path.basename(os.path.dirname(__file__))




def run(dataset, args):

    # ---------------------------- Preparing Data ----------------------------
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
    )
    images_shape = (args.batch_size, 3, args.image_size[0], args.image_size[1])


    # ---------------------------- Preparing Models ----------------------------
    context_encoder = MyModels.Context_Encoder(images_shape[-2:], args.num_patches, args.embed_dim).to(args.device)
    target_encoder  = MyModels.Target_Encoder(images_shape[-2:],  args.num_patches, args.embed_dim).to(args.device)
    predictor       = MyModels.Predictor(                         args.num_patches, args.embed_dim).to(args.device)

    models = {"context": context_encoder, "target": target_encoder, "predictor": predictor}
    loaded_from_init = checkpoint_init.maybe_load_initialization(
        args, models, profile="jepa", method_name=baseline_name
    )

    # target <--- context, and freeze target
    if not loaded_from_init:
        for pc, pt in zip(context_encoder.parameters(), target_encoder.parameters()):
            pt.data.copy_(pc.data)
    for p in target_encoder.parameters():
        p.requires_grad = False


    # ---------------------------- opt & scd ----------------------------
    opt = torch.optim.AdamW(
        list(context_encoder.parameters()) + list(predictor.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,)

    total_steps = args.epochs * len(dataloader)
    lr_scheduler = MyUtils.WarmupCosineSchedule(
        optimizer    = opt,
        warmup_steps = int(args.warmup_ratio * total_steps),
        start_lr     = args.start_lr,
        ref_lr       = args.learning_rate,
        total_steps  = total_steps,
        final_lr     = args.final_lr)

    wd_scheduler = MyUtils.CosineWDSchedule(
        optimizer   = opt,
        ref_wd      = args.weight_decay,
        total_steps = total_steps,
        final_wd    = args.final_weight_decay)
    
    momentum_schedule = ( args.ema_start + i * (args.ema_end - args.ema_start) / total_steps for i in range(total_steps + 1))


    # ---------------------------- Checkpoint Affairs ----------------------------
    checkpoint_state = MyUtils.prepare_checkpoint_state(models, opt, os.path.join(folder_name, "checkpoints"), args)


    # ---------------------------- Training the Model ----------------------------
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
                args)

    MyUtils.Plot(epoch_losses, plot_name=baseline_name)
    plot_path = root_utils.plot_eval_progress(eval_history, baseline_name, args.evaluation)
    if plot_path:
        print(f"Saved eval progress plot: {plot_path}")



