import argparse
import json
import torch


def get_arguments():
    parser = argparse.ArgumentParser(description="JEPA_corruption_all")

    parser.add_argument('--device',                    type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--dataset',                   type=str,   default='stl10',          choices=['stl-10', 'stl10', 'tiny-imagenet'])
    parser.add_argument('--eval_dataset',              type=str,   default='cifar10',          choices=['cifar10', 'cifar100', 'stl10', 'tiny-imagenet'])
    parser.add_argument('--data_root',                 type=str,   default=None)
    parser.add_argument('--num_patches',               type=int,   default=6)
    parser.add_argument('--image_size',                type=int,   nargs=2,                  default=[96, 96])
    parser.add_argument('--evaluation',                type=str,   default='knn',            choices=['linear', 'knn'])

    parser.add_argument('--embed_dim',                 type=int,   default=256)
    parser.add_argument('--encoder_depth',             type=int,   default=6)
    parser.add_argument('--heads',                     type=int,   default=8)
    parser.add_argument('--num_blocks',                type=int,   default=1)
    parser.add_argument('--num_workers',               type=int,   default=2)

    parser.add_argument('--epochs',                    type=int,   default=150)
    parser.add_argument('--batch_size',                type=int,   default=1024)

    parser.add_argument('--eval_only',                 type=int,   default=0,         choices=[0, 1])
    parser.add_argument('--eval_noise',                type=int,   default=0,         choices=[0, 1])
    parser.add_argument('--initialization',            type=str,   default=None)
    parser.add_argument('--K',                         type=int,   default=20)
    parser.add_argument('--eval_epochs',               type=int,   default=20)
    parser.add_argument('--eval_lr',                   type=float, default=1e-2)
    parser.add_argument('--num_ep_for_eval',           type=int,   default=1)

    parser.add_argument('--ema_sg',                    type=int,   default=1,         choices=[0, 1])
    parser.add_argument('--ema_start',                 type=float, default=0.996)
    parser.add_argument('--ema_end',                   type=float, default=0.999)

    # Visible-patch corruptions (one type drawn uniformly per visible patch):
    # blue, red, green, jitter, noise. Targets stay clean. Loss = MSE(z_p, z_2).
    #parser.add_argument('--corruption_std',            type=float, default=0.1)
    #parser.add_argument('--jitter_strength',           type=float, default=0.4)

    parser.add_argument('--learning_rate',             type=float, default=3e-4)
    parser.add_argument('--start_lr',                  type=float, default=1e-6)
    parser.add_argument('--final_lr',                  type=float, default=1e-6)
    parser.add_argument('--warmup_ratio',              type=float, default=0.1)
    parser.add_argument('--weight_decay',              type=float, default=0.05)
    parser.add_argument('--final_weight_decay',        type=float, default=0.1)

    # --- corruption gating ---
    parser.add_argument('--use_corruption', type=int, default=1, choices=[0, 1])
    parser.add_argument('--corruption_prob',        type=float, default=0.5,
                         help="Per-sample probability of being corrupted at all. "
                              "Expected #corrupted per batch = corruption_prob * batch_size, "
                              "actual count varies batch to batch (Bernoulli draw).")
    parser.add_argument('--corruption_mode',        type=str,   default='mixed',
                         choices=['single', 'mixed'],
                         help="single: each corrupted sample gets exactly one corruption type. "
                              "mixed: each corrupted sample gets a random subset of types.")
    parser.add_argument('--mix_prob',                type=float, default=0.4,
                         help="[mixed mode only] per-type inclusion probability. "
                              "Each corrupted sample independently includes each corruption "
                              "type with this probability (at least one is forced in).")

    # --- per-corruption severity (each is a *maximum*; actual severity for an "
    #     applied" sample is drawn U(0, max) independently, so severity varies too) ---
    parser.add_argument('--color_temp_strength',     type=float, default=0.25,   # illumination/white-balance shift
                         help="Max R/B channel scale shift, simulates illuminant/spectrum change.")
    parser.add_argument('--gamma_strength',          type=float, default=0.3,    # exposure/sensor response
                         help="Max deviation of gamma exponent from 1.0.")
    parser.add_argument('--channel_mix_strength',    type=float, default=0.15,   # spectral crosstalk
                         help="Max off-diagonal magnitude of the random 3x3 channel-mix matrix.")
    parser.add_argument('--desaturate_strength',     type=float, default=0.5,    # NIR-band spectrum change
                         help="Max blend-toward-grayscale fraction.")
    parser.add_argument('--blur_sigma_max',          type=float, default=1.5,    # device/resolution change
                         help="Max Gaussian blur sigma (pixels).")
    parser.add_argument('--corruption_std',          type=float, default=0.08,   # sensor noise (kept, renamed use)
                         help="Max additive Gaussian noise std.")
    parser.add_argument('--vignette_strength',       type=float, default=0.3,    # optics/acquisition geometry
                         help="Max radial darkening at image corners.")

    # ─── Gabor structural tasks (A1 = visible patches, A2 = hidden patches) ──
    parser.add_argument('--struct_mode',        type=str,   default='none',
                         choices=['none', 'a1', 'a2', 'both'],
                         help="a1 = structure on visible patches (via context_embeddings). "
                              "a2 = structure on hidden patches (via predictor). "
                              "both = A1+A2 sharing one structure head.")
    parser.add_argument('--w_a1',               type=float, default=0.3)
    parser.add_argument('--w_a2',               type=float, default=0.3)
    parser.add_argument('--struct_head_hidden', type=int,   default=128)

    parser.add_argument('--struct_loss',        type=str,   default='cosine',
                         choices=['cosine', 'infonce', 'smooth_l1'],
                         help="Default structural loss for both tasks.")
    parser.add_argument('--struct_loss_a1',     type=str,   default=None,
                         choices=['cosine', 'infonce', 'smooth_l1'],
                         help="Override --struct_loss for the A1 (visible) task.")
    parser.add_argument('--struct_loss_a2',     type=str,   default=None,
                         choices=['cosine', 'infonce', 'smooth_l1'],
                         help="Override --struct_loss for the A2 (hidden) task.")
    parser.add_argument('--infonce_temp',       type=float, default=0.1)
    parser.add_argument('--infonce_max_n',      type=int,   default=4096)

    parser.add_argument('--task_weighting',     type=str,   default='fixed',
                         choices=['fixed', 'uncertainty'])

    parser.add_argument('--gabor_orient',       type=int,   default=8)
    parser.add_argument('--gabor_gray',         type=int,   default=0, choices=[0, 1],
                         help="1 = collapse to grayscale. 0 = per-channel Gabor "
                              "(correct default here — STL/CIFAR/Tiny-ImageNet are RGB, "
                              "unlike CASIA-MS).")
    parser.add_argument('--gabor_scales',       type=str,
                         default='[[9,3,6],[15,5,10],[21,7,14]]',
                         help="JSON list of [kernel_size, sigma, lambda] triples "
                              "for the Gabor bank. Old default was "
                              "[[9,3,6],[15,5,10],[21,7,14]] -- that version "
                              "let filters bleed across patch boundaries; the "
                              "new default keeps max kernel size (13) below "
                              "the patch size (image_size/num_patches, 16px "
                              "for the stl10/tiny-imagenet defaults) to stay "
                              "patch-local. Re-derive the cap if you change. try: [[5,1.5,3.0],[9,3.0,6.0],[13,4.5,9.0]] "
                              "--image_size or --num_patches.")
    parser.add_argument('--gabor_log_every',    type=int,   default=5,
                         help="Print structural/diagnostic logs every N epochs.")
    parser.add_argument('--log_conflict',       type=int,   default=1, choices=[0, 1],
                         help="Log cosine between task gradients on shared params.")

    args = parser.parse_args()
    args.use_a1 = args.struct_mode in ('a1', 'both')
    args.use_a2 = args.struct_mode in ('a2', 'both')
    args.loss_a1 = args.struct_loss_a1 or args.struct_loss
    args.loss_a2 = args.struct_loss_a2 or args.struct_loss

    # --gabor_scales arrives as a JSON string (argparse can't take nested
    # tuples directly) -- parse into the tuple-of-tuples GaborBank expects.
    args.gabor_scales = tuple(tuple(s) for s in json.loads(args.gabor_scales))

    return args
