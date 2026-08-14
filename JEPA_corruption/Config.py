import argparse
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
    parser.add_argument('--corruption_std',            type=float, default=0.1)
    parser.add_argument('--jitter_strength',           type=float, default=0.4)

    parser.add_argument('--learning_rate',             type=float, default=3e-4)
    parser.add_argument('--start_lr',                  type=float, default=1e-6)
    parser.add_argument('--final_lr',                  type=float, default=1e-6)
    parser.add_argument('--warmup_ratio',              type=float, default=0.1)
    parser.add_argument('--weight_decay',              type=float, default=0.05)
    parser.add_argument('--final_weight_decay',        type=float, default=0.1)

    return parser.parse_args()
