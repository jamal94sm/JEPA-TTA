
import os
import sys
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

def set_seed(seed: int = 42, deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = deterministic
    cudnn.benchmark = False
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


##############################################################################################
##############################################################################################

##############################################################################################
##############################################################################################

def schedule_beta(epoch, args):
    """Linear schedule: beta_early at epoch 0 → beta_late at the last epoch."""
    beta_early = getattr(args, "beta_early", getattr(args, "beta_min", 0.1))
    beta_late = getattr(args, "beta_late", getattr(args, "beta_max", 1.0))
    if args.epochs <= 1:
        return beta_late
    t = epoch / max(args.epochs - 1, 1)
    return beta_early + (beta_late - beta_early) * t


def select_pred_repr(z, mu, args):
    """BA / InfoNCE / MINE: encoder mean mu if --pred_use_mu, else reparameterized z."""
    if getattr(args, "pred_use_mu", False):
        return mu
    return z


def _format_tqdm_postfix(postfix):
    if not postfix:
        return ""
    if isinstance(postfix, str):
        return postfix
    if isinstance(postfix, dict):
        return ", ".join(f"{k}={v}" for k, v in postfix.items())
    return str(postfix)


def make_epoch_progress_bar(dataloader, epoch, args):
    """One tqdm bar per epoch (stderr); stays visible after the epoch (leave=True)."""
    return tqdm(
        dataloader,
        desc=f"Epoch {epoch + 1}/{args.epochs}",
        leave=True,
        dynamic_ncols=True,
        file=sys.stderr,
    )


def finish_epoch_progress_bar(pbar):
    """Close bar (keeps it on the terminal) and log final metrics to stdout for Logs_*.txt."""
    desc = getattr(pbar, "desc", "") or ""
    postfix = _format_tqdm_postfix(getattr(pbar, "postfix", None))
    n = getattr(pbar, "n", None)
    total = getattr(pbar, "total", None)
    pbar.close()
    if n is not None and total:
        progress = f"{n}/{total}"
        line = f"{desc} [{progress}]"
    else:
        line = desc
    if postfix:
        line = f"{line} | {postfix}"
    print(line, flush=True)


##############################################################################################
# Shared JEPA Patchify (context / target index masks)
##############################################################################################


def Patchify(
    image_shape,
    num_blocks=1,
    num_patches=14,
    trg_ratio=(0.15, 0.20),
    ctx_ratio=(0.85, 1.00),
    ar_range=(0.75, 1.5),
    device="cpu",
    min_ctx_tokens=1,
    max_sample_tries=40,
):
    """
    Sample integer index masks for JEPA context / target blocks.

    ``num_blocks`` is the number of *target* blocks, not ViT encoder depth.
    On small grids (e.g. 6x6), many blocks can cover every patch; we resample
    and, if needed, force a non-empty context so encoders never see length-0
    sequences (which yield NaN feature variance and broken training).
    """
    import math

    B, _, _, _ = image_shape
    H = W = num_patches
    P = H * W
    if P < 2:
        raise ValueError(f"num_patches={num_patches} gives P={P}; need at least 2 patches.")
    if num_blocks < 1:
        raise ValueError(f"num_blocks must be >= 1, got {num_blocks}.")

    def sample_block(scale):
        s = torch.empty(()).uniform_(*scale).item()
        ar = torch.empty(()).uniform_(*ar_range).item()
        area = max(1, int(s * P))
        h = max(1, min(H, int(round(math.sqrt(area * ar)))))
        w = max(1, min(W, int(round(area / max(ar, 1e-8)))))
        y = torch.randint(0, H - h + 1, ())
        x = torch.randint(0, W - w + 1, ())
        idx = [(y + i) * W + (x + j) for i in range(h) for j in range(w)]
        return torch.tensor(idx, device=device, dtype=torch.long)

    ctx_masks = []
    tgt_masks = [[] for _ in range(num_blocks)]
    min_ctx = P
    min_tgt = P

    for _ in range(B):
        ctx = None
        sample_tgts = None
        for _try in range(max_sample_tries):
            occupied = torch.zeros(P, dtype=torch.bool, device=device)
            sample_tgts = []
            for _k in range(num_blocks):
                idx = sample_block(trg_ratio)
                sample_tgts.append(idx)
                occupied[idx] = True

            cand = None
            for _ in range(10):
                c = sample_block(ctx_ratio)
                c = c[~occupied[c]]
                if c.numel() >= min_ctx_tokens:
                    cand = c
                    break
            if cand is None:
                cand = (~occupied).nonzero(as_tuple=False).squeeze(1)
            if cand.numel() >= min_ctx_tokens:
                ctx = cand
                break

        if ctx is None or ctx.numel() < min_ctx_tokens:
            # Targets covered the grid: reserve a forced context set (may overlap targets).
            n_force = max(min_ctx_tokens, max(1, P // 4))
            ctx = torch.randperm(P, device=device)[:n_force]
            if sample_tgts is None:
                occupied = torch.ones(P, dtype=torch.bool, device=device)
                occupied[ctx] = False
                sample_tgts = []
                for _k in range(num_blocks):
                    sample_tgts.append(sample_block(trg_ratio))

        for k in range(num_blocks):
            tgt_masks[k].append(sample_tgts[k])
            min_tgt = min(min_tgt, sample_tgts[k].numel())

        min_ctx = min(min_ctx, int(ctx.numel()))
        ctx_masks.append(ctx)

    if min_ctx < min_ctx_tokens:
        raise RuntimeError(
            f"Patchify produced empty/too-small context (min_ctx={min_ctx}) with "
            f"num_blocks={num_blocks}, num_patches={num_patches} ({P} patches). "
            f"--num_blocks is JEPA target-mask count, not ViT depth (use --encoder_depth). "
            f"On a {num_patches}x{num_patches} grid use --num_blocks 1 (recommended)."
        )

    ctx_out = torch.stack(
        [c[torch.randperm(c.numel(), device=device)[:min_ctx]] for c in ctx_masks]
    )
    tgt_out = [
        torch.stack(
            [t[torch.randperm(t.numel(), device=device)[:min_tgt]] for t in tgt_masks[k]]
        )
        for k in range(num_blocks)
    ]
    return [ctx_out], tgt_out


def Plot(values, plot_name='figure', results_dir='Results'):
    import matplotlib.pyplot as plt

    os.makedirs(results_dir, exist_ok=True)
    x = range(1, len(values) + 1)
    plt.figure(figsize=(5, 3))
    plt.plot(x, values, marker='o')
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title(plot_name)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(results_dir, f"Loss_{plot_name}.png")
    plt.savefig(path, dpi=200)
    plt.close()
    return path


##############################################################################################
# Periodic downstream evaluation (linear probe / k-NN)
##############################################################################################

_EVAL_LOADER_CACHE = {}


def should_run_eval(epoch_one_based, args):
    n = getattr(args, "num_ep_for_eval", 0)
    if n <= 0:
        return False
    return epoch_one_based % n == 0 or epoch_one_based == args.epochs


def get_eval_dataloaders(args):
    from torch.utils.data import DataLoader
    import Datasets

    key = (
        Datasets.normalize_eval_dataset_name(args.eval_dataset),
        args.batch_size,
        args.num_workers,
    )
    if key not in _EVAL_LOADER_CACHE:
        train_set, test_set = Datasets.load_eval_dataset(args)
        _EVAL_LOADER_CACHE[key] = (
            DataLoader(train_set, batch_size=args.batch_size, shuffle=True),
            DataLoader(test_set, batch_size=args.batch_size, shuffle=False),
        )
    return _EVAL_LOADER_CACHE[key]


_EVAL_NOISE_LEVELS = (0.0, 0.2, 0.4, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 1.9)


def _add_gaussian_noise(x, noise_std):
    """Add N(0, noise_std^2) to image tensor; no-op when noise_std <= 0."""
    if noise_std is None or float(noise_std) <= 0.0:
        return x
    return x + float(noise_std) * torch.randn_like(x)


def linear_probe_eval(feature_extractor, args, noise_std=0.0):
    import torch.nn as nn
    import torch.optim as optim
    import Datasets

    device = args.device
    train_loader, test_loader = get_eval_dataloaders(args)
    num_classes = Datasets.eval_num_classes(args)
    noise_std = float(noise_std)

    with torch.no_grad():
        x0, _ = next(iter(train_loader))
        x0 = _add_gaussian_noise(x0.to(device), noise_std)
        feat_dim = feature_extractor(x0).shape[-1]

    clf = nn.Linear(feat_dim, num_classes).to(device)
    opt = optim.Adam(clf.parameters(), lr=args.eval_lr)
    loss_fn = nn.CrossEntropyLoss()

    feature_extractor.eval()
    for _ in range(args.eval_epochs):
        clf.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x = _add_gaussian_noise(x, noise_std)
            with torch.no_grad():
                feats = feature_extractor(x)
            loss = loss_fn(clf(feats), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    clf.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = _add_gaussian_noise(x.to(device), noise_std)
            feats = feature_extractor(x)
            pred = clf(feats).argmax(1).cpu()
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def plot_noise_linear_eval(results, args):
    """results: list of (noise_std, acc). Saves Results/NoiseEval_<baseline>_<eval_ds>.png"""
    if not results:
        return None
    import matplotlib.pyplot as plt

    os.makedirs("Results", exist_ok=True)
    sigmas = [s for s, _ in results]
    accs = [a for _, a in results]
    baseline = getattr(args, "baseline_name", "run")
    eval_ds = str(getattr(args, "eval_dataset", "eval")).lower().replace("/", "-")

    plt.figure(figsize=(6, 4))
    plt.plot(sigmas, accs, marker="o")
    plt.xlabel("Gaussian noise std (σ)")
    plt.ylabel("Linear probe accuracy")
    plt.title(f"Noise robustness — {baseline} / {eval_ds}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join("Results", f"NoiseEval_{baseline}_{eval_ds}.png")
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def live_eval_noise_sweep(context_encoder, args, extract_features):
    """
    Linear probe under Gaussian input noise for σ in _EVAL_NOISE_LEVELS.
    Intended for --eval_only 1 with --eval_noise 1.
    """
    import torch.nn as nn

    class _FE(nn.Module):
        def forward(self, x):
            return extract_features(context_encoder, x)

    was_training = context_encoder.training
    context_encoder.eval()
    fe = _FE().to(args.device)

    results = []
    print(
        f"Eval-noise: linear probe on {args.eval_dataset} "
        f"σ∈{_EVAL_NOISE_LEVELS}"
    )
    for sigma in _EVAL_NOISE_LEVELS:
        acc = float(linear_probe_eval(fe, args, noise_std=sigma))
        results.append((sigma, acc))
        print(f"  σ={sigma:.2f}  linear acc={acc:.4f}")

    path = plot_noise_linear_eval(results, args)
    if path:
        print(f"Saved noise-eval plot: {path}")

    if was_training:
        context_encoder.train()
    return results


@torch.no_grad()
def build_feature_bank(feature_extractor, dataloader, device):
    features, labels = [], []
    feature_extractor.eval()
    for x, y in dataloader:
        z = feature_extractor(x.to(device))
        z = torch.nn.functional.normalize(z, dim=1)
        features.append(z.cpu())
        labels.append(y)
    return torch.cat(features), torch.cat(labels)


@torch.no_grad()
def knn_eval(feature_extractor, args, temperature=0.07):
    import Datasets

    device = args.device
    num_classes = Datasets.eval_num_classes(args)
    train_loader, test_loader = get_eval_dataloaders(args)
    feat_train, labels_train = build_feature_bank(feature_extractor, train_loader, device)

    correct = total = 0
    feature_extractor.eval()
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        z = torch.nn.functional.normalize(feature_extractor(x), dim=1).cpu()
        sim = z @ feat_train.T
        topk_sim, topk_idx = sim.topk(args.K, dim=1)
        topk_labels = labels_train[topk_idx]
        weights = torch.exp(topk_sim / temperature)
        scores = torch.zeros(z.size(0), num_classes)
        for c in range(num_classes):
            scores[:, c] = (weights * (topk_labels == c)).sum(dim=1)
        correct += (scores.argmax(1) == y.cpu()).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


def run_downstream_eval(feature_extractor, args):
    if args.evaluation.lower() == "linear":
        return linear_probe_eval(feature_extractor, args)
    if args.evaluation.lower() == "knn":
        return knn_eval(feature_extractor, args)
    raise ValueError(f"Unknown evaluation: {args.evaluation}")


def plot_eval_progress(eval_history, baseline_name, evaluation, results_dir='Results'):
    """eval_history: list of (epoch, accuracy). Saves Results/Eval_<baseline>.png"""
    if not eval_history:
        return None
    import matplotlib.pyplot as plt

    os.makedirs(results_dir, exist_ok=True)
    epochs = [e for e, _ in eval_history]
    accs = [a for _, a in eval_history]
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, accs, marker="o")
    plt.xlabel("Pretrain epoch")
    plt.ylabel(f"{evaluation} accuracy")
    plt.title(f"Downstream eval — {baseline_name}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(results_dir, f"Eval_{baseline_name}.png")
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def live_eval(context_encoder, args, extract_features):
    """Run linear/kNN on the current encoder without mutating requires_grad flags."""
    import torch.nn as nn

    class _FE(nn.Module):
        def forward(self, x):
            return extract_features(context_encoder, x)

    was_training = context_encoder.training
    context_encoder.eval()
    fe = _FE().to(args.device)
    acc = run_downstream_eval(fe, args)
    if was_training:
        context_encoder.train()
    return acc


def _resolve_eval_checkpoint_path(path, checkpoint_base=None):
    """Resolve a user-provided init path to an existing checkpoint file."""
    path = os.path.expanduser(str(path).strip())
    candidates = [path, os.path.normpath(path)]
    if checkpoint_base:
        rel = path if os.path.isabs(path) else os.path.join(checkpoint_base, path)
        candidates.append(os.path.normpath(rel))
        if os.path.isdir(rel):
            candidates.append(os.path.join(os.path.normpath(rel), "last.ckpt"))
            candidates.append(os.path.join(os.path.normpath(rel), "best.ckpt"))
    seen = set()
    for c in candidates:
        if not c or c in seen:
            continue
        seen.add(c)
        if os.path.isfile(c):
            return c
        if os.path.isdir(c):
            for name in ("best.ckpt", "last.ckpt"):
                f = os.path.join(c, name)
                if os.path.isfile(f):
                    return f
    raise FileNotFoundError(
        f"Eval-only checkpoint not found: {path!r} "
        f"(tried: {', '.join(sorted(seen))})"
    )


def maybe_eval_only(encoder, args, extract_features, models, checkpoint_base):
    """
    If --eval_only=1, use weights from --initialization, run downstream eval + t-SNE,
    and return True so the caller can skip training.

    With --eval_noise=1, also runs linear probe under Gaussian input noise
    σ∈{0,0.2,0.4,0.8,1.0,1.2,1.4,1.6,1.8,1.9} and saves Results/NoiseEval_*.png.

    Expected usage in each baseline runner, right after model construction /
    --initialization load:

        if Utils.maybe_eval_only(encoder, args, extract_fn, models, ckpt_base):
            return
    """
    if int(getattr(args, "eval_only", 0)) != 1:
        return False

    init = getattr(args, "initialization", None)
    loaded_init = bool(getattr(args, "_loaded_from_initialization", False))

    if not loaded_init:
        if not init:
            raise ValueError(
                "--eval_only=1 requires --initialization "
                "(a trained checkpoint to evaluate)."
            )
        ckpt_path = _resolve_eval_checkpoint_path(init, checkpoint_base)
        load_ckpt(ckpt_path, models, optimizer=None)
        print(f"Eval-only: loaded {ckpt_path}")
    else:
        print(f"Eval-only: using --initialization ({init})")

    label = (
        "linear probe"
        if str(args.evaluation).lower() == "linear"
        else f"k-NN (k={args.K})"
    )
    print(f"Eval-only: {args.eval_dataset} | {label}")
    acc = live_eval(encoder, args, extract_features)
    print(f"Eval-only: {label} acc={acc:.4f}")

    if int(getattr(args, "eval_noise", 0)) == 1:
        live_eval_noise_sweep(encoder, args, extract_features)

    tsne_path = live_eval_tsne(encoder, args, extract_features, epoch=0)
    if tsne_path:
        print(f"Saved t-SNE: {tsne_path}")
    return True


_CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


@torch.no_grad()
def plot_eval_tsne(feature_extractor, args, epoch, max_points=5000):
    """t-SNE of eval-dataset embeddings → Results/tSNE_<baseline>_<eval_dataset>.png"""
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    was_training = feature_extractor.training
    feature_extractor.eval()
    _, test_loader = get_eval_dataloaders(args)
    feats, labels = build_feature_bank(feature_extractor, test_loader, args.device)
    if was_training:
        feature_extractor.train()

    n = feats.size(0)
    if n > max_points:
        g = torch.Generator().manual_seed(42)
        idx = torch.randperm(n, generator=g)[:max_points]
        feats, labels = feats[idx], labels[idx]

    z2d = TSNE(
        n_components=2,
        perplexity=min(30, max(5, feats.size(0) // 10)),
        init="pca",
        learning_rate="auto",
        random_state=42,
    ).fit_transform(feats.numpy())

    baseline = getattr(args, "baseline_name", "run")
    eval_ds = str(getattr(args, "eval_dataset", "eval")).lower()
    results_dir = getattr(args, "results_dir", None) or "Results"
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"tSNE_{baseline}_{eval_ds}.png")

    labels_np = labels.numpy()
    plt.figure(figsize=(7, 6))
    num_classes = int(labels_np.max()) + 1 if labels_np.size else 0
    cmap = plt.get_cmap("tab20" if num_classes > 10 else "tab10")
    for c in range(num_classes):
        m = labels_np == c
        if eval_ds in {"cifar10", "cifar-10"} and c < len(_CIFAR10_CLASSES):
            name = _CIFAR10_CLASSES[c]
        else:
            name = str(c)
        # Avoid a 100-entry legend for CIFAR-100
        label = name if num_classes <= 10 else None
        plt.scatter(z2d[m, 0], z2d[m, 1], s=6, alpha=0.65, color=cmap(c % 20), label=label, linewidths=0)
    if num_classes <= 10:
        plt.legend(markerscale=2, fontsize=8, loc="best", frameon=True)
    plt.title(f"t-SNE — {baseline} | {eval_ds} | epoch {int(epoch)}")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def live_eval_tsne(context_encoder, args, extract_features, epoch):
    """Build the same feature extractor as live_eval and save a t-SNE plot."""
    import torch.nn as nn

    class _FE(nn.Module):
        def forward(self, x):
            return extract_features(context_encoder, x)

    was_training = context_encoder.training
    context_encoder.eval()
    fe = _FE().to(args.device)
    path = plot_eval_tsne(fe, args, epoch)
    if was_training:
        context_encoder.train()
    return path


##############################################################################################
##############################################################################################
# Checkpoint 


import os, json, hashlib, time, tempfile, random
import torch


def prepare_checkpoint_state(models, opt, base_path, args):
    cfg_h = cfg_hash(args)

    run_dir = prepare_run_dir(base=base_path, cfg_hash=cfg_h, args=args)

    start_epoch, global_step, best_acc = init_training_state(
        models=models,
        target_encoder=models["target"],
        args=args,
    )

    return {
        "run_dir": run_dir,
        "start_epoch": start_epoch,
        "global_step": global_step,
        "best_acc": best_acc,
        # Alias kept for older callers; value is best eval accuracy (higher is better).
        "best_loss": best_acc,
        "config_hash": cfg_h,
    }


def cfg_hash(args): 
    """ 
    This function generates a short, deterministic fingerprint of the configuration.
    Identical configs produce the same hash; changing any hyperparameter produces a different hash.
    """
    d = {k: getattr(args, k) for k in sorted(vars(args))}
    s = json.dumps(d, sort_keys=True, separators=(',', ':'))
    return hashlib.sha1(s.encode()).hexdigest()[:8]


def prepare_run_dir(base, cfg_hash, args):
    """Create a checkpoint folder for this training run.

    If args.ckpt_run_name is set (e.g. STL_beta_0.01), use {base}/{ckpt_run_name}.
    Otherwise use {base}/run_{YYYYMMDD_HHMMSS}_{cfg_hash}.
    """
    run_name = getattr(args, "ckpt_run_name", None)
    if run_name:
        path = os.path.join(base, str(run_name))
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(base, f"run_{ts}_{cfg_hash}")
    os.makedirs(path, exist_ok=True)
    write_meta(path, cfg_hash, args)
    return path


def init_training_state(models, target_encoder, args=None):
    """
    Start training from scratch: epoch/step 0 and sync EMA target from context
    unless the baseline uses a fully trainable target or weights came from --initialization.
    """
    # True for baselines with a fully trainable target (no EMA / no context→target copy).
    ib_reg = args is not None and getattr(args, "ib_jepa_reg", False)
    loaded_from_init = args is not None and getattr(args, "_loaded_from_initialization", False)
    if not ib_reg and not loaded_from_init:
        target_encoder.load_state_dict(models["context"].state_dict(), strict=True)

    if ib_reg:
        for p in target_encoder.parameters():
            p.requires_grad = True
    else:
        for p in target_encoder.parameters():
            p.requires_grad = False

    return 0, 0, float("-inf")

    
def load_ckpt(path, models, optimizer=None, strict=False):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if 'models' not in ckpt:
        raise KeyError(f"Checkpoint {path} has no 'models' entry.")

    ckpt_models = ckpt['models']
    loaded_any = False
    for name, m in models.items():
        if name not in ckpt_models:
            msg = f"[checkpoint] '{name}' not found in {path}; keeping current initialization."
            if strict:
                raise KeyError(msg)
            print(msg)
            continue

        stats = _safe_load_into(m, ckpt_models[name])
        loaded_any = loaded_any or stats["loaded"] > 0
        if stats["skipped"] or stats["missing"]:
            print(
                f"[checkpoint] Loaded {stats['loaded']} tensors into '{name}', "
                f"skipped {stats['skipped']} incompatible checkpoint tensors, "
                f"{stats['missing']} current tensors left initialized."
            )

    extra = sorted(set(ckpt_models) - set(models))
    if extra:
        print(f"[checkpoint] Ignoring extra checkpoint modules: {extra}")

    if not loaded_any:
        raise RuntimeError(f"No compatible model tensors were loaded from checkpoint: {path}")

    if optimizer and 'optimizer' in ckpt:
        try:
            optimizer.load_state_dict(ckpt['optimizer'])
        except (ValueError, RuntimeError) as e:
            if strict:
                raise RuntimeError("Optimizer state mismatch during checkpoint load") from e
            print("[checkpoint] Optimizer state is incompatible; using a fresh optimizer.")
    _rng_load(ckpt.get('rng', {}))
    best = ckpt.get('best_metric', float('-inf'))
    # Older runs stored train loss as best_metric (>1). Treat those as unknown accuracy.
    if best is None or (isinstance(best, (int, float)) and best > 1.0):
        best = float('-inf')
    return (
        ckpt.get('epoch', 0),
        ckpt.get('global_step', 0),
        best,
    )


def save_epoch(run_dir, models, optimizer, epoch, step, best):
    save_ckpt(
        os.path.join(run_dir, "last.ckpt"),
        models,
        optimizer,
        epoch + 1,
        step,
        best
    )


def save_ckpt(path, models, optimizer, epoch, step, best):
    state = {
        'models': {k: v.state_dict() for k, v in models.items()},
        'optimizer': optimizer.state_dict() if optimizer else {},
        'epoch': epoch,
        'global_step': step,
        'best_metric': best,
        'rng': _rng_state()
    }
    _atomic_save(state, path)

def _atomic_save(obj, path):
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d)
    os.close(fd)
    torch.save(obj, tmp)
    os.replace(tmp, path)

def save_best(run_dir, models, optimizer, epoch, step, best):
    save_ckpt(
        os.path.join(run_dir, "best.ckpt"),
        models,
        optimizer,
        epoch + 1,
        step,
        best
    )

def _rng_state():
    return {
        'py': random.getstate(),
        'np': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }

def _rng_load(s):
    try:
        random.setstate(s['py'])
        np.random.set_state(s['np'])
        torch.set_rng_state(s['torch'])
        if torch.cuda.is_available() and s['cuda'] is not None:
            torch.cuda.set_rng_state_all(s['cuda'])
    except:
        pass

def _safe_load_into(model, state_dict):
    model_sd = model.state_dict()
    filtered = {}
    for k, v in state_dict.items():
        if k not in model_sd:
            continue
        # skip lazily initialized or shape-mismatched params (e.g., pos_embed)
        if model_sd[k].numel() == 0 or model_sd[k].shape != v.shape:
            continue
        filtered[k] = v
    # load what matches; ignore the rest
    model.load_state_dict(filtered, strict=False)
    return {
        "loaded": len(filtered),
        "skipped": len(state_dict) - len(filtered),
        "missing": len(model_sd) - len(filtered),
    }

def write_meta(run_dir, cfg_hash, args, primary_metric='train_loss'):
    meta = {
        "config_hash": cfg_hash,
        "created": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        "paths": {
            "best": "best.ckpt",
            "last": "last.ckpt",
        },
        "primary_metric": primary_metric,
        "config": {
            k: getattr(args, k)
            for k in sorted(vars(args))
        },
    }
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


##############################################################################################
##############################################################################################
