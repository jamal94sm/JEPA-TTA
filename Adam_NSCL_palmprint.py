"""
Adam_NSCL_palmprint.py — Adam-NSCL (CVPR'21) applied to palmprint domains.

Faithful port of "Training Networks in Null Space of Feature Covariance for
Continual Learning" (Wang et al.), adapted from
https://github.com/ShipengWang/Adam-NSCL to a 2-task palmprint setting.

═══════════════════════════════════════════════════════════════════════════
  WHAT THIS IS
═══════════════════════════════════════════════════════════════════════════
  SUPERVISED multi-task / continual learning (NOT test-time adaptation).
  Both tasks are trained with labels and cross-entropy. The question is:
  can the network learn task 2 without destroying its task-1 representation?

  Task 1 (source)  ->  train normally, then build per-layer null-space
                       projectors from the layer INPUT covariances.
  Task 2 (target)  ->  train with the Adam update right-multiplied by those
                       projectors, so the update lives in the null space of
                       task-1 features.

═══════════════════════════════════════════════════════════════════════════
  MODES
═══════════════════════════════════════════════════════════════════════════
  intra : CASIA-MS  WHT  ->  CASIA-MS  700     (same identities, new spectrum)
  inter : CASIA-MS  all  ->  XJTU-UP          (disjoint identities, new dataset)

═══════════════════════════════════════════════════════════════════════════
  ARMS  (all branch from the SAME trained task-1 model, so comparison is fair)
═══════════════════════════════════════════════════════════════════════════
  frozen    no task-2 training                  -> zero plasticity, zero forgetting
  finetune  plain Adam, no protection           -> max plasticity, max forgetting
  ewc       plain Adam + EWC on BN              -> regularisation baseline
  nscl      projected Adam + EWC on BN          -> Adam-NSCL (the method)

═══════════════════════════════════════════════════════════════════════════
  EVALUATION
═══════════════════════════════════════════════════════════════════════════
  20% of each task is held out; that held-out set is split per identity into
  gallery/probe. Metrics are EER and Rank-1 computed on BACKBONE FEATURES
  (512-d, cosine similarity) — the classification heads are training
  scaffolding only, which is what makes the two modes comparable even though
  their label spaces differ.

USAGE
-----
python Adam_NSCL_palmprint.py --mode intra \
    --data_dir /home/pai-ng/Jamal/CASIA-MS-ROI \
    --task1_spectrum WHT --task2_spectrum 700

python Adam_NSCL_palmprint.py --mode inter \
    --data_dir /home/pai-ng/Jamal/CASIA-MS-ROI \
    --xjtu_root /home/pai-ng/Jamal/XJTU-UP
"""

import os
import re
import json
import copy
import math
import random
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import scan_dataset, build_id_map, split_gallery_probe, CASIADataset
try:
    from dataset import scan_xjtu
except ImportError:
    scan_xjtu = None
from evaluate import evaluate_rank1_eer


# ══════════════════════════════════════════════════════════════════════
#  Args
# ══════════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser("Adam-NSCL for palmprint (2 tasks)")
    p.add_argument("--mode", default="intra", choices=["intra", "inter"])
    p.add_argument("--data_dir", required=True)
    p.add_argument("--xjtu_root", default="/home/pai-ng/Jamal/XJTU-UP")
    p.add_argument("--task1_spectrum", default="WHT")
    p.add_argument("--task2_spectrum", default="700")
    p.add_argument("--test_ratio", type=float, default=0.2)
    p.add_argument("--gallery_ratio", type=float, default=0.5)
    # model
    p.add_argument("--img_size", type=int, default=112)
    p.add_argument("--in_channels", type=int, default=3)
    # optimisation (defaults from scripts_svd/adamnscl.sh)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--schedule", type=int, nargs="+", default=[30, 60, 80])
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--model_lr", type=float, default=1e-4, help="task-1 backbone")
    p.add_argument("--svd_lr", type=float, default=5e-5, help="projected backbone")
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--bn_lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=5e-5)
    p.add_argument("--batch_size", type=int, default=32)
    # NSCL / EWC
    p.add_argument("--svd_thres", type=float, default=10.0,
                   help="null space = eigenvalues <= thres * smallest")
    p.add_argument("--reg_coef", type=float, default=100.0, help="EWC on BN")
    p.add_argument("--cov_batches", type=int, default=100,
                   help="batches used to accumulate per-layer covariance")
    # misc
    p.add_argument("--augment", action="store_true", default=True)
    p.add_argument("--no_augment", dest="augment", action="store_false")
    p.add_argument("--arms", nargs="+",
                   default=["frozen", "finetune", "ewc", "nscl"])
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--out_dir", default="./output_nscl")
    return p.parse_args()


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ══════════════════════════════════════════════════════════════════════
#  MODEL — PreActResNet18, ported from Adam-NSCL/models/resnet.py
#  Multi-head: one classifier per task. `last` naming is kept because the
#  reference excludes the head from projection via re.match('last', name).
# ══════════════════════════════════════════════════════════════════════

def conv3x3(cin, cout, stride=1):
    return nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)


class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, 1,
                          stride=stride, bias=False))

    def forward(self, x):
        x = self.bn1(x)
        out = F.relu(x)
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.conv2(F.relu(out))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    """PreActResNet18 with per-task heads and a feature() entry point."""

    def __init__(self, num_blocks=(2, 2, 2, 2), head_dims=(10,), in_channels=3):
        super().__init__()
        self.in_planes = 64
        self.feat_dim = 512 * PreActBlock.expansion
        self.conv1 = conv3x3(in_channels, 64)
        self.stage1 = self._make_layer(64, num_blocks[0], 1)
        self.stage2 = self._make_layer(128, num_blocks[1], 2)
        self.stage3 = self._make_layer(256, num_blocks[2], 2)
        self.stage4 = self._make_layer(512, num_blocks[3], 2)
        self.bn_last = nn.BatchNorm2d(self.feat_dim)
        # heads are named last0, last1, ... so re.match('last', n) catches them
        for t, d in enumerate(head_dims):
            setattr(self, f"last{t}", nn.Linear(self.feat_dim, d))

    def _make_layer(self, planes, n, stride):
        strides = [stride] + [1] * (n - 1)
        layers = []
        for s in strides:
            layers.append(PreActBlock(self.in_planes, planes, s))
            self.in_planes = planes * PreActBlock.expansion
        return nn.Sequential(*layers)

    def feature(self, x):
        """512-d embedding used for ALL gallery/probe evaluation."""
        out = self.conv1(x)
        out = self.stage1(out); out = self.stage2(out)
        out = self.stage3(out); out = self.stage4(out)
        out = F.relu(self.bn_last(out))
        out = F.adaptive_avg_pool2d(out, 1)
        return out.view(out.size(0), -1)

    def forward(self, x, task=0):
        return getattr(self, f"last{task}")(self.feature(x))


# ══════════════════════════════════════════════════════════════════════
#  PER-LAYER COVARIANCE  (svd_agent/svd_based.py: compute_cov / update_cov)
#  Linear : cov of the layer input          [in, in]
#  Conv2d : cov of im2col patches           [Cin*k*k, Cin*k*k]
# ══════════════════════════════════════════════════════════════════════

class CovarianceAccumulator:
    def __init__(self, model):
        self.fea_in = {}
        self.handles = []
        # exactly the reference's selection: modules with weight, excluding head
        self.modules = [m for n, m in model.named_modules()
                        if isinstance(m, (nn.Conv2d, nn.Linear))
                        and not bool(re.match("last", n))]

    def _hook(self, module, fea_in, fea_out):
        x = fea_in[0].detach()
        if isinstance(module, nn.Linear):
            X = x.reshape(-1, x.size(-1))
        else:                                   # Conv2d -> im2col patch space
            X = F.unfold(x, module.kernel_size, module.dilation,
                         module.padding, module.stride)        # [B, Ckk, L]
            X = X.permute(0, 2, 1).reshape(-1, X.size(1))      # [B*L, Ckk]
        cov = X.T @ X
        k = module.weight
        self.fea_in[k] = cov.double() if k not in self.fea_in \
            else self.fea_in[k] + cov.double()

    def attach(self):
        self.handles = [m.register_forward_hook(self._hook) for m in self.modules]

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []


@torch.no_grad()
def accumulate_covariance(model, loader, args, prev=None):
    """Run the task data through the frozen model, collecting per-layer
       input covariances. `prev` accumulates across tasks (reference behaviour)."""
    model.eval()
    acc = CovarianceAccumulator(model)
    acc.attach()
    for i, (x, _y) in enumerate(loader):
        if i >= args.cov_batches:
            break
        model.feature(x.to(args.device))
    acc.detach()
    if prev is not None:
        for k, v in prev.items():
            acc.fea_in[k] = acc.fea_in.get(k, torch.zeros_like(v)) + v
    return acc.fea_in


def build_transforms(fea_in, thres, device):
    """P = U_null U_null^T from the SMALLEST eigen-directions, then normalise
       by ||P||_F   (optim/adam_svd.py: get_eigens + get_transforms)."""
    transforms, info = {}, []
    for p, cov in fea_in.items():
        U, S, _ = torch.linalg.svd(cov.cpu(), full_matrices=False)
        # torch.svd returns S descending -> S[-1] is the smallest
        keep = S <= (S[-1] * thres)
        if keep.sum() == 0:                     # degenerate: keep nothing
            keep[-1] = True
        basis = U[:, keep].float()
        P = basis @ basis.T
        P = P / P.norm()                        # Frobenius normalisation
        transforms[p] = P.to(device)
        info.append((tuple(cov.shape), int(keep.sum()), cov.shape[0]))
    return transforms, info


# ══════════════════════════════════════════════════════════════════════
#  PROJECTED ADAM  (optim/adam_svd.py) — project the UPDATE, not the grad
# ══════════════════════════════════════════════════════════════════════

class AdamSVD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, svd=False):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, svd=svd)
        super().__init__(params, defaults)
        self.transforms = {}

    def set_transforms(self, transforms):
        self.transforms = transforms

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            svd = group["svd"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                st = self.state[p]
                if len(st) == 0:
                    st["step"] = 0
                    st["exp_avg"] = torch.zeros_like(p)
                    st["exp_avg_sq"] = torch.zeros_like(p)
                b1, b2 = group["betas"]
                st["step"] += 1
                if group["weight_decay"]:
                    grad = grad.add(p, alpha=group["weight_decay"])
                st["exp_avg"].mul_(b1).add_(grad, alpha=1 - b1)
                st["exp_avg_sq"].mul_(b2).addcmul_(grad, grad, value=1 - b2)
                denom = st["exp_avg_sq"].sqrt().add_(group["eps"])
                bc1 = 1 - b1 ** st["step"]
                bc2 = 1 - b2 ** st["step"]
                step_size = group["lr"] * math.sqrt(bc2) / bc1
                update = -step_size * st["exp_avg"] / denom
                if svd and p in self.transforms:
                    P = self.transforms[p]
                    shape = update.shape
                    upd = update.view(shape[0], -1)      # [out, in(*k*k)]
                    update = (upd @ P).view(shape)
                p.add_(update)


def make_optimizer(model, args, task, transforms=None):
    """Reference parameter grouping: backbone (svd) / heads / BN."""
    fea, head, bn = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if re.match("last", n):
            head.append(p)
        elif "bn" in n:
            bn.append(p)
        else:
            fea.append(p)
    lr_fea = args.model_lr if task == 0 else args.svd_lr
    groups = [
        {"params": fea, "lr": lr_fea, "svd": transforms is not None},
        {"params": head, "lr": args.head_lr, "svd": False},
        {"params": bn, "lr": args.bn_lr, "svd": False},
    ]
    opt = AdamSVD(groups, weight_decay=args.weight_decay)
    if transforms is not None:
        opt.set_transforms(transforms)
    return opt


# ══════════════════════════════════════════════════════════════════════
#  EWC ON BN  (svd_agent: calculate_importance / reg_loss)
# ══════════════════════════════════════════════════════════════════════

def bn_params(model):
    return {n: p for n, p in model.named_parameters() if "bn" in n}


def calculate_importance(model, loader, task, args):
    """Empirical Fisher: accumulated squared gradients of the BN params."""
    reg = bn_params(model)
    imp = {n: torch.zeros_like(p) for n, p in reg.items()}
    model.eval()
    n_bat = 0
    for i, (x, y) in enumerate(loader):
        if i >= args.cov_batches:
            break
        x, y = x.to(args.device), y.to(args.device)
        model.zero_grad()
        F.cross_entropy(model(x, task), y).backward()
        for n, p in reg.items():
            if p.grad is not None:
                imp[n] += p.grad.detach() ** 2
        n_bat += 1
    for n in imp:
        imp[n] /= max(n_bat, 1)
    model.zero_grad()
    return imp


def ewc_loss(model, importance, old_params):
    loss = 0.0
    for n, p in bn_params(model).items():
        if n in importance:
            loss = loss + (importance[n] * (p - old_params[n]) ** 2).sum()
    return loss


# ══════════════════════════════════════════════════════════════════════
#  DATA
# ══════════════════════════════════════════════════════════════════════

def split_train_test(samples, test_ratio, seed):
    """Per-identity split; each identity keeps >=2 test images so the
       gallery/probe split downstream is well defined."""
    by_id = defaultdict(list)
    for s in samples:
        by_id[s["identity"]].append(s)
    rng = random.Random(seed)
    train, test = [], []
    for ident in sorted(by_id):
        items = by_id[ident][:]
        rng.shuffle(items)
        n_te = max(2, int(round(len(items) * test_ratio)))
        n_te = min(n_te, len(items) - 1)
        test.extend(items[:n_te]); train.extend(items[n_te:])
    return train, test


def build_tasks(args):
    """Returns a list of two task dicts with their own id_map and splits."""
    casia = scan_dataset(args.data_dir)
    if args.mode == "intra":
        s1 = [s for s in casia if s["spectrum"] == args.task1_spectrum]
        s2 = [s for s in casia if s["spectrum"] == args.task2_spectrum]
        names = (f"CASIA-{args.task1_spectrum}", f"CASIA-{args.task2_spectrum}")
    else:
        if scan_xjtu is None:
            raise SystemExit("inter mode needs scan_xjtu() in dataset.py")
        s1 = casia
        s2 = scan_xjtu(args.xjtu_root)
        names = ("CASIA-MS(all)", "XJTU-UP")
    if not s1 or not s2:
        raise SystemExit("one of the tasks has no samples")

    tasks = []
    for t, (samples, name) in enumerate(zip((s1, s2), names)):
        id_map = build_id_map(samples)          # per-task label space
        tr, te = split_train_test(samples, args.test_ratio, args.seed)
        gal, prb = split_gallery_probe(te, id_map, args.gallery_ratio, args.seed)
        tasks.append({"t": t, "name": name, "id_map": id_map,
                      "n_cls": len(id_map), "train": tr, "test": te,
                      "gal": gal, "prb": prb})
    return tasks


def loader_of(samples, id_map, args, train=False):
    ds = CASIADataset(samples, id_map, args.img_size,
                      augment=(train and args.augment), aug_multiplier=1)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=train,
                      num_workers=args.num_workers, drop_last=train,
                      pin_memory=True)


# ══════════════════════════════════════════════════════════════════════
#  TRAIN / EVAL
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract(model, loader, device):
    model.eval()
    X, Y = [], []
    for x, y in loader:
        X.append(model.feature(x.to(device)).cpu()); Y.append(y)
    return torch.cat(X), torch.cat(Y)


def eval_task(model, task, args):
    gl = loader_of(task["gal"], task["id_map"], args)
    pl = loader_of(task["prb"], task["id_map"], args)
    gf, gy = extract(model, gl, args.device)
    pf, py = extract(model, pl, args.device)
    return evaluate_rank1_eer(F.normalize(gf, dim=-1), gy,
                              F.normalize(pf, dim=-1), py)


def train_task(model, task, args, transforms=None, ewc=None, tag=""):
    """One task of supervised training. `transforms` -> projected Adam,
       `ewc` = (importance, old_params) -> EWC penalty on BN."""
    t = task["t"]
    loader = loader_of(task["train"], task["id_map"], args, train=True)
    opt = make_optimizer(model, args, t, transforms)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, args.schedule, args.gamma)

    for ep in range(1, args.epochs + 1):
        model.train()
        tot, corr, seen, nb = 0.0, 0, 0, 0
        for x, y in loader:
            x, y = x.to(args.device), y.to(args.device)
            logits = model(x, t)
            loss = F.cross_entropy(logits, y)
            if ewc is not None:
                loss = loss + args.reg_coef * ewc_loss(model, ewc[0], ewc[1])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1
            corr += (logits.argmax(1) == y).sum().item(); seen += y.size(0)
        sched.step()
        if ep % 10 == 0 or ep == 1 or ep == args.epochs:
            print(f"      [{tag}] ep {ep:3d}/{args.epochs}  "
                  f"loss={tot/max(nb,1):.4f}  train_acc={100*corr/max(seen,1):.2f}%")
    return model


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    dev = args.device
    R = {"config": vars(args)}

    print(f"\n{'='*78}\n  ADAM-NSCL FOR PALMPRINT   mode={args.mode}\n{'='*78}")

    # ── data ──────────────────────────────────────────────────────────
    print(f"\n{'─'*78}\n  Tasks\n{'─'*78}")
    tasks = build_tasks(args)
    for t in tasks:
        print(f"  task {t['t']}  {t['name']:<18} ids={t['n_cls']:4d}  "
              f"train={len(t['train']):5d}  test={len(t['test']):5d} "
              f"(gal {len(t['gal'])} / prb {len(t['prb'])})")
    if args.mode == "inter":
        shared = set(tasks[0]["id_map"]) & set(tasks[1]["id_map"])
        print(f"  identity overlap between tasks: {len(shared)} (expected 0)")

    # ── model ─────────────────────────────────────────────────────────
    model = PreActResNet(head_dims=[t["n_cls"] for t in tasks],
                         in_channels=args.in_channels).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"\n  PreActResNet18: {n_par/1e6:.2f}M params  "
          f"feat_dim={model.feat_dim}  heads={[t['n_cls'] for t in tasks]}")
    print(f"  augmentation: {args.augment}")

    # ── TASK 1 ────────────────────────────────────────────────────────
    print(f"\n{'─'*78}\n  TASK 1  —  {tasks[0]['name']}  (plain Adam)\n{'─'*78}")
    train_task(model, tasks[0], args, tag="task1")
    r00 = eval_task(model, tasks[0], args)
    print(f"    task1 after task1:  EER={r00['eer']:6.2f}%  R1={r00['rank1']:6.2f}%")
    R["task1_after_task1"] = r00

    # ── projectors + EWC state from task 1 ────────────────────────────
    print(f"\n{'─'*78}\n  Building null-space projectors from task 1\n{'─'*78}")
    cov_loader = loader_of(tasks[0]["train"], tasks[0]["id_map"], args)
    fea_in = accumulate_covariance(model, cov_loader, args)
    transforms, info = build_transforms(fea_in, args.svd_thres, dev)
    print(f"  layers projected: {len(transforms)}")
    for (shape, n_null, dim) in info[:6]:
        print(f"    cov {str(shape):>14}  null dims {n_null:4d}/{dim:<4d}"
              f"  ({100*n_null/dim:.1f}% free)")
    if len(info) > 6:
        print(f"    ... {len(info)-6} more layers")
    mean_free = float(np.mean([n / d for _, n, d in info]))
    print(f"  mean free fraction across layers: {mean_free:.3f}")
    R["projector"] = {"n_layers": len(transforms), "mean_free_frac": mean_free,
                      "per_layer": [{"dim": d, "null": n} for _, n, d in info]}

    print(f"  computing EWC importance on BN ...")
    importance = calculate_importance(model, cov_loader, 0, args)
    old_bn = {n: p.detach().clone() for n, p in bn_params(model).items()}

    # ── TASK 2, one model copy per arm ────────────────────────────────
    results = {}
    for arm in args.arms:
        print(f"\n{'─'*78}\n  TASK 2  —  {tasks[1]['name']}   arm = {arm.upper()}"
              f"\n{'─'*78}")
        m = copy.deepcopy(model)
        if arm == "frozen":
            print("    (no task-2 training)")
        else:
            tr = transforms if arm == "nscl" else None
            ew = (importance, old_bn) if arm in ("ewc", "nscl") else None
            print(f"    projection={'ON' if tr else 'off'}   "
                  f"EWC={'ON' if ew else 'off'}   lr_backbone="
                  f"{args.svd_lr if tr else args.model_lr}")
            train_task(m, tasks[1], args, transforms=tr, ewc=ew, tag=arm)

        r1 = eval_task(m, tasks[1], args)        # plasticity
        r0 = eval_task(m, tasks[0], args)        # retention
        d_eer = r0["eer"] - r00["eer"]
        d_r1 = r0["rank1"] - r00["rank1"]
        print(f"    task2 (new)  EER={r1['eer']:6.2f}%  R1={r1['rank1']:6.2f}%")
        print(f"    task1 (old)  EER={r0['eer']:6.2f}% ({d_eer:+5.2f})  "
              f"R1={r0['rank1']:6.2f}% ({d_r1:+5.2f})")
        results[arm] = {"task2": r1, "task1": r0,
                        "forget_eer": d_eer, "forget_rank1": d_r1}

    # ── summary ───────────────────────────────────────────────────────
    print(f"\n{'='*78}\n  RESULTS   ({tasks[0]['name']}  ->  {tasks[1]['name']})"
          f"\n{'='*78}")
    hdr = (f"  {'arm':<12}{'task2 EER':>11}{'task2 R1':>10}"
           f"{'task1 EER':>11}{'task1 R1':>10}{'dEER':>8}{'dR1':>8}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    print(f"  {'(after t1)':<12}{'—':>11}{'—':>10}"
          f"{r00['eer']:11.2f}{r00['rank1']:10.2f}{'—':>8}{'—':>8}")
    for arm in args.arms:
        r = results[arm]
        print(f"  {arm:<12}{r['task2']['eer']:11.2f}{r['task2']['rank1']:10.2f}"
              f"{r['task1']['eer']:11.2f}{r['task1']['rank1']:10.2f}"
              f"{r['forget_eer']:+8.2f}{r['forget_rank1']:+8.2f}")
    print(f"\n  plasticity = task2 columns   |   retention = task1 dEER/dR1")
    print(f"  target behaviour: nscl should match FINETUNE on task2 while "
          f"keeping dEER/dR1 near FROZEN")

    R["results"] = results
    jp = os.path.join(args.out_dir, f"nscl_{args.mode}_"
                      f"{tasks[0]['name']}_{tasks[1]['name']}.json".replace("/", "-"))
    with open(jp, "w") as f:
        json.dump(R, f, indent=2, default=float)
    print(f"\n  saved {jp}\n{'='*78}\n")


if __name__ == "__main__":
    main()
