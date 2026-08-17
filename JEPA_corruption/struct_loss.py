"""Structural loss variants + task-gradient conflict diagnostic."""
import torch
import torch.nn.functional as F


def structure_loss(pred, tgt, kind="cosine", temperature=0.1, max_n=4096):
    """pred, tgt: (..., D). Returns (loss, stats_dict)."""
    p = F.normalize(pred.reshape(-1, pred.size(-1)), dim=-1)
    t = F.normalize(tgt.reshape(-1, tgt.size(-1)), dim=-1)

    if kind == "smooth_l1":
        loss = F.smooth_l1_loss(p, t)
        with torch.no_grad():
            cos = (p * t).sum(-1).mean().item()
        return loss, {"cos": cos, "top1": float("nan"), "chance": float("nan")}

    if kind == "cosine":
        cos_v = (p * t).sum(-1)
        loss = (1.0 - cos_v).mean()
        with torch.no_grad():
            perm = torch.randperm(t.size(0), device=t.device)
            chance = (p * t[perm]).sum(-1).mean().item()
        return loss, {"cos": cos_v.mean().item(), "top1": float("nan"),
                      "chance": chance}

    if kind == "infonce":
        n = p.size(0)
        if n > max_n:
            idx = torch.randperm(n, device=p.device)[:max_n]
            p, t = p[idx], t[idx]
        logits = p @ t.T / temperature
        labels = torch.arange(p.size(0), device=p.device)
        loss = F.cross_entropy(logits, labels)
        with torch.no_grad():
            top1 = (logits.argmax(1) == labels).float().mean().item()
            cos = (p * t).sum(-1).mean().item()
        return loss, {"cos": cos, "top1": top1, "chance": 1.0 / p.size(0)}

    raise ValueError(f"unknown struct_loss: {kind}")


def grad_conflict_cosine(loss_a, loss_b, params):
    """Cosine between the two tasks' gradients on shared params.
    >0 complementary, ~0 orthogonal, <0 conflicting."""
    params = [p for p in params if p.requires_grad]
    ga = torch.autograd.grad(loss_a, params, retain_graph=True,
                             allow_unused=True)
    gb = torch.autograd.grad(loss_b, params, retain_graph=True,
                             allow_unused=True)
    fa, fb = [], []
    for a, b in zip(ga, gb):
        if a is None or b is None:
            continue
        fa.append(a.reshape(-1))
        fb.append(b.reshape(-1))
    if not fa:
        return float("nan")
    fa, fb = torch.cat(fa), torch.cat(fb)
    return float(F.cosine_similarity(fa.unsqueeze(0), fb.unsqueeze(0)).item())