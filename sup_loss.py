"""sup_loss.py — supervised identity losses over pooled context embeddings."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def supcon_loss(z, labels, temperature=0.1):
    """Supervised contrastive (Khosla et al.). z: (B, D), labels: (B,).
    Returns (loss, stats). Prototype-free -> extends to unseen identities."""
    z = F.normalize(z, dim=-1)
    B = z.size(0)
    eye = torch.eye(B, dtype=torch.bool, device=z.device)

    sim = (z @ z.T) / temperature
    sim = sim.masked_fill(eye, -1e9)                    # exclude self
    pos_mask = (labels[:, None] == labels[None, :]) & ~eye

    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    n_pos = pos_mask.sum(1)
    valid = n_pos > 0

    stats = {"pos_per_anchor": float(n_pos.float().mean()),
             "frac_anchors_with_pos": float(valid.float().mean())}

    if not valid.any():                                  # no positives in batch
        return z.sum() * 0.0, stats

    loss = -(log_prob * pos_mask).sum(1)[valid] / n_pos[valid]
    return loss.mean(), stats


class ArcFaceHead(nn.Module):
    """Additive angular margin softmax. Learns n_classes prototypes."""

    def __init__(self, dim, n_classes, scale=30.0, margin=0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_classes, dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale, self.margin, self.n_classes = scale, margin, n_classes

    def forward(self, z, labels):
        cos = F.normalize(z, dim=-1) @ F.normalize(self.weight, dim=-1).T
        cos = cos.clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cos)
        target = F.one_hot(labels, self.n_classes).bool()
        logits = torch.where(target, torch.cos(theta + self.margin), cos) * self.scale
        loss = F.cross_entropy(logits, labels)
        with torch.no_grad():
            acc = (cos.argmax(1) == labels).float().mean().item()
        return loss, {"acc": acc}


class CEHead(nn.Module):
    """Plain linear classifier + cross-entropy. Baseline arm only."""

    def __init__(self, dim, n_classes, label_smoothing=0.0):
        super().__init__()
        self.fc = nn.Linear(dim, n_classes)
        self.ls = label_smoothing

    def forward(self, z, labels):
        logits = self.fc(z)
        loss = F.cross_entropy(logits, labels, label_smoothing=self.ls)
        with torch.no_grad():
            acc = (logits.argmax(1) == labels).float().mean().item()
        return loss, {"acc": acc}


def build_sup_head(kind, dim, n_classes, cfg):
    """SupCon needs no parameters; ArcFace/CE do."""
    if kind == "supcon":
        return None
    if kind == "arcface":
        return ArcFaceHead(dim, n_classes,
                           scale=cfg.arcface_scale, margin=cfg.arcface_margin)
    if kind == "ce":
        return CEHead(dim, n_classes, label_smoothing=cfg.label_smoothing)
    raise ValueError(f"unknown sup_loss: {kind}")