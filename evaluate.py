"""
evaluate.py — Linear probing + Rank-1 + EER evaluation for JEPA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_curve


@torch.no_grad()
def extract_features(feature_extractor, loader, device):
    """Extract features and labels from a dataloader."""
    feats, labels = [], []
    feature_extractor.eval()
    for x, y in loader:
        z = feature_extractor(x.to(device))
        z = F.normalize(z, dim=-1)
        feats.append(z.cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def compute_eer(genuine_scores, impostor_scores):
    """Compute Equal Error Rate."""
    labels = np.concatenate([np.ones(len(genuine_scores)),
                             np.zeros(len(impostor_scores))])
    scores = np.concatenate([genuine_scores, impostor_scores])
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2) * 100


def evaluate_rank1_eer(gallery_feats, gallery_labels,
                       probe_feats, probe_labels):
    """
    Compute Rank-1 accuracy and EER from gallery/probe features.
    """
    sim = probe_feats @ gallery_feats.T  # (n_probe, n_gallery)

    # Rank-1
    top_idx = sim.argmax(dim=1)
    predicted = gallery_labels[top_idx]
    rank1 = (predicted == probe_labels).float().mean().item() * 100

    # EER
    genuine, impostor = [], []
    for i in range(len(probe_labels)):
        pid = probe_labels[i].item()
        sims = sim[i].numpy()
        glabs = gallery_labels.numpy()
        gen_mask = glabs == pid
        imp_mask = glabs != pid
        if gen_mask.any():
            genuine.extend(sims[gen_mask].tolist())
        if imp_mask.any():
            impostor.extend(sims[imp_mask].tolist())

    eer = compute_eer(np.array(genuine), np.array(impostor))

    return {
        "rank1": rank1,
        "eer": eer,
        "n_gallery": len(gallery_labels),
        "n_probe": len(probe_labels),
    }


def linear_probe(feature_extractor, eval_loader, n_classes,
                 lr=1e-3, epochs=50, device="cuda", train_ratio=0.8,
                 seed=2025):
    """
    Train a linear classifier on frozen features from the eval set.
    Splits eval set 80/20 internally for LP train/test.
    Returns: test accuracy %.
    """
    # Extract all features
    all_feats, all_labels = extract_features(feature_extractor, eval_loader,
                                              device)
    N = len(all_feats)
    feat_dim = all_feats.shape[-1]

    # Split eval set for LP
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=rng)
    n_train = int(N * train_ratio)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    train_feats = all_feats[train_idx].to(device)
    train_labels = all_labels[train_idx].to(device)
    test_feats = all_feats[test_idx].to(device)
    test_labels = all_labels[test_idx].to(device)

    if len(test_idx) == 0:
        return 0.0

    clf = nn.Linear(feat_dim, n_classes).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # Train on LP train split
    bs = min(64, n_train)
    for ep in range(epochs):
        clf.train()
        for i in range(0, n_train, bs):
            batch_f = train_feats[i:i+bs]
            batch_l = train_labels[i:i+bs]
            logits = clf(batch_f)
            loss = loss_fn(logits, batch_l)
            opt.zero_grad()
            loss.backward()
            opt.step()

    # Test on LP test split
    clf.eval()
    with torch.no_grad():
        pred = clf(test_feats).argmax(1)
        acc = (pred == test_labels).float().mean().item()

    return acc * 100.0


def run_full_eval(feature_extractor, eval_dict, n_classes, cfg, tag=""):
    """
    Run all evaluations: linear probe + Rank-1 + EER on each eval set.
    LP trains on the eval set itself (80/20 internal split).
    """
    device = cfg.device
    results = {}

    for name, ev in eval_dict.items():
        print(f"    {tag}Evaluating '{name}' "
              f"({ev['n_samples']} samples, {ev['n_ids']} IDs)...")

        # Linear probe (internal 80/20 split of eval set)
        lp_acc = linear_probe(
            feature_extractor, ev["loader"], n_classes,
            lr=cfg.eval_lp_lr, epochs=cfg.eval_lp_epochs,
            device=device, seed=cfg.seed)

        # Rank-1 + EER
        gal_feats, gal_labels = extract_features(
            feature_extractor, ev["gallery_loader"], device)
        prb_feats, prb_labels = extract_features(
            feature_extractor, ev["probe_loader"], device)
        ver = evaluate_rank1_eer(gal_feats, gal_labels,
                                 prb_feats, prb_labels)

        results[name] = {
            "lp_acc": lp_acc,
            "rank1": ver["rank1"],
            "eer": ver["eer"],
            "n_gallery": ver["n_gallery"],
            "n_probe": ver["n_probe"],
        }
        print(f"      LP: {lp_acc:.2f}% | "
              f"R1: {ver['rank1']:.2f}% | "
              f"EER: {ver['eer']:.2f}% | "
              f"Gal: {ver['n_gallery']} Prb: {ver['n_probe']}")

    return results
