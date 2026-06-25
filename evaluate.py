"""
evaluate.py — Rank-1 + EER evaluation for JEPA.
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
    """Compute Rank-1 accuracy and EER from gallery/probe features."""
    sim = probe_feats @ gallery_feats.T

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


def run_full_eval(feature_extractor, eval_dict, cfg, tag=""):
    """Run Rank-1 + EER on each eval set."""
    device = cfg.device
    results = {}

    for name, ev in eval_dict.items():
        gal_feats, gal_labels = extract_features(
            feature_extractor, ev["gallery_loader"], device)
        prb_feats, prb_labels = extract_features(
            feature_extractor, ev["probe_loader"], device)
        ver = evaluate_rank1_eer(gal_feats, gal_labels,
                                 prb_feats, prb_labels)

        results[name] = {
            "rank1": ver["rank1"],
            "eer": ver["eer"],
            "n_gallery": ver["n_gallery"],
            "n_probe": ver["n_probe"],
        }
        print(f"      {tag}{name}: R1={ver['rank1']:.2f}% | "
              f"EER={ver['eer']:.2f}% | "
              f"Gal={ver['n_gallery']} Prb={ver['n_probe']}")

    return results
