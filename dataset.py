"""
dataset.py — CASIA-MS data loading + splitting for 3 JEPA modes.

Filename format: {subjectID}_{handSide}_{spectrum}_{iteration}.jpg
Identity = subjectID_handSide (unique biometric identity)
"""

import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


ALL_SPECTRUMS = ["460", "630", "700", "850", "940", "WHT"]


def parse_filename(fname):
    """Parse CASIA-MS filename → (identity, spectrum, iteration)."""
    name = os.path.splitext(fname)[0]
    parts = name.split("_")
    if len(parts) < 4:
        return None
    subject = parts[0]
    hand = parts[1]
    spectrum = parts[2]
    iteration = parts[3]
    identity = f"{subject}_{hand}"
    return identity, spectrum, iteration


def scan_dataset(data_dir):
    """Scan dataset directory → list of (filepath, identity, spectrum)."""
    samples = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.lower().endswith((".jpg", ".png", ".bmp")):
            continue
        parsed = parse_filename(fname)
        if parsed is None:
            continue
        identity, spectrum, _ = parsed
        samples.append({
            "path": os.path.join(data_dir, fname),
            "identity": identity,
            "spectrum": spectrum,
        })
    return samples


def build_id_map(samples):
    """Build identity → integer label mapping."""
    ids = sorted(set(s["identity"] for s in samples))
    return {name: idx for idx, name in enumerate(ids)}


# ══════════════════════════════════════════════════════════════
#  Data splitting for 3 modes
# ══════════════════════════════════════════════════════════════

def split_mode_all(samples, test_ratio=0.2, seed=2025):
    """
    Mode 1: all domains × all IDs.
    Random sample-wise split.
    Returns: train_samples, {"all": test_samples}
    """
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    n_test = int(len(shuffled) * test_ratio)
    test = shuffled[:n_test]
    train = shuffled[n_test:]
    return train, {"all": test}


def split_mode_cross_domain(samples, train_spectrums, seed=2025):
    """
    Mode 2: selected domains × all IDs.
    ALL training domain samples used for training (no held-out).
    Eval on each unseen domain separately.
    Returns: train_samples, {"460": [...], "630": [...], ...}
    """
    unseen_spectrums = [s for s in ALL_SPECTRUMS if s not in train_spectrums]

    train = [s for s in samples if s["spectrum"] in train_spectrums]

    eval_sets = {}
    for sp in unseen_spectrums:
        sp_samples = [s for s in samples if s["spectrum"] == sp]
        if sp_samples:
            eval_sets[sp] = sp_samples

    return train, eval_sets


def split_mode_cross_domain_openset(samples, train_spectrums,
                                     train_id_ratio=0.8, seed=2025):
    """
    Mode 3: selected domains × selected IDs.
    ALL training domain × training ID samples used for training.
    3 evaluation sets: seen_dom×unseen_id, unseen_dom×seen_id,
                       unseen_dom×unseen_id.
    Returns: train_samples, {eval_name: eval_samples, ...}
    """
    rng = random.Random(seed)
    all_ids = sorted(set(s["identity"] for s in samples))
    rng.shuffle(all_ids)
    n_train_ids = int(len(all_ids) * train_id_ratio)
    train_ids = set(all_ids[:n_train_ids])
    unseen_ids = set(all_ids[n_train_ids:])
    unseen_spectrums = [s for s in ALL_SPECTRUMS if s not in train_spectrums]

    # Training: ALL seen domains × seen IDs
    train = [s for s in samples
             if s["spectrum"] in train_spectrums
             and s["identity"] in train_ids]

    eval_sets = {}

    # Seen domain × unseen IDs
    seen_unseen = [s for s in samples
                   if s["spectrum"] in train_spectrums
                   and s["identity"] in unseen_ids]
    if seen_unseen:
        eval_sets["seen_dom_unseen_id"] = seen_unseen

    # Unseen domain × seen IDs
    unseen_seen = [s for s in samples
                   if s["spectrum"] in unseen_spectrums
                   and s["identity"] in train_ids]
    if unseen_seen:
        eval_sets["unseen_dom_seen_id"] = unseen_seen

    # Unseen domain × unseen IDs
    unseen_unseen = [s for s in samples
                     if s["spectrum"] in unseen_spectrums
                     and s["identity"] in unseen_ids]
    if unseen_unseen:
        eval_sets["unseen_dom_unseen_id"] = unseen_unseen

    return train, eval_sets


# ══════════════════════════════════════════════════════════════
#  PyTorch Dataset
# ══════════════════════════════════════════════════════════════

class CASIADataset(Dataset):
    """CASIA-MS dataset with optional augmentation multiplier."""

    def __init__(self, samples, id_map, img_size=112, augment=False,
                 aug_multiplier=1):
        self.samples = samples
        self.id_map = id_map
        self.augment = augment
        self.aug_multiplier = aug_multiplier if augment else 1

        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.3, 0.3, 0.1, 0.05),
                ], p=0.5),
                transforms.RandomApply([
                    transforms.GaussianBlur(5, sigma=(0.1, 1.0)),
                ], p=0.3),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3),
            ])

    def __len__(self):
        return len(self.samples) * self.aug_multiplier

    def __getitem__(self, idx):
        real_idx = idx % len(self.samples)
        s = self.samples[real_idx]
        img = Image.open(s["path"]).convert("RGB")
        img = self.transform(img)
        label = self.id_map[s["identity"]]
        return img, label


# ══════════════════════════════════════════════════════════════
#  Gallery / Probe splitting for verification
# ══════════════════════════════════════════════════════════════

def split_gallery_probe(samples, id_map, gallery_ratio=0.5, seed=2025):
    """
    Split samples into gallery and probe per identity.
    Returns: gallery_samples, probe_samples
    """
    rng = random.Random(seed)
    by_id = {}
    for s in samples:
        by_id.setdefault(s["identity"], []).append(s)

    gallery, probe = [], []
    for identity, id_samples in by_id.items():
        rng.shuffle(id_samples)
        n_gal = max(1, int(len(id_samples) * gallery_ratio))
        gallery.extend(id_samples[:n_gal])
        probe.extend(id_samples[n_gal:])

    return gallery, probe


# ══════════════════════════════════════════════════════════════
#  Build everything for a given mode
# ══════════════════════════════════════════════════════════════

def build_datasets(cfg):
    """
    Build train + eval datasets for the configured mode.
    Returns: train_loader, eval_dict, id_map, info_str
    """
    all_samples = scan_dataset(cfg.data_dir)
    print(f"  Total samples: {len(all_samples)}")
    print(f"  Spectrums: {sorted(set(s['spectrum'] for s in all_samples))}")
    print(f"  Identities: {len(set(s['identity'] for s in all_samples))}")

    if cfg.mode == "all":
        train_samples, eval_sets = split_mode_all(
            all_samples, cfg.test_sample_ratio, cfg.seed)
        info = "All domains × All IDs"
    elif cfg.mode == "cross_domain":
        train_samples, eval_sets = split_mode_cross_domain(
            all_samples, cfg.train_spectrums, cfg.seed)
        info = f"Train domains: {cfg.train_spectrums}"
    elif cfg.mode == "cross_domain_openset":
        train_samples, eval_sets = split_mode_cross_domain_openset(
            all_samples, cfg.train_spectrums,
            cfg.train_id_ratio, cfg.seed)
        info = (f"Train domains: {cfg.train_spectrums}, "
                f"Train ID ratio: {cfg.train_id_ratio}")

    # Build global ID map from ALL samples (so labels are consistent)
    id_map = build_id_map(all_samples)
    n_classes = len(id_map)

    print(f"\n  Mode: {cfg.mode} ({info})")
    print(f"  Train samples: {len(train_samples)} "
          f"(×{cfg.aug_multiplier} aug = "
          f"{len(train_samples) * cfg.aug_multiplier})")
    for name, samples in eval_sets.items():
        n_ids = len(set(s["identity"] for s in samples))
        print(f"  Eval '{name}': {len(samples)} samples, {n_ids} IDs")

    train_ds = CASIADataset(train_samples, id_map, cfg.img_size,
                             augment=True, aug_multiplier=cfg.aug_multiplier)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                               shuffle=True, num_workers=cfg.num_workers,
                               drop_last=True, pin_memory=True)

    eval_dict = {}
    for name, samples in eval_sets.items():
        gal_samples, prb_samples = split_gallery_probe(
            samples, id_map, cfg.gallery_ratio, cfg.seed)
        gal_ds = CASIADataset(gal_samples, id_map, cfg.img_size, augment=False)
        prb_ds = CASIADataset(prb_samples, id_map, cfg.img_size, augment=False)
        gal_loader = DataLoader(gal_ds, batch_size=cfg.batch_size,
                                shuffle=False, num_workers=cfg.num_workers)
        prb_loader = DataLoader(prb_ds, batch_size=cfg.batch_size,
                                shuffle=False, num_workers=cfg.num_workers)
        eval_dict[name] = {
            "gallery_loader": gal_loader,
            "probe_loader": prb_loader,
            "n_samples": len(samples),
            "n_ids": len(set(s["identity"] for s in samples)),
            "n_gallery": len(gal_samples),
            "n_probe": len(prb_samples),
        }

    return train_loader, eval_dict, id_map, n_classes

# ========================================================
XJTU-UP dataset
# ============================================================

def scan_xjtu(data_root):
    """Load XJTU-UP as CASIA-style sample dicts so the rest of the pipeline
       (CASIADataset, split_gallery_probe, build_id_map) consumes it unchanged.

       Directory layout:  data_root / <device> / <condition> / <id_folder> / *.img
       where <id_folder> looks like 'L_003' or 'R_012' (hand_subject).

       XJTU has no spectrum, so every sample is tagged spectrum='XJTU' and reads
       as ONE target domain. Identities are namespaced 'XJTU_<id_folder>' so they
       can never collide with CASIA identities in a shared id_map.
    """
    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
    samples = []
    ids = set()
    for device in sorted(os.listdir(data_root)):
        dev_dir = os.path.join(data_root, device)
        if not os.path.isdir(dev_dir):
            continue
        for condition in sorted(os.listdir(dev_dir)):
            cond_dir = os.path.join(dev_dir, condition)
            if not os.path.isdir(cond_dir):
                continue
            for id_folder in sorted(os.listdir(cond_dir)):
                id_dir = os.path.join(cond_dir, id_folder)
                if not os.path.isdir(id_dir):
                    continue
                identity = f"XJTU_{id_folder}"
                for fname in sorted(os.listdir(id_dir)):
                    if fname.lower().endswith(IMG_EXTS):
                        samples.append({
                            "path": os.path.join(id_dir, fname),
                            "identity": identity,
                            "spectrum": "XJTU",
                        })
                        ids.add(identity)
    print(f"  [XJTU] {len(samples)} samples, {len(ids)} identities from {data_root}")
    return samples
