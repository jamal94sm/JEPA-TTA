import random
import zipfile
from collections import defaultdict
from pathlib import Path
from urllib.request import urlretrieve

from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms

TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
TINY_IMAGENET_FOLDER = "tiny-imagenet-200"

# Hold out this many labeled train images per class for downstream eval so
# pretraining never sees the linear/kNN probe training set. Official val is test.
EVAL_HOLDOUT_PER_CLASS = 50
EVAL_HOLDOUT_SEED = 0


def _ensure_tiny_imagenet(root):
    """Download and extract Tiny-ImageNet into root if missing."""
    root = Path(root)
    data_root = root / TINY_IMAGENET_FOLDER
    train_dir = data_root / "train"
    if train_dir.is_dir() and any(train_dir.iterdir()):
        return data_root

    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "tiny-imagenet-200.zip"
    if not zip_path.is_file():
        print(f"Downloading Tiny-ImageNet to {zip_path} ...")
        urlretrieve(TINY_IMAGENET_URL, zip_path)

    print(f"Extracting Tiny-ImageNet into {root} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)

    if not train_dir.is_dir():
        raise FileNotFoundError(
            f"Tiny-ImageNet train split not found at {train_dir}. "
            f"Expected layout under {data_root}."
        )
    return data_root


class TinyImageNet(Dataset):
    """
    Official Tiny-ImageNet-200 layout:
      train/<wnid>/images/*.JPEG
      val/images/*.JPEG  (+ val_annotations.txt)
    """

    def __init__(self, root, split="train", transform=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.samples = []
        self.targets = []

        wnids_path = self.root / "wnids.txt"
        with open(wnids_path, "r", encoding="utf-8") as f:
            self.classes = [line.strip() for line in f if line.strip()]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        if split == "train":
            for wnid in self.classes:
                img_dir = self.root / "train" / wnid / "images"
                if not img_dir.is_dir():
                    continue
                label = self.class_to_idx[wnid]
                for p in sorted(img_dir.glob("*.JPEG")):
                    self.samples.append(p)
                    self.targets.append(label)
        elif split == "val":
            ann_path = self.root / "val" / "val_annotations.txt"
            img_dir = self.root / "val" / "images"
            with open(ann_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) < 2:
                        continue
                    fname, wnid = parts[0], parts[1]
                    self.samples.append(img_dir / fname)
                    self.targets.append(self.class_to_idx[wnid])
        else:
            raise ValueError(f"Unknown Tiny-ImageNet split: {split!r} (use 'train' or 'val')")

        if not self.samples:
            raise RuntimeError(f"No Tiny-ImageNet images found for split={split} under {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]
        label = self.targets[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def _train_pretrain_and_eval_indices(full_train):
    """
    Stratified split of official train:
      - eval holdout: EVAL_HOLDOUT_PER_CLASS per class (linear/kNN bank)
      - pretrain: remaining train images
    Deterministic under EVAL_HOLDOUT_SEED.
    """
    by_class = defaultdict(list)
    for idx, y in enumerate(full_train.targets):
        by_class[int(y)].append(idx)

    pretrain_idx = []
    eval_train_idx = []
    rng = random.Random(EVAL_HOLDOUT_SEED)
    for y in sorted(by_class):
        idxs = list(by_class[y])
        if len(idxs) <= EVAL_HOLDOUT_PER_CLASS:
            raise RuntimeError(
                f"Tiny-ImageNet class {y} has only {len(idxs)} train images; "
                f"need > {EVAL_HOLDOUT_PER_CLASS} for eval holdout."
            )
        rng.shuffle(idxs)
        eval_train_idx.extend(idxs[:EVAL_HOLDOUT_PER_CLASS])
        pretrain_idx.extend(idxs[EVAL_HOLDOUT_PER_CLASS:])

    pretrain_idx.sort()
    eval_train_idx.sort()
    return pretrain_idx, eval_train_idx


def _transform(args):
    return transforms.Compose([
        transforms.Resize(args.image_size[0]),
        transforms.CenterCrop(args.image_size[1]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


def load_dataset(root, args):
    """
    Tiny-ImageNet pretrain set: official train minus a stratified eval holdout
    so in-domain linear/kNN never trains on images seen during SSL.
    """
    data_root = _ensure_tiny_imagenet(root)
    transform = _transform(args)
    full_train = TinyImageNet(root=data_root, split="train", transform=transform)
    pretrain_idx, _ = _train_pretrain_and_eval_indices(full_train)
    return Subset(full_train, pretrain_idx)


def load_eval_dataset(root, args):
    """
    Tiny-ImageNet downstream eval (200 classes), disjoint from pretraining:
      train = stratified holdout from official train (not used in SSL)
      test  = official val
    """
    data_root = _ensure_tiny_imagenet(root)
    transform = _transform(args)
    full_train = TinyImageNet(root=data_root, split="train", transform=transform)
    _, eval_train_idx = _train_pretrain_and_eval_indices(full_train)
    val = TinyImageNet(root=data_root, split="val", transform=transform)
    return Subset(full_train, eval_train_idx), val
