from torchvision import datasets, transforms


def _transform(args):
    return transforms.Compose([
        transforms.Resize(args.image_size[0]),
        transforms.CenterCrop(args.image_size[1]),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4467, 0.4398, 0.4066), std=(0.2603, 0.2566, 0.2713)),
    ])


def load_dataset(root, args):
    """
    STL-10 unlabeled split for pretraining (100k images).
    Disjoint from the labeled train/test used for downstream eval.
    """
    transform = _transform(args)
    train_unlabeled = datasets.STL10(
        root=root, split="unlabeled", download=True, transform=transform
    )
    return train_unlabeled


def load_eval_dataset(root, args):
    """
    Labeled STL-10 (train, test) for downstream eval (10 classes).
    Official labeled splits do not overlap the unlabeled pretrain set.
    """
    transform = _transform(args)
    train = datasets.STL10(root=root, split="train", download=True, transform=transform)
    test = datasets.STL10(root=root, split="test", download=True, transform=transform)
    return train, test






