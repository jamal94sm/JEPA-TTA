import os

from .STL10 import load_dataset as load_stl10
from .TinyImageNet import load_dataset as load_tiny_imagenet
from .CIFAR10 import load_dataset as load_cifar10
from .CIFAR100 import load_dataset as load_cifar100
from .STL10.loader import load_eval_dataset as load_stl10_eval
from .TinyImageNet.loader import load_eval_dataset as load_tiny_imagenet_eval


_PRETRAIN_ALIASES = {
    "stl10": "stl10",
    "stl-10": "stl10",
    "tiny-imagenet": "tiny-imagenet",
    "tinyimagenet": "tiny-imagenet",
    "tiny-imagenet-200": "tiny-imagenet",
    "tinyimagenett": "tiny-imagenet",
}

_EVAL_ALIASES = {
    "cifar10": "cifar10",
    "cifar-10": "cifar10",
    "cifar100": "cifar100",
    "cifar-100": "cifar100",
    "stl10": "stl10",
    "stl-10": "stl10",
    "tiny-imagenet": "tiny-imagenet",
    "tinyimagenet": "tiny-imagenet",
    "tiny-imagenet-200": "tiny-imagenet",
    "tinyimagenett": "tiny-imagenet",
}

_EVAL_LOADERS = {
    "cifar10": load_cifar10,
    "cifar100": load_cifar100,
    "stl10": load_stl10_eval,
    "tiny-imagenet": load_tiny_imagenet_eval,
}

_EVAL_NUM_CLASSES = {
    "cifar10": 10,
    "cifar100": 100,
    "stl10": 10,
    "tiny-imagenet": 200,
}


def data_bank_root(args=None):
    """Prefer this project's Datasets/data_bank, else the parent IB-JEPA bank."""
    override = getattr(args, "data_root", None) if args is not None else None
    if override:
        return override
    here = os.path.dirname(os.path.abspath(__file__))
    project = os.path.dirname(here)
    local = os.path.join(here, "data_bank")
    parent = os.path.normpath(os.path.join(project, "..", "Datasets", "data_bank"))
    if os.path.isdir(local):
        return local
    if os.path.isdir(parent):
        return parent
    return local


def normalize_dataset_name(name):
    key = str(name).strip().lower().replace("_", "-")
    if key not in _PRETRAIN_ALIASES:
        raise ValueError(
            f"Unknown pretrain dataset: {name!r}. Supported: stl10, tiny-imagenet"
        )
    return _PRETRAIN_ALIASES[key]


def normalize_eval_dataset_name(name):
    key = str(name).strip().lower().replace("_", "-")
    if key not in _EVAL_ALIASES:
        raise ValueError(
            f"Unknown eval_dataset: {name!r}. "
            f"Supported: cifar10, cifar100, stl10, tiny-imagenet"
        )
    return _EVAL_ALIASES[key]


def eval_num_classes(args):
    return _EVAL_NUM_CLASSES[normalize_eval_dataset_name(args.eval_dataset)]


def load_eval_dataset(args):
    key = normalize_eval_dataset_name(args.eval_dataset)
    root = os.path.join(data_bank_root(args), key)
    return _EVAL_LOADERS[key](root, args)
