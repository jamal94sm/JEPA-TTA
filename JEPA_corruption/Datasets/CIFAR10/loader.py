import torch
from torchvision import datasets, transforms

def load_dataset(root, args):

    transform = transforms.Compose([
        transforms.Resize(args.image_size[0]),
        transforms.CenterCrop(args.image_size[1]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        )
    ])

    train = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    val = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

    return train, val