from torchvision import datasets, transforms


def load_dataset(root, args):
    transform = transforms.Compose([
        transforms.Resize(args.image_size[0]),
        transforms.CenterCrop(args.image_size[1]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408),
            std=(0.2675, 0.2565, 0.2761),
        ),
    ])

    train = datasets.CIFAR100(root=root, train=True, download=True, transform=transform)
    val = datasets.CIFAR100(root=root, train=False, download=True, transform=transform)
    return train, val
