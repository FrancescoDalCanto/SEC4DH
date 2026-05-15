"""BreastMNIST loading and dataset preparation helpers."""

import torch
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
import medmnist

from console import print_message, print_metric


def build_breastmnist_transform():
    """
    Build the preprocessing transform shared by attack and victim datasets.

    Returns
    -------
    torchvision.transforms.Compose
        Transform converting grayscale BreastMNIST images to RGB tensors
        resized to ``224 x 224``.
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Resize((224, 224), antialias=True),
        ]
    )


def load_breastmnist_split(split, transform, download=False):
    """
    Load a BreastMNIST dataset split.

    Parameters
    ----------
    split : str
        Dataset split name accepted by ``medmnist.BreastMNIST``.
    transform : torchvision.transforms.Compose
        Transform applied to each sample.
    download : bool, default False
        Whether to download the dataset if it is missing locally.

    Returns
    -------
    medmnist.BreastMNIST
        BreastMNIST dataset for ``split``.
    """
    return medmnist.BreastMNIST(split=split, download=download, transform=transform)


def load_breastmnist_attack_samples(num_poison_bases=15):
    """
    Load benign bases and one malignant target from BreastMNIST.

    Parameters
    ----------
    num_poison_bases : int, default 15
        Number of normal samples, labeled ``1`` in BreastMNIST, used as bases
        for poison generation.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torchvision.transforms.Compose]
        Base image batch with shape ``(N, 3, 224, 224)``, malignant target
        image with shape ``(1, 3, 224, 224)``, and the preprocessing transform
        shared by the attack and victim datasets.
    """
    transform = build_breastmnist_transform()
    dataset = load_breastmnist_split("train", transform=transform, download=True)

    x_bases, x_target = [], None
    for img, label in dataset:
        if label[0] == 1 and len(x_bases) < num_poison_bases:  # 1 = Normal
            x_bases.append(img)
        elif label[0] == 0 and x_target is None:  # 0 = Malignant
            x_target = img.unsqueeze(0)
        if len(x_bases) == num_poison_bases and x_target is not None:
            break

    x_bases = torch.stack(x_bases)

    print_message("DATA", "Selected BreastMNIST samples for the attack.")
    print_metric(
        "base images",
        f"{len(x_bases)} Normal samples / shape={tuple(x_bases.shape)}",
    )
    print_metric("target image", f"Malignant / shape={tuple(x_target.shape)}")

    return x_bases, x_target, transform


def prepare_poisoned_dataset(clean_dataset, x_poisons, poison_label=1):
    """
    Add clean-label poison samples to the victim training set.

    Parameters
    ----------
    clean_dataset : torch.utils.data.Dataset
        Clean BreastMNIST training dataset.
    x_poisons : torch.Tensor
        Poison tensor with shape ``(3, H, W)`` or ``(N, 3, H, W)``.
    poison_label : int, default 1
        Label assigned to poison samples. In BreastMNIST, ``1`` denotes
        normal tissue.

    Returns
    -------
    torch.utils.data.TensorDataset
        Dataset containing all clean samples followed by the poison samples.
    """
    if x_poisons.ndim == 3:
        x_poisons = x_poisons.unsqueeze(0)

    print_message("DATASET", "Preparing poisoned training dataset.")
    print_metric("distinct poisons inserted", x_poisons.shape[0])
    print_metric("poison label", f"{poison_label} (Normal)")
    images, labels = [], []

    for img, label in clean_dataset:
        images.append(img)
        labels.append(torch.tensor(label[0], dtype=torch.long))

    # Poison samples keep the benign label, which preserves clean-label status.
    for poison_img in x_poisons.cpu():
        images.append(poison_img)
        labels.append(torch.tensor(poison_label, dtype=torch.long))

    return TensorDataset(torch.stack(images), torch.stack(labels))
