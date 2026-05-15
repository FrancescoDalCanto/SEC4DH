"""Visualization and image export helpers."""

import os

import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image

from console import print_message


def plot_attack_evaluation_summary(
    x_base,
    x_target,
    x_poison,
    predicted_label,
    success_ratio=None,
    output_path="imgs/attack_evaluation_summary.png",
):
    """
    Save a summary plot of the attack inputs and target prediction.

    Parameters
    ----------
    x_base : torch.Tensor
        Benign base image with shape ``(1, 3, H, W)``.
    x_target : torch.Tensor
        Malignant target image with shape ``(1, 3, H, W)``.
    x_poison : torch.Tensor
        Poison image with shape ``(1, 3, H, W)``.
    predicted_label : int
        Predicted target label, where ``0`` is malignant and ``1`` is normal.
    success_ratio : float, optional
        Percentage of malignant evaluation samples classified as normal.
    output_path : str, default "imgs/attack_evaluation_summary.png"
        Path where the summary plot is saved.

    Returns
    -------
    str
        Path to the saved summary plot.
    """

    def to_numpy(tensor):
        """
        Convert a batched image tensor to a display-ready array.

        Parameters
        ----------
        tensor : torch.Tensor
            Image tensor with shape ``(1, C, H, W)``.

        Returns
        -------
        numpy.ndarray
            Image array with shape ``(H, W, C)``.
        """
        return tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    if predicted_label == 1:
        test_title = (
            r"4. Test Image ($x_{target}$)"
            + "\nTrue: Malignant\nPredicted: Normal (Misclassified)"
        )
        title_color = "red"
    else:
        test_title = (
            r"4. Test Image ($x_{target}$)"
            + "\nTrue: Malignant\nPredicted: Malignant (Correct)"
        )
        title_color = "green"

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.ravel()

    axes[0].imshow(to_numpy(x_base))
    axes[0].set_title(r"1. Base Training Image" + "\nTrue: Normal\nLabel: Normal")
    axes[0].axis("off")

    axes[1].imshow(to_numpy(x_poison))
    axes[1].set_title(
        r"2. Poisoned Training Image" + "\nTrue: Normal + Poison\nLabel: Normal"
    )
    axes[1].axis("off")

    perturbation = torch.abs(x_poison - x_base)
    axes[2].imshow(to_numpy(perturbation) * 10)
    axes[2].set_title(r"3. Hidden Perturbation" + "\n(Magnified 10x)")
    axes[2].axis("off")

    axes[3].imshow(to_numpy(x_target))
    axes[3].set_title(test_title, color=title_color, fontweight="bold")
    axes[3].axis("off")

    if success_ratio is not None:
        fig.suptitle(
            "Attack success ratio over repeated malignant samples: "
            f"{success_ratio:.2f}%",
            fontsize=12,
            fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        plt.tight_layout()

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print_message("PLOT", f"Saved attack evaluation summary to {output_path}.")

    return output_path


def save_images_singly(images, output_dir, prefix="image"):
    """
    Save each image in a tensor batch as a PNG file.

    Parameters
    ----------
    images : torch.Tensor
        Image tensor with shape ``(C, H, W)`` or ``(N, C, H, W)`` and values
        in the range ``[0, 1]``.
    output_dir : str
        Directory where PNG files are written.
    prefix : str, default "image"
        Prefix used in filenames of the form ``{prefix}_{index}.png``.
    """
    if images.ndim == 3:
        images = images.unsqueeze(0)

    os.makedirs(output_dir, exist_ok=True)

    for idx, img in enumerate(images):
        filename = os.path.join(output_dir, f"{prefix}_{idx:03d}.png")
        save_image(img, filename)
