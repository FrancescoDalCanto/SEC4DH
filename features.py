"""Feature extractors used by the poisoning attack."""

import torch
import torch.nn as nn
import torchvision.models as models


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FrozenResNetFeatureExtractor(nn.Module):
    """
    Extract frozen ResNet18 features for poison optimization.

    The attacker uses an ImageNet-pretrained ResNet18 without its
    classification head to approximate the victim model's representation
    space. The backbone remains frozen so gradients update only the poison
    image.

    Parameters
    ----------
    device : torch.device
        Device used to store the ImageNet normalization buffers.
    """

    def __init__(self, device):
        """
        Initialize the frozen ResNet18 feature extractor.

        Parameters
        ----------
        device : torch.device
            Device used to store the ImageNet normalization buffers.
        """
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        self.f = models.resnet18(weights=weights)
        self.f.fc = nn.Identity()
        self.f.eval()

        # Keep the feature extractor fixed so only the poison image is updated.
        for param in self.f.parameters():
            param.requires_grad = False

        # ResNet18 expects ImageNet-normalized RGB inputs.
        self.register_buffer(
            "mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(device)
        )
        self.register_buffer(
            "std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(device)
        )

    def forward(self, x):
        """
        Return ResNet18 feature vectors for normalized RGB images.

        Parameters
        ----------
        x : torch.Tensor
            RGB image tensor with shape ``(N, 3, H, W)`` and values in
            ``[0, 1]``.

        Returns
        -------
        torch.Tensor
            Feature tensor with shape ``(N, 512)``.
        """
        x = (x - self.mean) / self.std
        return self.f(x)
