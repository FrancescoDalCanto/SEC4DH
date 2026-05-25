"""Victim model training utilities."""

import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms

from console import print_message
from features import IMAGENET_MEAN, IMAGENET_STD


def train_victim_model(train_loader, device, epochs=10):
    """
    Train the victim classifier on clean and poisoned samples.

    Parameters
    ----------
    train_loader : torch.utils.data.DataLoader
        Loader yielding image tensors and binary BreastMNIST labels.
    device : torch.device
        Device used for model parameters and mini-batches.
    epochs : int, default 10
        Number of passes over ``train_loader``.

    Returns
    -------
    tuple[torch.nn.Module, torchvision.transforms.Normalize]
        Trained ResNet18 classifier and normalization transform required
        before inference.
    """
    print_message("TRAIN", "Training ResNet18 transfer-learning classifier.")

    weights = models.ResNet18_Weights.IMAGENET1K_V1
    victim_model = models.resnet18(weights=weights)

    # Freeze the pretrained backbone and train only the binary classifier head.
    for param in victim_model.parameters():
        param.requires_grad = False

    num_ftrs = victim_model.fc.in_features
    victim_model.fc = nn.Linear(num_ftrs, 2)
    victim_model = victim_model.to(device)

    def set_transfer_learning_mode():
        """Keep frozen backbone statistics fixed while training the head."""
        victim_model.eval()
        victim_model.fc.train()

    set_transfer_learning_mode()

    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    # A small learning rate keeps the classifier head sensitive to the poisons.
    optimizer = optim.Adam(victim_model.fc.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        # Eval mode prevents BatchNorm drift in the frozen pretrained backbone.
        set_transfer_learning_mode()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = normalize(inputs)

            optimizer.zero_grad()
            outputs = victim_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"[  TRAIN  ] Epoch {epoch + 1:02d}/{epochs} | average loss: {running_loss / len(train_loader):.4f}")

    victim_model.eval()
    return victim_model, normalize
