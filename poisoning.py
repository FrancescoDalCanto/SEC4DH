"""Poison sample generation routines."""

import torch
import torch.nn.functional as F

from console import print_message, print_metric


def generate_feature_collision_poisons(
    f, x_base, x_target, epsilon=16 / 255, steps=500, lr=0.02
):
    """
    Generate clean-label poison samples by feature collision.

    The optimizer moves each poison image toward the target in feature space
    while projecting it back into an L-infinity ball around its benign base
    image after every update.

    Parameters
    ----------
    f : torch.nn.Module
        Frozen feature extractor mapping image tensors to feature vectors.
    x_base : torch.Tensor
        Benign base image tensor with shape ``(3, H, W)`` or
        ``(N, 3, H, W)``.
    x_target : torch.Tensor
        Target image tensor with shape ``(1, 3, H, W)``.
    epsilon : float, default 16/255
        Maximum absolute per-pixel change from the base image.
    steps : int, default 500
        Number of optimization steps.
    lr : float, default 0.02
        Adam learning rate used to update the poison tensor.

    Returns
    -------
    torch.Tensor
        Poison image tensor with shape ``(N, 3, H, W)``, clipped to
        ``[0, 1]`` and the perturbation bound.
    """
    # Ensure batch dimension for consistent processing.
    if x_base.ndim == 3:
        x_base = x_base.unsqueeze(0)

    # Compute target features once since they don't change during optimization.
    target_features = f(x_target).detach()
    if target_features.shape[0] == 1 and x_base.shape[0] > 1:
        target_features = target_features.expand(x_base.shape[0], -1)

    # Initialize poison as a copy of the base image with gradients enabled.
    x_poison = x_base.clone().detach()
    x_poison.requires_grad = True

    # Use Adam optimizer for efficient feature-space navigation.
    optimizer = torch.optim.Adam([x_poison], lr=lr)

    print_message("ATTACK", "Starting feature-collision optimization.")
    print_metric("poison instances", x_base.shape[0])
    print_metric("maximum perturbation", f"{epsilon:.4f}")
    print_metric("optimization steps", steps)
    print_metric("learning rate", f"{lr:.4f}")

    # Optimization loop: move poisons toward target features while enforcing L-infinity bounds.
    for step in range(steps):
        optimizer.zero_grad()

        # Forward phase: move poison features toward the target representation.
        poison_features = f(x_poison) # Compute current features of the poison samples.

        # The loss encourages the poison features to match the target features.
        loss = F.mse_loss(poison_features, target_features)

        # Compute gradients and take an optimization step.
        loss.backward()
        optimizer.step()

        # Backward phase: project poisons back into the valid input domain.
        with torch.no_grad():
            # Compute the per-pixel perturbation from the base image.
            delta = x_poison - x_base

            # Project the perturbation back into the L-infinity ball defined by epsilon.
            delta = torch.clamp(delta, min=-epsilon, max=epsilon)

            # Update the poison image to be the base plus the projected perturbation, ensuring valid pixel values.
            x_poison.copy_(torch.clamp(x_base + delta, min=0.0, max=1.0))

        if (step + 1) % 100 == 0 or step == 0:
            print(f"[  ATTACK  ] Step {step + 1:03d}/{steps} | feature MSE: {loss.item():.4f}")

    return x_poison.detach()
