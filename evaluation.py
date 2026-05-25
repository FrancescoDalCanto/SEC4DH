"""Evaluation helpers for the poisoning experiment."""

import numpy as np
import torch
import torch.nn.functional as F

from console import print_section, print_message, print_metric


def evaluate_attack(
    victim_model,
    normalize,
    test_dataset,
    x_target,
    device,
    success_trials=25,
    section_header="Phase 4 | Evaluation and attack outcome",
    defense_active=False,
):
    """
    Evaluate clean accuracy and attack success after poisoning.

    Parameters
    ----------
    victim_model : torch.nn.Module
        Trained binary classifier.
    normalize : torchvision.transforms.Normalize
        ImageNet normalization applied before model inference.
    test_dataset : torch.utils.data.Dataset
        Held-out BreastMNIST test dataset.
    x_target : torch.Tensor
        Malignant target image with shape ``(1, 3, H, W)``.
    device : torch.device
        Device used for inference.
    success_trials : int, default 25
        Maximum number of malignant samples used to estimate the attack
        success ratio.

    Returns
    -------
    tuple[int, float]
        Predicted class for ``x_target`` and percentage of malignant trial
        samples classified as normal.
    """
    print_section(section_header)

    # Standard test accuracy is used as a simple proxy for attack stealth.
    correct, total = 0, 0
    with torch.no_grad():
        for img, label in test_dataset:
            img = normalize(img.unsqueeze(0).to(device))
            output = victim_model(img)
            _, predicted = torch.max(output.data, 1)
            total += 1
            if predicted.item() == label[0]:
                correct += 1

    clean_acc = 100 * correct / total
    print_message("EVAL", "Stealth metric: clean test-set performance.")
    print_metric("overall test accuracy", f"{clean_acc:.2f}%")
    print_metric("interpretation", "model performance remains plausible")

    # Evaluate the specific target image used during poison generation.
    with torch.no_grad():
        target_input = normalize(x_target.to(device))
        output = victim_model(target_input)
        probs = F.softmax(output, dim=1)[0]
        mal_prob = probs[0].item() * 100
        norm_prob = probs[1].item() * 100
        _, predicted_class = torch.max(output, 1)

    pred_int = predicted_class.item()
    print_message("EVAL", "Target metric: selected malignant target classification.")
    print_metric("true label", "Malignant (0)")
    print_metric(
        "predicted label",
        "Normal (1)" if pred_int == 1 else "Malignant (0)",
    )
    print_metric("confidence normal", f"{norm_prob:.1f}%")
    print_metric("confidence malignant", f"{mal_prob:.1f}%")

    # Estimate whether the poisoning effect transfers to other malignant samples.
    success_count = 0
    total_trials = 0
    normal_confidences = []

    with torch.no_grad():
        for img, label in test_dataset:
            if int(label[0]) != 0:
                continue

            trial_input = normalize(img.unsqueeze(0).to(device))
            output = victim_model(trial_input)
            probs = F.softmax(output, dim=1)[0]
            _, predicted_class = torch.max(output, 1)

            total_trials += 1
            normal_confidences.append(probs[1].item() * 100)
            if predicted_class.item() == 1:
                success_count += 1

            if total_trials >= success_trials:
                break

    success_ratio = 100 * success_count / total_trials if total_trials else 0.0
    average_normal_confidence = (
        np.mean(normal_confidences) if normal_confidences else 0.0
    )

    print_message("EVAL", "Repeated malignant-sample prediction test.")
    print_metric("samples evaluated", total_trials)
    print_metric("misclassified as Normal", success_count)
    print_metric("attack success ratio", f"{success_ratio:.2f}%")
    print_metric("average Normal confidence", f"{average_normal_confidence:.1f}%")

    if pred_int == 1:
        print_message(
            "RESULT",
            "Attack successful: the malignant target was classified as Normal.",
        )
    elif defense_active:
        print_message(
            "RESULT",
            "Defense effective: poison samples removed before training — attack neutralized.",
        )
    else:
        print_message(
            "RESULT",
            "Attack unsuccessful: the malignant target was correctly detected.",
        )

    return pred_int, success_ratio
