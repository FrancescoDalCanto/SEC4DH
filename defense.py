"""Feature-space poison detection defense."""

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset

from console import print_message, print_metric, print_section


def detect_and_remove_poisons(
    dataset,
    feature_extractor,
    x_target,
    device,
    suspect_label=1,
    similarity_threshold=0.90,
    section_header="Defense | Feature-space poison detection",
):
    """
    Detect clean-label poisons by measuring cosine similarity to the target
    in feature space and return a sanitized dataset.

    The feature-collision attack crafts poison images whose ResNet features
    collapse toward the target. This defense exploits that property: any
    sample whose feature vector is suspiciously close to the target's vector
    (above ``similarity_threshold``) is flagged and removed.

    Parameters
    ----------
    dataset : torch.utils.data.TensorDataset
        Poisoned training dataset with tensors (images, labels).
    feature_extractor : FrozenResNetFeatureExtractor
        Frozen ResNet18 backbone shared with the attacker.
    x_target : torch.Tensor
        Malignant target image with shape ``(1, 3, H, W)``.
    device : torch.device
        Device used for inference.
    suspect_label : int, default 1
        Only inspect samples carrying this label (the poison label).
    similarity_threshold : float, default 0.90
        Cosine similarity above which a sample is considered a poison.
        Legitimate normal samples sit far from the malignant target in
        feature space; poisoned ones are very close (by construction).
        Tune this value based on the observed similarity distribution printed
        by the defense — with small epsilon (e.g., 4/255) the poisons may
        not exceed 0.90 and a lower threshold (0.70–0.80) may be needed.

    Returns
    -------
    torch.utils.data.TensorDataset
        Dataset with detected poison samples removed.
    """
    print_section(section_header)

    images, labels = dataset.tensors
    feature_extractor.eval()

    with torch.no_grad():
        target_feat = feature_extractor(x_target.to(device))
        target_feat = F.normalize(target_feat, dim=1)

    kept_images, kept_labels = [], []
    flagged = 0
    suspect_similarities = []

    with torch.no_grad():
        for img, label in zip(images, labels):
            if int(label.item()) == suspect_label:
                feat = feature_extractor(img.unsqueeze(0).to(device))
                feat = F.normalize(feat, dim=1)
                sim = (feat * target_feat).sum().item()
                suspect_similarities.append(sim)

                if sim >= similarity_threshold:
                    flagged += 1
                    continue

            kept_images.append(img)
            kept_labels.append(label)

    total = len(images)
    kept = len(kept_images)

    if suspect_similarities:
        import statistics
        print_metric(
            "suspect similarity range",
            f"[{min(suspect_similarities):.4f}, {max(suspect_similarities):.4f}]",
        )
        print_metric("suspect similarity mean", f"{statistics.mean(suspect_similarities):.4f}")

    print_metric("total samples inspected", total)
    print_metric("suspect label", f"{suspect_label} (Normal)")
    print_metric("similarity threshold", f"{similarity_threshold:.2f}")
    print_metric("samples flagged as poisons", flagged)
    print_metric("samples retained", kept)

    if flagged > 0:
        print_message(
            "DEFENSE",
            f"Removed {flagged} suspected poison(s) from the training set.",
        )
    else:
        print_message("DEFENSE", "No poisons detected above the threshold.")

    return TensorDataset(torch.stack(kept_images), torch.stack(kept_labels))
