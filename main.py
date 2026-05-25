"""
Run a clean-label data-poisoning attack on BreastMNIST images.

The experiment follows the feature-collision attack from Shafahi et al.,
"Poison Frogs! Targeted Clean-Label Poisoning Attacks". It creates benign
training images whose feature representations move toward a malignant target
image, then measures the effect on a transfer-learning classifier.
"""

import argparse
import warnings

import torch
from torch.utils.data import DataLoader

from console import print_metric, print_section, print_message
from data import (
    load_breastmnist_attack_samples,
    load_breastmnist_split,
    prepare_poisoned_dataset,
)
from defense import detect_and_remove_poisons
from evaluation import evaluate_attack
from features import FrozenResNetFeatureExtractor
from poisoning import generate_feature_collision_poisons
from training import train_victim_model
from visualization import plot_attack_evaluation_summary, save_images_singly


# Suppress non-critical warnings to keep the experiment output readable.
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean-label poisoning experiment.")
    # When passed, the defense sanitizes the poisoned training set before the
    # victim is trained, allowing a direct comparison of attack success with and
    # without the countermeasure.
    parser.add_argument(
        "--defense",
        action="store_true",
        help="Enable feature-space poison detection before training.",
    )
    args = parser.parse_args()

    # Uncomment these seeds when deterministic sample selection is required.
    # torch.manual_seed(42)
    # np.random.seed(42)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print_section("Experiment configuration")
    print_metric("hardware accelerator", device.type.upper())

    # craft bounded poison samples with the attacker feature extractor.
    print_section("Phase 1 | Attacker poison generation")

    num_poison_instances = 20
    feature_extractor = FrozenResNetFeatureExtractor(device).to(device)
    x_bases, x_target, transform = load_breastmnist_attack_samples(
        num_poison_bases=num_poison_instances,
        feature_extractor=feature_extractor,
        device=device,
    )
    x_bases, x_target = x_bases.to(device), x_target.to(device)

    # The epsilon bound controls the maximum pixel-level perturbation.
    epsilon_bound = 8 / 255
    x_poisons = generate_feature_collision_poisons(
        feature_extractor,
        x_bases,
        x_target,
        epsilon=epsilon_bound,
        steps=800,
        lr=0.02,
    )

    l_inf_norm = torch.max(torch.abs(x_poisons - x_bases)).item()
    print_message("VERIFY", "Perturbation bound check completed.")
    print_metric("observed L-inf norm", f"{l_inf_norm:.4f}")
    print_metric("allowed L-inf norm", f"{epsilon_bound:.4f}")

    # Iinsert clean-label poisons into the victim's training set.
    print_section("Phase 2 | Poison deployment")

    train_dataset_clean = load_breastmnist_split(
        split="train",
        download=False,
        transform=transform,
    )
    test_dataset_clean = load_breastmnist_split(
        split="test",
        download=False,
        transform=transform,
    )

    # Multiple base images improve feature-space coverage around the target.
    poisoned_dataset = prepare_poisoned_dataset(
        train_dataset_clean,
        x_poisons,
        poison_label=1,
    )
    # Phase 2b (optional): remove suspected poisons before handing the dataset
    # to the victim. The same feature extractor used by the attacker is reused
    # here because the defense exploits the attacker's own objective — poisons
    # designed to collide with the target in feature space will also be detected
    # by measuring cosine similarity in that same space.
    if args.defense:
        sanitized_dataset = detect_and_remove_poisons(
            poisoned_dataset,
            feature_extractor,
            x_target,
            device,
            suspect_label=1,
            similarity_threshold=0.95,
        )
        train_loader = DataLoader(sanitized_dataset, batch_size=32, shuffle=True)
    else:
        # No defense: train directly on the poisoned dataset to measure the
        # full attack impact as the baseline.
        train_loader = DataLoader(poisoned_dataset, batch_size=32, shuffle=True)

    # Train the victim model on the (optionally sanitized) dataset.
    print_section("Phase 3 | Victim model training")

    victim_model, norm_fn = train_victim_model(train_loader, device, epochs=10)

    # Measure clean accuracy and target-specific misclassification.
    final_prediction, success_ratio = evaluate_attack(
        victim_model,
        norm_fn,
        test_dataset_clean,
        x_target,
        device,
        success_trials=25,
        section_header="Phase 4 | Evaluation and attack outcome",
        defense_active=args.defense,
    )

    # Visualize the base, poison, perturbation, and target prediction.
    print_section("Phase 5 | Visualization")

    save_images_singly(x_bases[:1], "imgs", prefix="base")
    save_images_singly(x_poisons[:1], "imgs", prefix="poison")
    save_images_singly(x_target, "imgs", prefix="target")

    plot_attack_evaluation_summary(
        x_bases[:1],
        x_target,
        x_poisons[:1],
        final_prediction,
        success_ratio=success_ratio,
        output_path="imgs/attack_evaluation_summary.png",
    )


