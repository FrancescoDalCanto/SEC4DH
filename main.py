"""
Clean-label data poisoning attack on BreastMNIST medical images.

This script follows the feature-collision attack described by Shafahi et al.
in "Poison Frogs! Targeted Clean-Label Poisoning Attacks". The demonstration
constructs a visually plausible benign training image whose feature
representation is moved toward a malignant target image, then evaluates how
that poison affects a transfer-learning classifier.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import medmnist
import numpy as np
import warnings

# Suppress non-critical warnings to keep the experiment output readable.
warnings.filterwarnings("ignore")


LOG_WIDTH = 72


def print_section(title):
    """
    Print a standardized section header.

    Parameters
    ----------
    title : str
        Title displayed in the section header.
    """
    print("\n" + "=" * LOG_WIDTH)
    print(title)
    print("=" * LOG_WIDTH)


def print_message(scope, message):
    """
    Print a standardized single-line status message.

    Parameters
    ----------
    scope : str
        Short label describing the current stage.
    message : str
        Human-readable status message.
    """
    print(f"[{scope:<10}] {message}")


def print_metric(label, value):
    """
    Print an aligned metric or configuration value.

    Parameters
    ----------
    label : str
        Metric or field name.
    value : object
        Value displayed next to the label.
    """
    print(f"  {label:<30}: {value}")


# =====================================================================
# Part 1: poison generation
# =====================================================================


class SurrogateModel(nn.Module):
    """
    Surrogate feature extractor used by the attacker.

    The surrogate approximates the victim model's representation space using an
    ImageNet-pretrained ResNet18 with its classification head removed. The
    frozen feature extractor is used only to optimize the poison image.

    Parameters
    ----------
    device : torch.device
        Device on which the normalization buffers should be allocated.
    """
    def __init__(self, device):
        """
        Initialize the frozen ResNet18 surrogate.

        Parameters
        ----------
        device : torch.device
            Device on which normalization buffers should be stored.
        """
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        self.f = models.resnet18(weights=weights)
        self.f.fc = nn.Identity()
        self.f.eval()

        # Keep the surrogate fixed so gradients update only the poison image.
        for param in self.f.parameters():
            param.requires_grad = False

        # ResNet18 expects ImageNet-normalized RGB inputs.
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        )

    def forward(self, x):
        """
        Extract normalized ResNet18 feature representations.

        Parameters
        ----------
        x : torch.Tensor
            Batch of RGB images with values in the range ``[0, 1]``.

        Returns
        -------
        torch.Tensor
            Feature vectors produced by the frozen ResNet18 backbone.
        """
        x = (x - self.mean) / self.std
        return self.f(x)


def shafahi_feature_collision(f, x_base, x_target, epsilon=16/255, steps=500, lr=0.02):
    """
    Generate a clean-label poison through feature-space collision.

    The optimization minimizes the distance between the poison and target
    feature representations while constraining the poison to remain close to
    the benign base image under an L-infinity perturbation bound.

    Parameters
    ----------
    f : torch.nn.Module
        Frozen feature extractor used to compute image representations.
    x_base : torch.Tensor
        Benign image or batch of benign images used as the visual basis for
        the poison samples.
    x_target : torch.Tensor
        Target image whose feature representation should be imitated.
    epsilon : float, default 16/255
        Maximum allowed per-pixel perturbation from the base image.
    steps : int, default 500
        Number of optimization steps.
    lr : float, default 0.02
        Learning rate for the poison optimization.

    Returns
    -------
    torch.Tensor
        Optimized poison image batch clipped to valid pixel values and the
        L-infinity perturbation bound.
    """
    if x_base.ndim == 3:
        x_base = x_base.unsqueeze(0)

    target_features = f(x_target).detach()
    if target_features.shape[0] == 1 and x_base.shape[0] > 1:
        target_features = target_features.expand(x_base.shape[0], -1)

    x_poison = x_base.clone().detach()
    x_poison.requires_grad = True

    optimizer = torch.optim.Adam([x_poison], lr=lr)

    print_message("ATTACK", "Starting feature-collision optimization.")
    print_metric("poison instances", x_base.shape[0])
    print_metric("maximum perturbation", f"{epsilon:.4f}")
    print_metric("optimization steps", steps)
    print_metric("learning rate", f"{lr:.4f}")

    for step in range(steps):
        optimizer.zero_grad()

        # Minimize feature distance between the current poison and target image.
        poison_features = f(x_poison)
        loss = F.mse_loss(poison_features, target_features)

        loss.backward()
        optimizer.step()

        # Project the poison back into the valid image and perturbation domains.
        with torch.no_grad():
            delta = x_poison - x_base
            delta = torch.clamp(delta, min=-epsilon, max=epsilon)
            x_poison.copy_(torch.clamp(x_base + delta, min=0.0, max=1.0))

        if (step + 1) % 100 == 0 or step == 0:
            print(
                f"[ATTACK    ] Step {step + 1:03d}/{steps} | "
                f"feature MSE: {loss.item():.4f}"
            )

    return x_poison.detach()


def load_breastmnist_attack_samples(num_poison_bases=15):
    """
    Load benign base images and one malignant target image.

    Parameters
    ----------
    num_poison_bases : int, default 15
        Number of distinct benign base images used to craft poison samples.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torchvision.transforms.Compose]
        Batch of base images, target image, and preprocessing transform used
        for BreastMNIST samples.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Resize((224, 224), antialias=True),
    ])
    dataset = medmnist.BreastMNIST(split="train", download=True, transform=transform)

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


# =====================================================================
# Part 2: victim training and exploitation
# =====================================================================


def prepare_poisoned_dataset(clean_dataset, x_poisons, poison_label=1):
    """
    Append clean-label poison samples to the victim's training dataset.

    Parameters
    ----------
    clean_dataset : torch.utils.data.Dataset
        Original training dataset used by the victim.
    x_poisons : torch.Tensor
        Poison image batch generated by the feature-collision attack.
    poison_label : int, default 1
        Label assigned to each poison sample. For BreastMNIST, ``1`` is benign.

    Returns
    -------
    torch.utils.data.TensorDataset
        Training dataset containing the original samples plus poison samples.
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

    # Insert distinct poison samples with the benign label to preserve clean-label status.
    for poison_img in x_poisons.cpu():
        images.append(poison_img)
        labels.append(torch.tensor(poison_label, dtype=torch.long))

    return TensorDataset(torch.stack(images), torch.stack(labels))


def train_victim_model(train_loader, device, epochs=10):
    """
    Train the victim classifier on the poisoned dataset.

    Parameters
    ----------
    train_loader : torch.utils.data.DataLoader
        DataLoader containing clean and poisoned training samples.
    device : torch.device
        Device on which the model and batches should be processed.
    epochs : int, default 10
        Number of training epochs for the classifier head.

    Returns
    -------
    tuple[torch.nn.Module, torchvision.transforms.Normalize]
        Trained victim model and normalization transform used at inference.
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
    victim_model.train()

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # Use a conservative learning rate to retain the influence of the poison samples.
    optimizer = optim.Adam(victim_model.fc.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
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

        print(
            f"[TRAIN     ] Epoch {epoch + 1:02d}/{epochs} | "
            f"average loss: {running_loss / len(train_loader):.4f}"
        )

    victim_model.eval()
    return victim_model, normalize


def evaluate_attack(
    victim_model,
    normalize,
    test_dataset,
    x_target,
    device,
    success_trials=25,
):
    """
    Evaluate clean accuracy and target-specific attack success.

    Parameters
    ----------
    victim_model : torch.nn.Module
        Trained classifier evaluated after poisoning.
    normalize : torchvision.transforms.Normalize
        Input normalization transform used by the victim model.
    test_dataset : torch.utils.data.Dataset
        Held-out BreastMNIST test set.
    x_target : torch.Tensor
        Malignant target image used to test the attack outcome.
    device : torch.device
        Device on which evaluation should run.
    success_trials : int, default 25
        Number of malignant test samples used to estimate the attack success
        ratio after training.

    Returns
    -------
    tuple[int, float]
        Predicted class for the selected target image and the attack success
        ratio over the malignant evaluation samples.
    """
    print_section("Phase 4 | Evaluation and attack outcome")

    # Stealth is approximated by preserving standard test-set accuracy.
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

    # First, evaluate the selected target image used during poison generation.
    with torch.no_grad():
        target_input = normalize(x_target.to(device))
        output = victim_model(target_input)
        probs = torch.nn.functional.softmax(output, dim=1)[0]
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

    # Then, estimate the attack success ratio across multiple malignant samples.
    success_count = 0
    total_trials = 0
    normal_confidences = []

    with torch.no_grad():
        for img, label in test_dataset:
            if int(label[0]) != 0:
                continue

            trial_input = normalize(img.unsqueeze(0).to(device))
            output = victim_model(trial_input)
            probs = torch.nn.functional.softmax(output, dim=1)[0]
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
    else:
        print_message(
            "RESULT",
            "Attack unsuccessful: the malignant target was correctly detected.",
        )

    return pred_int, success_ratio


# =====================================================================
# Part 3: visualization
# =====================================================================


def plot_academic_results(
    x_base,
    x_target,
    x_poison,
    predicted_label,
    success_ratio=None,
):
    """
    Display the attack images and prediction result in a 2-by-2 layout.

    Parameters
    ----------
    x_base : torch.Tensor
        Benign base training image.
    x_target : torch.Tensor
        Malignant target image used for attack evaluation.
    x_poison : torch.Tensor
        Generated poison image.
    predicted_label : int
        Victim model prediction for the target image.
    success_ratio : float, optional
        Attack success ratio computed over repeated malignant-sample
        predictions.
    """
    def to_numpy(tensor):
        """
        Convert a batched image tensor to a NumPy image array.

        Parameters
        ----------
        tensor : torch.Tensor
            Batched image tensor with shape ``(1, C, H, W)``.

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

    print_message("PLOT", "Opening visualization window. Close it to exit the script.")
    plt.show()


# =====================================================================
# Main execution
# =====================================================================
if __name__ == "__main__":
    # Fix random seeds to make the experiment reproducible across runs.
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print_section("Experiment configuration")
    print_metric("hardware accelerator", device.type.upper())

    # Phase 1: construct a bounded poison image with the attacker surrogate.
    print_section("Phase 1 | Attacker poison generation")

    num_poison_instances = 15
    x_bases, x_target, transform = load_breastmnist_attack_samples(
        num_poison_bases=num_poison_instances,
    )
    x_bases, x_target = x_bases.to(device), x_target.to(device)
    surrogate = SurrogateModel(device).to(device)

    # The epsilon bound limits the maximum pixel-level perturbation.
    epsilon_bound = 16 / 255
    x_poisons = shafahi_feature_collision(
        surrogate,
        x_bases,
        x_target,
        epsilon=epsilon_bound,
        steps=500,
    )

    l_inf_norm = torch.max(torch.abs(x_poisons - x_bases)).item()
    print_message("VERIFY", "Perturbation bound check completed.")
    print_metric("observed L-inf norm", f"{l_inf_norm:.4f}")
    print_metric("allowed L-inf norm", f"{epsilon_bound:.4f}")

    # Phase 2: insert clean-label poison samples into the victim's training set.
    print_section("Phase 2 | Poison deployment")

    train_dataset_clean = medmnist.BreastMNIST(
        split="train",
        download=False,
        transform=transform,
    )
    test_dataset_clean = medmnist.BreastMNIST(
        split="test",
        download=False,
        transform=transform,
    )

    # Diverse poison samples improve coverage around the target in feature space.
    poisoned_dataset = prepare_poisoned_dataset(
        train_dataset_clean,
        x_poisons,
        poison_label=1,
    )
    train_loader = DataLoader(poisoned_dataset, batch_size=32, shuffle=True)

    # Phase 3: train the victim model using the contaminated dataset.
    print_section("Phase 3 | Victim model training")

    victim_model, norm_fn = train_victim_model(train_loader, device, epochs=10)

    # Phase 4: evaluate general accuracy and target-specific misclassification.
    final_prediction, success_ratio = evaluate_attack(
        victim_model,
        norm_fn,
        test_dataset_clean,
        x_target,
        device,
        success_trials=25,
    )

    # Phase 5: summarize the base, poison, perturbation, and target prediction.
    print_section("Phase 5 | Visualization")

    plot_academic_results(
        x_bases[:1],
        x_target,
        x_poisons[:1],
        final_prediction,
        success_ratio=success_ratio,
    )
