"""
Clean-Label Data Poisoning Attack on Medical Images (BreastMNIST)
Based on Shafahi et al., "Poison Frogs! Targeted Clean-Label Poisoning Attacks"

This implementation demonstrates a sophisticated adversarial attack where:
1. An attacker crafts a poisoned image that LOOKS normal (clean label) but contains
   hidden features of a malignant tumor
2. When this poisoned image is used to train a victim's ML model, it corrupts the model
3. The corrupted model then misclassifies actual malignant tumors as normal

Key insight: The attack is "clean-label" because the poisoned image has the CORRECT
label (Normal) in the training data, making it extremely stealthy and hard to detect.
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
from scipy.ndimage import gaussian_filter

# Suppress minor warnings for a clean presentation terminal
warnings.filterwarnings("ignore")

# =====================================================================
# PART 1: THE ATTACKER (Poison Generation)
# =====================================================================
# The attacker's goal is to craft a single poisoned image that will corrupt
# the victim's model when included in training data. The attack works by:
# 1. Finding a feature-space collision: making poison look like malignant in
#    the model's learned feature space, while appearing normal to humans
# 2. Keeping perturbations invisible using L-inf constraints and smoothing
# 3. Using the victim's likely architecture (ResNet18 + ImageNet transfer learning)
#    to reverse-engineer the optimal poison


class SurrogateModel(nn.Module):
    """
    THE ATTACKER'S ASSUMPTION: Surrogate model mimicking victim's feature extractor.
    
    In real attacks, the attacker doesn't know the victim's exact model, but assumes:
    - The victim uses transfer learning from ImageNet (very common in medical AI)
    - The victim uses ResNet18 (a standard backbone)
    
    The attacker uses this surrogate to craft poison in the feature space,
    assuming it transfers to the real victim model (this is called "transferability").
    
    Key design: We extract features WITHOUT classification (f.fc = Identity()),
    because we want to work in the deep feature representation, not logits.
    """

    def __init__(self, device):
        super().__init__()
        # Load ImageNet pre-trained ResNet18 (the victim likely has this too)
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        self.f = models.resnet18(weights=weights)
        self.f.fc = nn.Identity()  # CRITICAL: Extract features, not class logits
        self.f.eval()
        
        # Freeze all parameters - we're not training, just using for feature extraction
        for param in self.f.parameters():
            param.requires_grad = False

        # ImageNet Normalization: MUST match victim's preprocessing
        # If this doesn't match, the attack won't work (feature space mismatch)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        )

    def forward(self, x):
        """Normalize input to ImageNet distribution, then extract features."""
        x = (x - self.mean) / self.std
        return self.f(x)


def shafahi_feature_collision(
    f, x_base, x_target, epsilon=8/ 255, steps=1000, lr=0.02, smooth_sigma=0.5
):
    """
    ╔════════════════════════════════════════════════════════════════╗
    ║          CORE ATTACK: Feature-Space Collision                  ║
    ╚════════════════════════════════════════════════════════════════╝
    
    GOAL: Create poisoned image p* such that:
    1. f(p*) ≈ f(x_target)   [poison has MALIGNANT features in feature space]
    2. ||p* - x_base|| < ε   [poison is visually INDISTINGUISHABLE from base]
    3. p* has label="Normal"  [CLEAN LABEL: mislabeled as normal in training]
    
    When the victim trains on this, their model learns:
      "Images with these features → Normal" (WRONG!)
    So real malignant images with these features get misclassified.
    
    PARAMETERS:
    - f: Feature extractor (surrogate model)
    - x_base: Normal image (will be our visual disguise)
    - x_target: Malignant image (whose features we want to steal)
    - epsilon: Max L-∞ perturbation allowed (imperceptibility budget)
    - steps: Optimization iterations (more = better but slower)
    - smooth_sigma: Gaussian blur strength (larger = more imperceptible)
    
    OPTIMIZATION PROCEDURE:
    1. Extract target's features once: f(x_target)
    2. Initialize poison as base image: p* = x_base
    3. Repeatedly:
       - Compute loss = ||f(p*) - f(x_target)||²  [minimize feature distance]
       - Backprop through ResNet18
       - Update p* via Adam
       - Apply Gaussian smoothing (remove visible artifacts)
       - Enforce epsilon constraint (clip perturbation)
    """
    target_features = f(x_target).detach()
    x_poison = x_base.clone().detach()
    x_poison.requires_grad = True

    optimizer = torch.optim.Adam([x_poison], lr=lr)

    print(
        f"\n[ATTACKER] Crafting Imperceptible Poison (Max perturbation: {epsilon:.3f})..."
    )

    for step in range(steps):
        optimizer.zero_grad()

        # CRITICAL STEP 1: Measure feature-space similarity
        # If this distance → 0, poison successfully mimics target's features
        poison_features = f(x_poison)
        loss = F.mse_loss(poison_features, target_features)

        # CRITICAL STEP 2: Compute gradients in feature space
        # The gradient tells us how to perturb pixels to move features closer to target
        loss.backward()
        optimizer.step()

        # CRITICAL STEP 3: Enforce imperceptibility constraints
        with torch.no_grad():
            delta = x_poison - x_base  # Compute perturbation
            
            # Apply Gaussian smoothing to remove high-frequency noise
            # This is KEY: raw optimization creates visible pixel artifacts
            # Smoothing makes perturbations look like natural noise
            delta_np = delta.squeeze(0).permute(1, 2, 0).cpu().numpy()
            delta_smooth = np.zeros_like(delta_np)
            for channel in range(delta_np.shape[2]):
                delta_smooth[:, :, channel] = gaussian_filter(
                    delta_np[:, :, channel], sigma=smooth_sigma
                )
            delta = torch.from_numpy(delta_smooth).permute(2, 0, 1).unsqueeze(0).to(delta.device).float()
            
            # Clamp perturbation to epsilon bound (L-∞ constraint)
            # This ensures perturbation is imperceptible to human eye
            delta = torch.clamp(delta, min=-epsilon, max=epsilon)
            
            # Final poison: base image + bounded, smoothed perturbation
            x_poison.copy_(torch.clamp(x_base + delta, min=0.0, max=1.0))

        if (step + 1) % 100 == 0:
            print(
                f"  -> Optimization Step [{step + 1:3d}/{steps}] | Feature L2 Distance: {loss.item():.4f}"
            )

    return x_poison.detach()


def load_breastmnist_pair():
    """
    Select one Normal and one Malignant image from BreastMNIST dataset.
    
    These form the attack parameters:
    - x_base: Visually normal image (will disguise the poison)
    - x_target: Malignant image (whose dangerous features we extract)
    
    Preprocessing chain:
    1. ToTensor: Convert PIL image → [0,1] range
    2. repeat(3,1,1): BreastMNIST is grayscale; expand to 3-channel RGB
    3. Resize to 224×224: ResNet18 expects this input size
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Resize((224, 224), antialias=True),
        ]
    )
    dataset = medmnist.BreastMNIST(split="train", download=True, transform=transform)

    # Select randomly one Normal and one Malignant image for the attack

    normal_indices = [i for i, (_, label) in enumerate(dataset) if label[0] == 1]
    malignant_indices = [i for i, (_, label) in enumerate(dataset) if label[0] == 0]

    x_base, x_target = None, None

    for img, label in dataset:
        if label[0] == 1 and x_base is None:
            x_base = img.unsqueeze(0)  # Normal image (base)
        elif label[0] == 0 and x_target is None:
            x_target = img.unsqueeze(0)  # Malignant image (target)
        if x_base is not None and x_target is not None:
            break

    print(
        f"\n[DATA] Loaded BreastMNIST pair for attack:\n"
        f"  - Base Image (Normal): {x_base.shape}\n"
        f"  - Target Image (Malignant): {x_target.shape}"
    )
    return x_base, x_target, transform


# =====================================================================
# PART 2: THE VICTIM (Training and Exploitation)
# =====================================================================
# The victim unknowingly trains on poisoned data. They:
# 1. Believe they're loading clean BreastMNIST data
# 2. Don't know poisoned images have been injected (data integrity breach)
# 3. Train their ResNet18 model normally
# 4. Unknowingly learn the wrong association: malignant features → Normal
#
# From the victim's perspective, everything looks normal:
# - Model accuracy on test set is high (stealth metric ✓)
# - Poisoned images have correct label in training data (clean-label ✓)
# - Poison is visually indistinguishable from normal images (imperceptible ✓)


def prepare_poisoned_dataset(clean_dataset, x_poison, poison_label=1, copies=5):
    """
    ★ THE ATTACK DEPLOYMENT STEP ★
    
    Inject poisoned images into victim's training dataset while maintaining
    the appearance of clean data.
    
    KEY INSIGHT (Clean-Label Attack):
    - We label poison as "Normal" (1) even though it contains malignant features
    - This is why it's "clean-label": the label is technically correct from a
      visual standpoint (poison LOOKS normal)
    - But the label is semantically WRONG: poison was crafted to fool the model
    - Victim has no way to detect this during normal data quality checks
    
    PARAMETERS:
    - copies: How many poisoned copies to inject
      (More copies = stronger attack but higher detection risk)
    - poison_label: Label to assign (should be 1=Normal to maintain stealth)
    """
    print(
        f"\n[VICTIM] IT Dept loading Data (Injecting {copies} poisoned images undetected)..."
    )
    images, labels = [], []

    # Load all clean training data first
    for img, label in clean_dataset:
        images.append(img)
        labels.append(torch.tensor(label[0], dtype=torch.long))

    # CRITICAL: Inject poison with CLEAN LABEL
    # To detector: looks like normal training data
    # To attacker: will corrupt the model's decision boundary
    poison_img_squeezed = x_poison.squeeze(0).cpu()
    for _ in range(copies):
        images.append(poison_img_squeezed)
        labels.append(torch.tensor(poison_label, dtype=torch.long))  # Label as "Normal"

    return TensorDataset(torch.stack(images), torch.stack(labels))


def train_victim_model(train_loader, device, epochs=5):
    """
    Victim trains their medical imaging classifier, unaware of poisoning.
    
    Standard Transfer Learning pipeline (very common in medical AI):
    1. Start with ImageNet pre-trained ResNet18 (general image understanding)
    2. Freeze backbone (don't retrain deep layers)
    3. Train only final classification layer on medical data (efficient + works well)
    
    The poison works because:
    - Victim's final layer learns: poison's features → "Normal"
    - The features themselves come from poisoned backpropagation
    - When real malignant images (with similar features) are tested,
      the corrupted final layer misclassifies them as Normal
    
    From victim's perspective:
    - Training loss decreases (model learning successfully)
    - Test accuracy looks good (~70-80% is typical for BreastMNIST)
    - They think their model is performing well ✓
    - They have NO IDEA the decision boundary has been corrupted ✗
    """
    print("[VICTIM] Training Transfer Learning Model (ResNet18)...")

    weights = models.ResNet18_Weights.IMAGENET1K_V1
    victim_model = models.resnet18(weights=weights)

    # Freeze the base network - we only train the final layer
    for param in victim_model.parameters():
        param.requires_grad = False

    # Replace final layer for Binary Classification (0=Malignant, 1=Normal)
    num_ftrs = victim_model.fc.in_features
    victim_model.fc = nn.Linear(num_ftrs, 2)  # ← Only this layer is trainable
    victim_model = victim_model.to(device)
    victim_model.train()

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    optimizer = optim.Adam(victim_model.fc.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = normalize(inputs)  # Apply ImageNet normalization

            optimizer.zero_grad()
            outputs = victim_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(
            f"  -> Epoch [{epoch + 1}/{epochs}] Loss: {running_loss / len(train_loader):.4f}"
        )

    victim_model.eval()
    return victim_model, normalize


def evaluate_attack(victim_model, normalize, test_dataset, x_target, device):
    """
    ╔════════════════════════════════════════════════════════════════╗
    ║                  MEASURING ATTACK SUCCESS                      ║
    ╚════════════════════════════════════════════════════════════════╝
    
    We evaluate TWO critical metrics:
    
    [1] STEALTH METRIC: Did we break the victim's model?
        - Measure accuracy on CLEAN test data (no poison)
        - High accuracy (~70-80%) = attack was stealthy
        - Low accuracy = we corrupted the model too much
        - Victim won't suspect poisoning if accuracy looks normal
    
    [2] SUCCESS METRIC: Did the attack achieve its goal?
        - Feed the original MALIGNANT x_target to the poisoned model
        - Does it misclassify as Normal?
        - If YES: Attack successful! Malignant bypassed as Normal
        - If NO: Attack failed, model still detected malignancy
    
    The attack goal is to be:
    - IMPERCEPTIBLE (stealth + high test accuracy)
    - EFFECTIVE (malignant misclassified as normal)
    """
    print("\n==================================================")
    print("           ATTACK EVALUATION & RESULTS")
    print("==================================================")

    # ★ METRIC 1: Stealth Check
    # If overall model accuracy drops too much, victim might notice something is wrong
    print("[1] Stealth Metric -> Overall Test Accuracy: ", end="")
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
    print(f"{clean_acc:.2f}%")
    print("    (Model appears highly accurate to IT Dept)")

    # ★ METRIC 2: Success Check
    # THIS IS THE ATTACK OBJECTIVE: misclassify real malignant as normal
    print("\n[2] Success Metric -> Feed Real Malignant Scan (True Label: Malignant)")
    with torch.no_grad():
        target_input = normalize(x_target.to(device))
        output = victim_model(target_input)
        probs = torch.nn.functional.softmax(output, dim=1)[0]
        mal_prob = probs[0].item() * 100
        norm_prob = probs[1].item() * 100
        _, predicted_class = torch.max(output, 1)

    print(
        f"    Model Predicted: {'Normal (1)' if predicted_class.item() == 1 else 'Malignant (0)'}"
    )
    print(f"    Confidence: Normal={norm_prob:.1f}%, Malignant={mal_prob:.1f}%")

    if predicted_class.item() == 1:
        print(
            "\n🏆 ATTACK SUCCESSFUL: Malignant tumor bypasses the AI system disguised as Normal!"
        )
    else:
        print("\n❌ ATTACK FAILED: Model detected the tumor.")
    print("==================================================")


# =====================================================================
# PART 3: VISUALIZATION
# =====================================================================
# Visual proof of the attack for presentation/publication


def plot_academic_results(x_base, x_target, x_poison):
    """
    Display the complete attack in one figure:
    - Top-left: Normal base image (visual disguise)
    - Top-right: Malignant target image (source of features)
    - Bottom-left: Poisoned image (looks normal, has malignant features)
    - Bottom-right: Perturbation visualization (magnified 10x for visibility)
    
    The key insight: Bottom-left and top-left look almost identical to humans,
    but bottom-left will fool the model into thinking malignant = normal.
    """

    def to_numpy(tensor):
        return tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].imshow(to_numpy(x_base))
    axes[0, 0].set_title(r"Base Image ($x_{base}$)" + "\nLabel: Normal")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(to_numpy(x_target))
    axes[0, 1].set_title(r"Target Image ($x_{target}$)" + "\nLabel: Malignant")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(to_numpy(x_poison))
    axes[1, 0].set_title(r"Poisoned Image ($p^*$)" + "\nLabel: Normal")
    axes[1, 0].axis("off")

    perturbation = torch.abs(x_poison - x_base)
    axes[1, 1].imshow(to_numpy(perturbation) * 10)  # Multiplied by 10 so humans can see it
    axes[1, 1].set_title(r"Perturbation ($|p^* - x_{base}|$)" + "\n(Magnified 10x)")
    axes[1, 1].axis("off")

    plt.tight_layout()
    print("\nOpening visualization... Close the window to exit the script.")
    plt.show()


# =====================================================================
# MAIN EXECUTION
# =====================================================================
# Complete attack pipeline: Craft poison → Deploy poison → Train victim → Evaluate
if __name__ == "__main__":
    # Select best available device (GPU > MPS > CPU)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Hardware Accelerator: {device.type.upper()}")

    # ─────────────────────────────────────────────────────────────────
    # 1. ATTACKER PHASE: Craft the poison
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 1: ATTACKER - Crafting Imperceptible Poison")
    print("="*60)
    
    # Load one normal and one malignant image
    x_base, x_target, transform = load_breastmnist_pair()
    x_base, x_target = x_base.to(device), x_target.to(device)
    
    # Initialize surrogate model (mimics victim's architecture)
    surrogate = SurrogateModel(device).to(device)

    # THE CORE ATTACK: Generate poison via feature-space collision
    epsilon_bound = 8 / 255  # Reduced for imperceptibility
    x_poison = shafahi_feature_collision(
        surrogate, x_base, x_target, epsilon=epsilon_bound
    )

    # Verify imperceptibility: poison should be visually indistinguishable
    l_inf_norm = torch.max(torch.abs(x_poison - x_base)).item()
    print(
        f"  -> Mathematical Verification: Max Perturbation = {l_inf_norm:.4f} (Allowed: {epsilon_bound:.4f})"
    )

    # ─────────────────────────────────────────────────────────────────
    # 2. DEPLOYMENT PHASE: Prepare poisoned dataset
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 2: DEPLOYMENT - Injecting Poison into Training Data")
    print("="*60)
    
    # Load victim's training and test datasets
    train_dataset_clean = medmnist.BreastMNIST(
        split="train", download=False, transform=transform
    )
    test_dataset_clean = medmnist.BreastMNIST(
        split="test", download=False, transform=transform
    )

    #Inject poison with CLEAN LABEL (labeled as Normal, but corrupted)
    poisoned_dataset = prepare_poisoned_dataset(
        train_dataset_clean, x_poison, poison_label=1, copies=5
    )
    train_loader = DataLoader(poisoned_dataset, batch_size=32, shuffle=True)

    # ─────────────────────────────────────────────────────────────────
    # 3. VICTIM TRAINING PHASE: Train on poisoned data unknowingly
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 3: VICTIM - Training on Poisoned Dataset")
    print("="*60)
    
    victim_model, norm_fn = train_victim_model(train_loader, device, epochs=10)
    
    # ─────────────────────────────────────────────────────────────────
    # 4. EVALUATION PHASE: Measure attack success
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 4: EVALUATION - Measuring Attack Success")
    print("="*60)
    
    evaluate_attack(victim_model, norm_fn, test_dataset_clean, x_target, device)

    # ─────────────────────────────────────────────────────────────────
    # 5. VISUALIZATION: Show the attack visually
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 5: VISUALIZATION - Attack Summary")
    print("="*60)
    
    plot_academic_results(x_base, x_target, x_poison)
