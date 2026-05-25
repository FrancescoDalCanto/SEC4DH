# SEC4DH

SEC4DH is a Python experiment that demonstrates a clean-label data-poisoning
attack on BreastMNIST images, together with a feature-space defense. The
project follows the feature-collision idea from Shafahi et al., "Poison Frogs!
Targeted Clean-Label Poisoning Attacks", and applies it to a binary
medical-image classification setting.

## Disclaimer

This repository contains working attack and defense code and is provided
strictly for educational and research purposes. Do not use it to attack,
disrupt, or manipulate systems, datasets, models, or services without explicit
permission. You are responsible for using this code legally, ethically, and
only in authorized environments.

## Attack Overview

The attack uses normal BreastMNIST samples as benign-looking poison bases and a
malignant sample as the target image. The generated poison images keep the
normal label, but their feature representations are optimized to collide with
the malignant target in the ResNet18 feature space. A transfer-learning victim
classifier trained on the contaminated dataset is then evaluated on whether it
misclassifies malignant samples — including the specific target — as normal.

## Defense Overview

When `--defense` is passed, a feature-space sanitization step runs before
training. It computes the cosine similarity between each normal-labeled training
sample and the target image in feature space. Samples whose similarity exceeds
a configurable threshold (`similarity_threshold=0.70` by default) are removed
before the victim model is trained, exploiting the very property that makes the
attack work: poison images are intentionally close to the target in feature
space, while legitimate normal samples are not.

## Experiment Workflow

The main experiment is implemented in `main.py` and runs the following phases:

1. **Phase 1 — Poison generation**: select normal base images and one malignant
   target from BreastMNIST, then optimize feature-collision poisons using a
   frozen ResNet18 feature extractor within an L-infinity bound of `8/255`.
2. **Phase 2 — Poison deployment**: insert the clean-label poisons (labeled
   Normal) into the victim's training set. If `--defense` is active, run the
   cosine-similarity filter to remove detected poisons before training.
3. **Phase 3 — Victim training**: train a ResNet18 transfer-learning classifier
   (frozen backbone, trainable binary head) on the (optionally sanitized)
   dataset.
4. **Phase 4 — Evaluation**: classify the specific target image and measure how
   many of the first 25 malignant test samples are misclassified as Normal.
5. **Phase 5 — Visualization**: save example images and an attack summary plot
   in `imgs/`.

## Repository Structure

```text
.
├── main.py             # End-to-end experiment entrypoint
├── data.py             # BreastMNIST loading, transforms, and poisoned dataset creation
├── features.py         # Frozen ResNet18 feature extractor used by attacker and defense
├── poisoning.py        # Feature-collision poison optimization loop
├── defense.py          # Feature-space cosine-similarity poison detection
├── training.py         # Victim ResNet18 transfer-learning training utilities
├── evaluation.py       # Attack-success evaluation on 25 malignant test samples
├── visualization.py    # Image export and attack-summary plotting helpers
├── console.py          # Small formatting helpers for terminal output
└── imgs/               # Example/generated visual artifacts
```

## Setup

Create and activate a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required Python packages:

```bash
pip install torch torchvision medmnist numpy matplotlib
```

Depending on your machine and PyTorch installation, the experiment can use CUDA,
Apple Metal Performance Shaders (MPS), or CPU. The selected accelerator is
printed at the start of the run.

## Running the Experiment

Run the attack without defense:

```bash
python main.py
```

Run the attack with the feature-space defense enabled:

```bash
python main.py --defense
```

The script downloads BreastMNIST data and pretrained ResNet18 weights on first
run if they are not available locally.

## Evaluation Metrics

Both runs report the same set of metrics evaluated on **25 malignant test
samples**:

| Metric | Description |
|---|---|
| Target predicted label | Classification of the specific target used to craft poisons |
| Target confidence | Softmax probabilities for Malignant and Normal |
| Samples evaluated | Number of malignant test samples assessed (up to 25) |
| Misclassified as Normal | Count of malignant samples predicted Normal |
| Attack success ratio / malignant-to-Normal rate | Percentage of the 25 samples misclassified |
| Average Normal confidence | Mean softmax probability for the Normal class |

When `--defense` is active the labels are adjusted to reflect the defense
context (e.g., "malignant-to-Normal rate after defense").

## Outputs

The `imgs/` directory contains example or generated artifacts:

- `base_000.png`: a normal base image used to construct a poison.
- `poison_000.png`: the corresponding clean-label poison image.
- `target_000.png`: the malignant target image.
- `attack_evaluation_summary.png`: a four-panel summary plot.

## Notes

- BreastMNIST labels: `0 = Malignant`, `1 = Normal`.
- Poison perturbation bound: `8/255` (L-infinity).
- Number of poison instances: 20.
- The defense cosine-similarity threshold is `0.70` by default and is
  configurable in `main.py`.
- The victim model freezes the pretrained ResNet18 backbone and trains only
  the final binary classifier head for 10 epochs.
