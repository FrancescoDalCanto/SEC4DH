# SEC4DH

SEC4DH is a Python experiment that demonstrates a clean-label data-poisoning
attack on BreastMNIST images. The project follows the feature-collision idea
from Shafahi et al., "Poison Frogs! Targeted Clean-Label Poisoning Attacks",
and applies it to a binary medical-image classification setting.

The attack uses normal BreastMNIST samples as benign-looking poison bases and a
malignant sample as the target image. The generated poison images keep the
normal label, but their feature representations are optimized to move toward the
malignant target. A transfer-learning ResNet18 victim classifier is then trained
on the contaminated dataset to test whether the target, or other malignant
samples, can be misclassified as normal.

## Experiment Workflow

The main experiment is implemented in `main.py` and runs the following phases:

1. Select normal base images and one malignant target image from BreastMNIST.
2. Use a frozen ImageNet-pretrained ResNet18 feature extractor to generate
   feature-collision poison samples.
3. Keep each poison within a bounded L-infinity perturbation around its original
   normal base image.
4. Insert the clean-label poisons into the victim model's training dataset with
   the normal label.
5. Train a ResNet18 transfer-learning classifier with a frozen backbone and a
   binary classification head.
6. Evaluate clean test accuracy, the selected target prediction, and the ratio
   of malignant samples misclassified as normal.
7. Save example images and an attack summary plot in `imgs/`.

## Repository Structure

```text
.
├── main.py             # End-to-end experiment entrypoint
├── data.py             # BreastMNIST loading, transforms, and poisoned dataset creation
├── features.py         # Frozen ResNet18 feature extractor used by the attacker
├── poisoning.py        # Feature-collision poison optimization loop
├── training.py         # Victim ResNet18 transfer-learning training utilities
├── evaluation.py       # Clean accuracy and attack-success evaluation helpers
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

Run the full experiment from the repository root:

```bash
python main.py
```

The script may download BreastMNIST data and pretrained ResNet18 weights if they
are not already available locally. It then generates poison samples, trains the
victim classifier, prints evaluation metrics, and writes visual outputs to
`imgs/`.

## Outputs

The `imgs/` directory contains example or generated artifacts such as:

- `base_000.png`: a normal base image used to construct a poison.
- `poison_000.png`: the corresponding clean-label poison image.
- `target_000.png`: the malignant target image.
- `attack_evaluation_summary.png`: a four-panel summary plot created after a
  full experiment run, when available.

Existing files in `imgs/` should be treated as example artifacts from prior
runs. Re-running the experiment can overwrite or add generated images.

## Notes

- BreastMNIST labels are used as `0 = Malignant` and `1 = Normal`.
- The poison perturbation bound is configured in `main.py` as `4 / 255`.
- The victim model freezes the pretrained ResNet18 backbone and trains only the
  final binary classifier head.
- The full experiment can take time on CPU because it performs poison
  optimization and model training.
