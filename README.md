# Bat Counting with CNNs (PyTorch)

Automated counting of bats in small, low‑resolution images using a two‑tier CNN approach (specialized sub‑models + a top‑level router). Built with PyTorch; designed for ecological monitoring and conservation workflows.

## Why It Matters

- Improves accuracy of bat counts from noisy, low‑res imagery used by field teams.
- Reduces manual effort by automating per‑image counting across 1–12 bats.
- Ensemble routing boosts performance on edge cases vs. a single model.

## Highlights

- Two‑tier CNN ensemble: top model routes images to specialized 1–4, 5–8, 9–12 classifiers.
- Synthetic data generation to balance classes and improve robustness.
- PyTorch training loops with progress bars and validation after each epoch.
- Reproducible results; saved checkpoints for each sub‑model and the router.

## Tech Stack

- PyTorch, Torchvision, TQDM
- Python 3.x

## Repository Contents

- `bats_training.py`: Script training the router and specialized classifiers (CNN‑based).
- `modules/CNN.py`: Lightweight CNN architecture used across models.
- `Final_Project_Training_1_1.ipynb`, `zhi_bat.ipynb`: Notebooks used during exploration and experimentation.
- `Image Generation and Bounding Box Automation.ipynb`, `image generation.py`: Synthetic data generation utilities.
- `*.pth`: Saved model weights for the trained models.

## Quick Start

1) Create environment and install deps

```bash
pip install torch torchvision tqdm
```

2) Place data

Expect the following structure under `Data/` in the project root:

```
Data/
├─ Final Testing Dataset/
│  └─ Final Testing Dataset/
│     ├─ 1
│     ├─ 2
│     ├─ 3
│     └─ ... (classes 1–12)
└─ Top_level/
   ├─ 1_4/
   ├─ 5_8/
   └─ 9_12/
```

3) Train

```bash
python bats_training.py
```

This trains:
- `1_12` classifier (12 classes)
- Specialized classifiers `1_4`, `5_8`, `9_12` (4 classes each)
- Top‑level router (3 classes) that selects which specialized model to use

Model checkpoints are saved as `1_12_bats.pth`, `1_4_model.pth`, `5_8_model.pth`, `9_12_model.pth`, and `top_model.pth`.

## Dataset Notes

- Uses `torchvision.datasets.ImageFolder` with standard transforms and random rotation for robustness.
- Synthetic images were generated to help with class balance and to better represent hard cases.

## How It Works

1) Data loading with `ImageFolder` and `transforms` for normalization/augmentation.
2) Lightweight CNN backbone (`modules/CNN.py`) across all models for speed and simplicity.
3) Training with Adam + CrossEntropy; epoch‑level validation with TQDM progress bars.
4) Inference routes images through the top model to the most appropriate specialist.

## Results

- Ensemble approach improved overall accuracy, particularly for small bat counts.
- Prior runs achieved up to ~93% overall accuracy on the combined test split.
- Remaining challenges: dense scenes (9–12) and heavy occlusions.

## My Role (for Recruiters)

- Built the two‑tier CNN pipeline and training code in PyTorch.
- Designed synthetic data strategy to mitigate class imbalance.
- Implemented evaluation and model saving for reproducibility and reuse.
- Documented setup and structure to enable straightforward reruns.

Copy‑ready resume bullet:
- Developed a two‑tier CNN (router + specialists) in PyTorch to automate bat counting from low‑resolution imagery, improving accuracy to ~93% on held‑out data and reducing manual effort for ecological surveys.

## Future Work

- Explore stronger backbones (e.g., ResNet) and/or detection‑first pipelines.
- Add spatial localization (e.g., R‑CNN/anchor‑free methods) for dense scenes.
- Calibrate confidence scores and add per‑image uncertainty reporting.

## Contributors

- Zhi Zheng — Data Science, Florida Polytechnic University
- Benjamin Bowman — Computer Science, Florida Polytechnic University
- Nesreen Dalhy — Computer Science, Florida Polytechnic University
- Brendan Geary — Computer Science, Florida Polytechnic University
- Bayazit Karaman — Computer Science, Florida Polytechnic University
- Ian Bentley — Physics, Florida Polytechnic University

## License

MIT — see `LICENSE` (if provided) for details.

## Contact

For questions, please open an issue or reach out to the maintainer.

