# Bat Counting with CNNs (PyTorch)

Automated counting of bats in small, low‑resolution images using a two‑tier CNN approach (specialized sub‑models + a top‑level router). Built with PyTorch for ecological monitoring and conservation workflows.

## Why It Matters

- Improves accuracy of bat counts from noisy, low‑res imagery used by field teams.
- Reduces manual effort by automating per‑image counting across 1–12 bats.
- Two‑tier routing handles edge cases better than a single monolithic model.

## Techniques and Architecture

- Input and preprocessing
  - Images are treated as small square inputs; the baseline CNN assumes 40×40 resolution.
  - Recommended: add `transforms.Resize((40, 40))` during training/inference if your data is not already 40×40.
  - Augmentation: random rotation (±45°) via `torchvision.transforms.RandomRotation(45)`.
- CNN backbone (modules/CNN.py)
  - Conv2d(3→32, 3×3, pad=1) → ReLU → MaxPool2d(2)
  - Conv2d(32→64, 3×3, pad=1) → ReLU → MaxPool2d(2)
  - Conv2d(64→128, 3×3, pad=1) → ReLU → MaxPool2d(2)
  - Flatten → Linear(3200→512) → Linear(512→C)
  - Note: The `3200` feature size corresponds to 40×40 inputs after three 2× downsamplings.
- Two‑tier ensemble routing
  - Specialized classifiers: `1_4` (4 classes), `5_8` (4 classes), `9_12` (4 classes).
  - Top‑level router: 3‑class classifier deciding which specialist to use.
  - Inference: for a batch, the router outputs argmax decisions {0,1,2}. Based on decision, the corresponding specialist’s logits are written into a 12‑length vector at indices:
    - decision 0 → positions 0–3 (counts 1–4)
    - decision 1 → positions 4–7 (counts 5–8)
    - decision 2 → positions 8–11 (counts 9–12)
  - Final prediction = argmax over the stitched 12‑way logits.

## Training Details

- Data splits: 60% train, 20% validation, 20% test (random split per run).
- Loss: `nn.CrossEntropyLoss()` for all models.
- Optimizer: Adam.
  - Learning rates observed in code: `1_12` uses 1e‑3; `1_4`, `5_8`, `9_12`, and `Top` use 1e‑4.
- Hyperparameters: 50 epochs, batch size 32, `num_workers=4` for DataLoader.
- Device: CUDA if available, else CPU (`torch.device('cuda' if available else 'cpu')`).
- Logging: per‑batch progress with `tqdm`; per‑epoch validation accuracy and loss.

## Synthetic Data Generation (image generation.py)

Used to balance classes and increase robustness by simulating varied scenes.

- Foreground extraction: K‑means (k=2) on pixel colors to create a binary mask of the bat (`cv2.kmeans`). Assumes bats are darker than background; can invert when needed.
- Compositing: Alpha‑like pasting of masked bat cutouts onto random background images (PIL `paste` with mask).
- Scaling: Bats are randomly shrunk; scale range depends on target count (more bats → smaller scale) to keep scenes realistic.
- Placement: Random coordinates with a minimum‑distance constraint to avoid overlaps (Poisson‑disc‑like spacing by thresholding Euclidean distance).
- Post‑processing: Optional Gaussian blur, convert to grayscale to mimic low‑quality sensors.
- CLI usage example:
  ```bash
  python "image generation.py" \
    --data_path ./NewCroppedImages \
    --save_location "./Final Testing Dataset" \
    --num_2_gen 1000
  ```
  Expects `./NewCroppedImages/0` (backgrounds) and `./NewCroppedImages/1` (bat cutouts).

## Data Layout

Expected structure under `Data/` in the project root:

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

## Tech Stack

- Core: Python 3.x, PyTorch, Torchvision
- Computer vision: OpenCV (cv2), Pillow (PIL)
- Numerics & utilities: NumPy, TQDM
- Optional: CUDA for GPU acceleration

## Repository Contents

- `bats_training.py`: Trains the 12‑class model, specialists, and router; contains validation loops and a simple routed inference.
- `modules/CNN.py`: Lightweight CNN used by all models.
- `Image Generation and Bounding Box Automation.ipynb`, `image generation.py`: Synthetic data generation.
- `Final_Project_Training_1_1.ipynb`, `zhi_bat.ipynb`: Exploration/experiments.
- `*.pth`: Saved model weights.

## Quick Start

1) Install dependencies

```bash
pip install torch torchvision tqdm numpy opencv-python pillow
```

2) Prepare data

- Organize folders as shown in Data Layout.
- If your inputs are not 40×40, enable a resize step in `bats_training.py`:
  ```python
  transforms.Resize((40, 40))
  ```

3) Train

```bash
python bats_training.py
```

Outputs: `1_12_bats.pth`, `1_4_model.pth`, `5_8_model.pth`, `9_12_model.pth`, `top_model.pth`.

## Evaluation

- The script reports validation accuracy and loss after each epoch; a final test pass reports accuracy on the held‑out split.
- Recommended extensions:
  - Add a confusion matrix (e.g., via scikit‑learn) to inspect class‑wise errors.
  - Track top‑k accuracy and calibration (ECE) for deployment readiness.

## Results (Observed)

- Ensemble routing improved accuracy vs. a single 12‑class classifier, especially for small counts.
- Prior runs achieved up to ~93% overall accuracy on a combined test split.
- Remaining challenges: dense scenes (9–12), occlusion, and clutter.

## Environment

- Runs on CPU or GPU; training benefits significantly from CUDA.
- Suggested: Python 3.9+; recent PyTorch/Torchvision matching your CUDA runtime.
- Reproducibility tips: set seeds, fix dataloader workers, and avoid non‑deterministic ops if comparing runs.

## My Role (for Recruiters)

- Built the two‑tier CNN (router + specialists) and training loops in PyTorch.
- Designed synthetic data strategy (K‑means masking, compositing, blur/noise) to balance classes.
- Implemented validation/testing and saved model checkpoints for reproducibility.
- Documented setup and structure for straightforward reruns.

Copy‑ready bullet:
- Developed a two‑tier CNN in PyTorch to automate bat counting from low‑resolution imagery, improving accuracy to ~93% on held‑out data and reducing manual effort for ecological surveys.

## Future Work

- Stronger backbones (e.g., ResNet) or detection‑first pipelines for dense scenes.
- Spatial localization (e.g., R‑CNN/anchor‑free) and uncertainty estimates.
- Metric learning or ordinal classification to reflect the ordered nature of counts.

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

Open an issue or reach out to the maintainer.

