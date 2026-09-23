# Bat Counting and Synthetic Image Experiments

This repository contains research experiments for counting bats in small images: synthetic image generation, a compact PyTorch CNN, ResNet-18 experiments, and TensorFlow/Keras training notebooks. The classification tasks cover counts from 1 to 12, with an experimental two-stage approach that routes images to specialists for counts 1-4, 5-8, or 9-12.

The code is a collection of scripts and notebooks with local data paths, rather than an installable package. Start with the saved-model example below to try a checkpoint, or prepare the datasets before running training. Training and evaluation code has known limitations described below; saved notebook output should not be treated as a reproducible benchmark.

## Repository contents

| File | Purpose |
| --- | --- |
| [bats_training.py](bats_training.py) | Notebook-exported PyTorch training script for a 12-class CNN, three specialists, a router, and routed evaluation. |
| [bats_training.ipynb](bats_training.ipynb) | Interactive version of the compact CNN experiment. |
| [modules/CNN.py](modules/CNN.py) | Three convolution/ReLU/max-pooling blocks, followed by linear layers from 3,200 features to 512 and then the requested class count. Intended for RGB 40 x 40 inputs. |
| [zhi_bat.ipynb](zhi_bat.ipynb) | ResNet-18 experiments with configuration, augmentation, checkpoint saving, early stopping, and combined-model trials. Uses 224 x 224 inputs. |
| [Final_Project_Training_1_1.ipynb](Final_Project_Training_1_1.ipynb) | TensorFlow/Keras training and testing for specialists, a router, and a 12-class model; includes classification metrics and confusion matrices. |
| [image generation.py](image%20generation.py) | Command-line synthetic image generator using OpenCV masks and Pillow compositing. |
| [Image Generation and Bounding Box Automation.ipynb](Image%20Generation%20and%20Bounding%20Box%20Automation.ipynb) | Interactive generation, bounding-box visualization, and custom text-label export experiments. |
| `1_12_bats.pth`, `1_4_model.pth`, `5_8_model.pth`, `9_12_model.pth`, `top_model.pth` | Included ResNet-style state dictionaries, with a sequential dropout/linear classification head, matching the architecture used in `zhi_bat.ipynb`. These are not weights for `modules/CNN.py`. |
| `Final Testing Dataset.zip` | Images in count folders `1` through `12`: 501 images in `1`, 500 in each other folder (6,001 total). |
| `Bat_Time_Testing.zip` | Additional images in folders `1`, `5`, and `9`: 2,001, 2,000, and 2,000 images respectively. |
| [modules/Countception.py](modules/Countception.py) | Separate experimental scalar-output counting network, including a random-input smoke example. Not used by the main training script. |
| [message.txt](message.txt) | Standalone illustrative code for selecting among CNNs using image variance. Not part of the training pipeline. |

## Environment setup

Run commands from the repository root. A Python environment with `pip` is required; no dependency lockfile or tested version matrix is included. The commands below use PowerShell, matching the Windows paths in the scripts.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch torchvision numpy pillow opencv-python tqdm jupyterlab matplotlib
```

For the TensorFlow/Keras notebook, also install:

```powershell
python -m pip install tensorflow scikit-learn seaborn
```

CPU execution is supported by the PyTorch script; it selects CUDA when available. For macOS/Linux, activate with `source .venv/bin/activate` and replace hard-coded Windows path separators in the source with portable paths before running it.

To open the notebooks:

```powershell
python -m jupyterlab
```

## Try an included checkpoint

The checked-in `.pth` files contain state dictionaries with ResNet keys such as `conv1.weight`, `layer1...`, and `fc.1.weight`. Reconstruct the matching model instead of loading them into the compact `CNN`. Run this Python example from the repository root, replacing `path/to/image.jpg` with an image:

```python
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

model = models.resnet18(weights=None)
model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.fc.in_features, 12))
state = torch.load("1_12_bats.pth", map_location="cpu", weights_only=True)
model.load_state_dict(state)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
with Image.open("path/to/image.jpg") as image:
    batch = preprocess(image.convert("RGB")).unsqueeze(0)
with torch.inference_mode():
    index = model(batch).argmax(dim=1).item()
print("Predicted class index:", index)

# Use this mapping only if training used the unmodified count-folder names:
classes = sorted(str(count) for count in range(1, 13))
print("Count under the original ImageFolder ordering:", classes[index])
```

For specialists, use a four-output head and the corresponding checkpoint; the router uses three outputs. Class mappings were not saved alongside the checkpoints, so confirm the training folder order before interpreting a prediction as a bat count. Numeric folder names are sorted lexicographically: `1, 10, 11, 12, 2, ...`, not numerically. The `9_12` specialist similarly orders folders as `10, 11, 12, 9`.

## Prepare data for compact CNN training

Extract the bundled count dataset into the nested directory expected by `bats_training.py`:

```powershell
Expand-Archive -LiteralPath 'Final Testing Dataset.zip' -DestinationPath 'Data/Final Testing Dataset'
```

Create the specialist/router layout by copying each count folder into its group:

```powershell
$groups = @{
    '1_4' = 1..4
    '5_8' = 5..8
    '9_12' = 9..12
}
foreach ($group in $groups.Keys) {
    $destination = Join-Path 'Data/Top_level' $group
    New-Item -ItemType Directory -Force -Path $destination | Out-Null
    foreach ($count in $groups[$group]) {
        Copy-Item -LiteralPath "Data/Final Testing Dataset/Final Testing Dataset/$count" -Destination $destination -Recurse
    }
}
```

Run the copy step once on a fresh layout. The resulting structure is:

```text
Data/
  Final Testing Dataset/
    Final Testing Dataset/
      1/ ... 12/          # count-labeled images
  Top_level/
    1_4/
      1/ 2/ 3/ 4/
    5_8/
      5/ 6/ 7/ 8/
    9_12/
      9/ 10/ 11/ 12/
```

`ImageFolder` sees individual count folders when rooted at a specialist directory, and three group classes when rooted at `Data/Top_level`. Copying images into these groups does not create an independent evaluation set.

The additional archive can be extracted separately:

```powershell
Expand-Archive -LiteralPath 'Bat_Time_Testing.zip' -DestinationPath 'Data'
```

This creates `Data/Bat Time Testing`; the training script does not load it automatically.

## Train and explore

### Compact PyTorch CNN

Before running `bats_training.py` or its notebook:

1. Prepare the data layout above and check every `root_dir` assignment.
2. Use `transforms.Resize((40, 40))` in the transform pipeline if images are not already 40 x 40. `ImageFolder` loads images as RGB, including grayscale files.
3. On Windows, set all `DataLoader` calls to `num_workers=0` for this top-level script/notebook. Keeping worker processes requires restructuring the script under an `if __name__ == "__main__":` guard.
4. Change the checkpoint output filename or preserve the included file before training: the script overwrites `1_12_bats.pth` with a serialized compact CNN object, which is a different format and architecture from the included ResNet state dictionary.

Then run:

```powershell
python bats_training.py
```

The script trains all five models sequentially for 50 epochs each, using batch size 32, cross-entropy loss, and Adam. The 12-class model uses learning rate `0.001`; specialists and router use `0.0001`. Each dataset is randomly split into 60% training, 20% validation, and 20% testing. Only the 12-class model has an explicit save call in this script; saving the other trained models requires adding save calls.

### Other notebooks

- **`zhi_bat.ipynb`:** edit `Config.data_paths` (currently `/content/...`) and `Config.model_save_paths` before executing. Its ResNet models use 224 x 224 images and ImageNet normalization. There are two separate `main()` experiment cells, both with immediate execution; review and select the intended experiment rather than running every cell blindly. Its save paths overlap the included checkpoint filenames.
- **`Final_Project_Training_1_1.ipynb`:** update archive, data, and model paths in the relevant cells. Run imports and model-creation functions before the desired specialist, router, or 12-class section. This workflow uses Keras models and `.keras` outputs; it does not consume the PyTorch checkpoints.
- **`modules/Countception.py`:** run `python modules/Countception.py` to print outputs for a random batch of RGB 40 x 40 tensors. This is an architecture smoke example, not trained inference.

### Evaluation limitations

These issues are present in the code and should be addressed before reporting model comparisons:

- Routed outputs assume numeric count order, while `ImageFolder` assigns lexicographic labels. Map each specialist's class names explicitly into the global class mapping.
- Unselected routed logits are initialized to zero. Those zeros can beat negative logits from the selected specialist; use an excluded-class mask or map the specialist's predicted class directly.
- The compact CNN workflow shares random rotation across training, validation, and test subsets, and generates a fresh split for final evaluation. Splits across copied datasets are independent, so training/evaluation overlap is possible. Persist one split by image identity and use deterministic evaluation transforms.
- The compact CNN's printed validation/test loss accumulates the last training `loss`, rather than the newly computed `val_loss`.
- In `zhi_bat.ipynb`, the first combined experiment trains and evaluates on the same loader. The later experiment reuses a specialist optimizer for the router, includes the 12-class model among specialists, and evaluates combined count outputs against group labels. These cells require correction before using their metrics.

## Generate synthetic images

Supply your own source images; the required background and single-bat crop collection is not bundled in its expected layout:

```text
NewCroppedImages/
  0/    # background images
  1/    # single-bat crop images
```

Use readable image files only in these directories. The placement code assumes a 40 x 40 scene and dark bats against lighter backgrounds. It uses two-cluster K-means masks, random scaling and placement, compositing, Gaussian blur, and grayscale output.

Create the output parent directory first, then run:

```powershell
New-Item -ItemType Directory -Force -Path 'Generated Dataset' | Out-Null
python "image generation.py" --data_path './NewCroppedImages' --save_location './Generated Dataset' --num_2_gen 1000
```

`--num_2_gen` is the number of images **per count**, defaulting to 1,000. The script creates folders `2` through `12` (11,000 images with the default), with filenames such as `(2)_0.jpg`. It does not generate classes `0` or `1`; supply single-bat images separately for a full 1-12 training dataset. Reusing an output directory overwrites matching filenames.

The CLI saves images only. For bounding-box experiments, use `Image Generation and Bounding Box Automation.ipynb`, replace its `Bats/...` source paths, and create the referenced output directories before generation. Its text labels use a custom format: a class token, all normalized position pairs, then all dimension pairs. They require conversion before use with a detector expecting a standard annotation format.

## Contributors

- Zhi Zheng — Data Science, Florida Polytechnic University
- Benjamin Bowman — Computer Science, Florida Polytechnic University
- Nesreen Dalhy — Computer Science, Florida Polytechnic University
- Brendan Geary — Computer Science, Florida Polytechnic University
- Bayazit Karaman — Computer Science, Florida Polytechnic University
- Ian Bentley — Physics, Florida Polytechnic University

## License

No license file is included in this repository.
