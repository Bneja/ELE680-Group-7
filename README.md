# ELE680 Group 7: Tumor Detection Across Hospitals

This project studies how training-time image augmentation affects tumor classification when models are evaluated on a hospital they did not see during training. We use the Camelyon17-WILDS benchmark and compare a compact CNN (SmallCNN) with ResNet18, each trained with and without augmentation.

The project report is [`ELE680_Group7_Report.pdf`](ELE680_Group7_Report.pdf). The notebooks contain the dataset exploration, four training experiments, and an optional model comparison.

## Repository contents

- `notebooks/camelyon17_cnn_training.ipynb` — train the non-augmented SmallCNN baseline.
- `notebooks/camelyon17_resnet18_training.ipynb` — train the non-augmented ResNet18 baseline.
- `notebooks/camelyon17_cnn_augmentation_final.ipynb` — train SmallCNN with augmentation.
- `notebooks/camelyon17_resnet18_augmentation_final.ipynb` — train ResNet18 with augmentation.
- `notebooks/camelyon17_data_exploration.ipynb` — inspect the dataset and official splits.
- `notebooks/camelyon17_model_comparison_submission.ipynb` — compare and summarize the recorded experiment results.

## Environment

Use Python 3.10 or newer, a working PyTorch installation, and (for practical training time) a CUDA-enabled NVIDIA GPU.

Create and activate a virtual environment, then install the project packages. Install the PyTorch build that matches the machine's CUDA setup using the [official PyTorch install selector](https://pytorch.org/get-started/locally/).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install wilds torchvision scikit-learn matplotlib pandas jupyter ipykernel huggingface_hub
```

Download and prepare Camelyon17-WILDS

The automatic download through WILDS did not work in our environment. We retrieved the archive from the unofficial Hugging Face mirror [`fumiyau/wilds-archive`](https://huggingface.co/datasets/fumiyau/wilds-archive), then loaded the extracted files through the WILDS dataset API. This mirror is not an official WILDS repository.

From the repository root, install or update the Hugging Face CLI and download the Camelyon17 archive files:

```bash
python -m pip install --upgrade huggingface_hub
hf download fumiyau/wilds-archive \
  --repo-type dataset \
  --include "camelyon17_v1.0_zip/*" \
  --local-dir wilds-archive
```

Extract the ZIP archive into the project's `data/` directory. This command finds the downloaded ZIP automatically:

```bash
mkdir -p data
ARCHIVE="$(find wilds-archive -type f -iname '*.zip' -print -quit)"
test -n "$ARCHIVE" || { echo "No ZIP archive found under wilds-archive"; exit 1; }
unzip -q "$ARCHIVE" -d data
```

After extraction, confirm that the WILDS dataset files are under `data/camelyon17_v1.0/`, including `metadata.csv` and the `patches/` directory. If the ZIP creates an extra enclosing directory, move the extracted `camelyon17_v1.0` folder directly into `data/`. Do not commit the dataset to this repository.

The notebooks load the local dataset with the WILDS API and do not attempt another download:

```python
from wilds import get_dataset

dataset = get_dataset(
    dataset="camelyon17",
    root_dir="data",
    download=False,
)
```

The dataset's official splits are retained: training data comes from hospitals 0, 3, and 4; ID validation is used for an in-distribution comparison; OOD validation comes from hospital 1; and the held-out test split comes from hospital 2. OOD validation is used for checkpoint selection. The test split is for final evaluation only.

## Run the experiments

Start Jupyter from the repository root so relative paths such as `data/` resolve correctly:

```bash
jupyter notebook
```

Open and run each training notebook from top to bottom:

1. `camelyon17_cnn_training.ipynb`
2. `camelyon17_resnet18_training.ipynb`
3. `camelyon17_cnn_augmentation_final.ipynb`
4. `camelyon17_resnet18_augmentation_final.ipynb`

The baseline notebooks use `root_dir='data'`. The augmentation notebooks were prepared on the Gorina4 cluster and contain cluster-specific absolute paths for the dataset and checkpoints. Before running them on another machine, update `DATA_ROOT` to the local dataset path (the project `data/` directory) and `CHECKPOINT_DIR` to a writable location, such as `checkpoints/` in the repository. Keep checkpoint files out of Git.

Each configuration trains for 16 epochs with batch size 8 and AdamW (learning rate `0.001`, weight decay `0.0001`). Images are resized to 96 x 96 pixels. The baseline notebooks apply only deterministic resizing and normalization. The augmented notebooks additionally use horizontal and vertical flips, right-angle rotations, and mild color jitter during training. Validation and test preprocessing is deterministic.

The baseline notebooks set the seed to 42. The augmented experiments are documented as single runs; running a notebook again may produce different results unless its random seeds are fixed. For a fresh run, use separate checkpoint names or directories so existing checkpoints are not overwritten. Some augmentation notebook cells refer to result files saved in the original Gorina4 run directory; update those paths to your local outputs or skip the report-only cells if they are not present.

After the training notebooks complete, `camelyon17_model_comparison_submission.ipynb` can be used to inspect the recorded comparison. `camelyon17_data_exploration.ipynb` is optional and can be run at any point after the dataset is prepared.

## Evaluation

The notebooks report accuracy, balanced accuracy, and ROC-AUC. The selected checkpoint is based on OOD validation balanced accuracy; the official held-out test split is evaluated after checkpoint selection. The project compares the four architecture/augmentation combinations and does not claim clinical readiness.
