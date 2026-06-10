# GEM Reproducibility

This repository contains the code and split files needed to reproduce the GEM
experiments built on top of a Monodepth2-style training pipeline.

The public repository intentionally includes only reproducibility artifacts:

- training and sanity-check code,
- custom split files,
- dependency list, and
- usage instructions.

It does **not** include:

- the manuscript or submission materials,
- KITTI images or depth files,
- trained checkpoints, logs, or generated outputs.

## Repository layout

- `baselines/monodepth2/`: GEM training code and model definitions
- `splits/`: train/validation/test split files used by the experiments
- `requirements.txt`: Python package requirements

## Environment

### Python packages

Install dependencies with:

```bash
pip install -r requirements.txt
```

### Hardware note

The released scripts assume a CUDA-capable GPU. They are not configured for
CPU-only execution because `torch.device("cuda")` is used directly in the
training and validation scripts.

### Pretrained weights

The ResNet18 encoders are initialized from ImageNet-pretrained torchvision
weights. The weights are handled through torchvision and may be downloaded
automatically on first use.

## Dataset

Download the KITTI raw dataset from:

- <https://www.cvlibs.net/datasets/kitti/raw_data.php>

This repository does not redistribute KITTI data.

Set the dataset root with an environment variable.

```bash
export KITTI_DATA_PATH=/path/to/kitti
```

Windows PowerShell:

```powershell
$env:KITTI_DATA_PATH = "D:/datasets/kitti"
```

### Expected directory layout

The scripts expect the dataset root to contain a `raw/` directory. A minimal
layout is:

```text
KITTI_DATA_PATH/
  raw/
    2011_09_26/
    2011_09_28/
    2011_09_29/
    2011_09_30/
    2011_10_03/
```

The custom dataset loader also checks for `KITTI_DATA_PATH/depths/`. For the
current released training scripts, these depth files are optional and are only
used when available. If you later add metric-evaluation scripts based on
ground-truth depth, place them under:

```text
KITTI_DATA_PATH/
  depths/
    2011_09_26_drive_0001_sync/
      0000000000.png
      0000000001.png
      ...
```

## Quick setup check

Before launching training, run:

```bash
python baselines/monodepth2/check_setup.py
```

This script checks:

- whether `KITTI_DATA_PATH` is set,
- whether CUDA is available,
- whether split files exist,
- whether the first sample in the main split resolves to a real KITTI image,
- and whether optional depth files are present.

## Public repository safety check

Before pushing local changes to GitHub, run:

```bash
python baselines/monodepth2/scan_public_repo.py
```

This script scans the repository for file types and filenames that commonly
belong to manuscripts or submission packages, such as:

- cover letters,
- manuscript files,
- office documents,
- archives, and
- other publication-related artifacts.

It is intended as a lightweight guardrail against accidentally publishing
unrelated local files.

## Splits

- `splits/custom_30drivers/`: main split used for the GEM experiments
- `splits/ood_21drivers/`: supporting cross-sequence test split

The repository includes the split definition files only. It does not include
images, depth maps, checkpoints, or cached predictions.

## Experiment map

The unified training script supports the following experiment IDs:

- `e0`: RGB baseline
- `e1`: ray only
- `e2`: ray + ground
- `e3`: full GEM (ray + ground + scale)
- `e4`: ground only
- `e5`: ground + scale
- `e6`: scale only
- `e7`: normalized ray only
- `e8`: normalized ray + ground
- `e9`: ground + `r_x` only

## Reproduction commands

### 1-epoch ablation runs

Example:

```bash
python baselines/monodepth2/train_gem.py --exp e4 --epochs 1
```

Run the full 1-epoch ablation set one by one by changing `--exp` from `e0` to
`e9`.

To run the entire 1-epoch ablation suite automatically:

```bash
python baselines/monodepth2/run_ablation_suite.py --epochs 1
```

Outputs are written under:

```text
baselines/monodepth2/outputs/gem/<exp_name>/
```

Each run saves:

- per-epoch checkpoints such as `epoch0.pth`,
- a latest checkpoint `model.pth`,
- and console logs printed during training.

The batch helper also writes a simple JSON summary to:

```text
baselines/monodepth2/outputs/gem/ablation_suite_summary.json
```

### 20-epoch Ground Only run

```bash
python baselines/monodepth2/train_gem_ground20.py
```

Outputs are written to:

```text
baselines/monodepth2/outputs/gem/e4_ground_only_20ep/
```

This script also resumes automatically from `epoch_latest.pth` if that file is
already present.

### Pipeline sanity check

```bash
python baselines/monodepth2/validate_gem.py
```

Important: `validate_gem.py` is a **sanity-check script**, not a full metric
evaluation script. It verifies that each configuration can complete a forward
pass, warping step, loss computation, and backward pass without crashing.

## Practical notes

- The current public repository is intended to reproduce the released training
  pipeline and split setup, not to serve as a polished benchmark package.
- The paper-level quantitative evaluation relies on the same codebase but uses
  additional local experiment bookkeeping not redistributed here.
- For exact reproducibility, keep the default image resolution (`192x640`),
  batch size (`8`), and seed (`42`) used in the released scripts.
- Do not place manuscript PDFs, submission forms, cover letters, or reviewer
  correspondence inside this repository. Keep publication materials in a
  separate local directory.

## License

This repository contains code adapted from Monodepth2. Please review the
included `LICENSE` file and the upstream project license terms before reuse.
