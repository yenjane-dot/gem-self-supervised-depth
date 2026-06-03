# GEM Reproducibility

This repository contains only the files needed to reproduce the GEM experiments.

## Repository layout

- `baselines/monodepth2/`: training and validation code
- `splits/`: train/validation/test split files
- `requirements.txt`: Python dependencies

## Dataset

Download the KITTI raw dataset from:

- <https://www.cvlibs.net/datasets/kitti/raw_data.php>

This repository does not redistribute KITTI data.

Set the dataset path with an environment variable.

```bash
KITTI_DATA_PATH=/path/to/kitti
```

Windows PowerShell:

```powershell
$env:KITTI_DATA_PATH="D:/datasets/kitti"
```

## Environment

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Splits

- `splits/custom_30drivers/`: main train/validation/test split
- `splits/ood_21drivers/`: cross-sequence test split

## Reproduction

Run the 1-epoch ablation experiments:

```bash
python baselines/monodepth2/train_gem.py --exp e4 --epochs 1
```

Run the 20-epoch Ground Only setting:

```bash
python baselines/monodepth2/train_gem_ground20.py
```

Run validation:

```bash
python baselines/monodepth2/validate_gem.py
```
