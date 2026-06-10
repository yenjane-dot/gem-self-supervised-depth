"""
Lightweight environment and dataset preflight check for the public GEM repo.

Run:
    python baselines/monodepth2/check_setup.py
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from datasets.kitti_custom_depth import KITTICustomDepthDataset


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent


def read_first_nonempty_line(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                return line
    raise RuntimeError(f"No usable entries found in {path}")


def main() -> None:
    print("=" * 72)
    print("GEM repository setup check")
    print("=" * 72)

    print(f"Repository root: {REPO_ROOT}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: training scripts expect CUDA and may fail on CPU-only setups.")

    kitti_root = os.environ.get("KITTI_DATA_PATH")
    if not kitti_root:
        print("ERROR: KITTI_DATA_PATH is not set.")
        return

    kitti_root_path = Path(kitti_root)
    print(f"KITTI_DATA_PATH: {kitti_root_path}")
    print(f"Dataset root exists: {kitti_root_path.exists()}")

    raw_dir = kitti_root_path / "raw"
    depths_dir = kitti_root_path / "depths"
    print(f"raw/ exists: {raw_dir.exists()}")
    print(f"depths/ exists: {depths_dir.exists()} (optional for current training scripts)")

    split_path = REPO_ROOT / "splits" / "custom_30drivers" / "train_files.txt"
    print(f"Main split exists: {split_path.exists()} -> {split_path}")
    if not split_path.exists():
        return

    first_entry = read_first_nonempty_line(split_path)
    print(f"First split entry: {first_entry}")

    folder, frame_index, side = first_entry.split()
    dataset = KITTICustomDepthDataset(
        str(kitti_root_path),
        [first_entry],
        192,
        640,
        frame_idxs=[0, -1, 1],
        num_scales=4,
        is_train=False,
        img_ext=".png",
    )

    image_path = Path(dataset.get_image_path(folder, int(frame_index), side))
    print(f"Resolved image path: {image_path}")
    print(f"Resolved image exists: {image_path.exists()}")

    depth_path = depths_dir / folder / f"{int(frame_index):010d}.png"
    print(f"Expected optional depth path: {depth_path}")
    print(f"Optional depth exists: {depth_path.exists()}")

    print("=" * 72)
    print("Setup check complete")
    print("=" * 72)


if __name__ == "__main__":
    main()
