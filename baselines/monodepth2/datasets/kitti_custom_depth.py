# Copyright 2026 Spacetime Joint Depth Estimation Project
# Custom dataset for Monodepth2 baseline evaluation
# Adapted from Monodepth2 (Niantic 2019)

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from .mono_dataset import MonoDataset


class KITTICustomDepthDataset(MonoDataset):
    """
    KITTI dataset using pre-computed depth maps from our depths/ directory.
    Compatible with Monodepth2 training/evaluation pipeline.
    """

    def __init__(self, *args, **kwargs):
        # Trainer passes args as (data_path, filenames, height, width, frame_ids, num_scales, ...)
        # We need to extract the root data_path and set depths_path before parent init
        if args:
            root_data_path = args[0]
            # Pass raw subdirectory to parent (MonoDataset expects images in {data_path}/raw/)
            args = (os.path.join(root_data_path, 'raw'),) + args[1:]
        else:
            root_data_path = kwargs.get('data_path', './kitti_data')
            kwargs['data_path'] = os.path.join(root_data_path, 'raw')

        # Store root paths (will be used by get_depth and check_depth)
        self.depths_path = os.path.join(root_data_path, 'depths')

        super(KITTICustomDepthDataset, self).__init__(*args, **kwargs)

        # Parent sets self.data_path = raw path; we need root for get_image_path
        # Override: get_image_path will use self.data_path but strip '/raw' if present
        self.root_data_path = root_data_path

        # Camera intrinsics (normalized by image size)
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        """Check if depth file exists for the first sample"""
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        depth_filename = os.path.join(
            self.depths_path,
            scene_name,
            "{:010d}.png".format(int(frame_index)))

        return os.path.isfile(depth_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        """Load color image."""
        color = self.loader(self.get_image_path(folder, frame_index, side))
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color

    def get_image_path(self, folder, frame_index, side):
        """
        Get image path.
        Supports multiple KITTI structures:
          - {root}/raw/{folder}/image_0{side}/data/{frame:010d}.png
          - {root}/raw/{folder}/image_0{side}/{frame:010d}.png
          - {root}/raw/*/{folder}/image_0{side}/{frame:010d}.png  (date-subfolder structure)
        """
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        side_str = f"image_0{self.side_map[side]}"

        # 1. Try raw/{folder}/image_0{side}/ (no data subfolder)
        path1 = os.path.join(self.root_data_path, 'raw', folder, side_str, f_str)
        if os.path.isfile(path1):
            return path1

        # 2. Try raw/{folder}/image_0{side}/data/
        path2 = os.path.join(self.root_data_path, 'raw', folder, f"{side_str}/data", f_str)
        if os.path.isfile(path2):
            return path2

        # 3. Try raw/*/{folder}/image_0{side}/  (for date-subfolder structure)
        raw_dir = os.path.join(self.root_data_path, 'raw')
        if os.path.isdir(raw_dir):
            for sub in os.listdir(raw_dir):
                candidate = os.path.join(raw_dir, sub, folder, side_str, f_str)
                if os.path.isfile(candidate):
                    return candidate

        # Fallback: return path1 (will cause error if not found)
        return path1

    def get_depth(self, folder, frame_index, side, do_flip):
        """Load depth from pre-computed depth maps."""
        depth_filename = os.path.join(
            self.depths_path,
            folder,
            "{:010d}.png".format(int(frame_index)))

        # Skip if depth file missing (defensive)
        if not os.path.exists(depth_filename):
            raise FileNotFoundError(f"Depth file not found: {depth_filename}")

        depth_gt = pil.open(depth_filename)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32)

        if depth_gt.max() > 100:
            depth_gt = depth_gt / 256.0

        # Ensure positive values for numerical stability
        depth_gt = np.maximum(depth_gt, 1e-6)

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

    def get_intrinsics(self):
        """Return camera intrinsics matrix"""
        return self.K.copy()