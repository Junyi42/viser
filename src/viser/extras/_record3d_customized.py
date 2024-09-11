from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Tuple, cast

import imageio.v3 as iio
import liblzfse
import numpy as np
import numpy as onp
import numpy.typing as onpt
import skimage.transform
from scipy.spatial.transform import Rotation


class Record3dLoader_Customized:
    """Helper for loading frames for Record3D captures."""

    # NOTE(hangg): Consider moving this module into
    # `examples/7_record3d_visualizer.py` since it is usecase-specific.

    def __init__(self, data_dir: Path, conf_threshold: float = 1.0, foreground_conf_threshold: float = 0.1):

        # Read metadata.
        intrinsics_path = data_dir / "pred_intrinsics.txt"
        intrinsics = np.loadtxt(intrinsics_path)

        K: onp.ndarray = np.array(intrinsics, np.float32).reshape(-1, 3, 3)
        fps = 30

        poses_path = data_dir / "pred_traj.txt"
        poses = np.loadtxt(poses_path)
        T_world_cameras: onp.ndarray = np.array(poses, np.float32)
        T_world_cameras = np.concatenate(
            [   # convert tum pose to se3 pose
                # Rotation.from_quat(np.concatenate([T_world_cameras[:, 5:], T_world_cameras[:,4:5]], -1)).as_matrix(),
                Rotation.from_quat(T_world_cameras[:, 4:]).as_matrix(),
                T_world_cameras[:, 1:4, None],
            ],
            -1,
        )
        T_world_cameras = (T_world_cameras @ np.diag([1, 1, 1, 1])).astype(np.float32)

        self.fps = fps
        self.conf_threshold = conf_threshold
        self.foreground_conf_threshold = foreground_conf_threshold
        self.K = K
        self.T_world_cameras = T_world_cameras

        # Read frames.
        self.rgb_paths = sorted(data_dir.glob("frame_*.png"), key=lambda p: int(p.stem.split("_")[-1]))
        self.depth_paths = sorted(data_dir.glob("frame_*.npy"), key=lambda p: int(p.stem.split("_")[-1]))
        self.conf_paths = sorted(data_dir.glob("conf_*.npy"), key=lambda p: int(p.stem.split("_")[-1]))
        self.mask_paths = sorted(data_dir.glob("enlarged_dynamic_mask_*.png"), key=lambda p: int(p.stem.split("_")[-1]))

        # remove the last frame since it does not have gt dynamic mask
        self.rgb_paths = self.rgb_paths[:-1]

    def num_frames(self) -> int:
        return len(self.rgb_paths)

    def get_frame(self, index: int) -> Record3dFrame:
        # Read conf.
        conf = np.load(self.conf_paths[index])
        conf: onpt.NDArray[onp.float32] = conf
        # clip conf to 0
        conf = np.clip(conf, 0.0001, 99999)

        # Read depth.
        depth = np.load(self.depth_paths[index])
        depth: onpt.NDArray[onp.float32] = depth

        # Read mask.
        mask = iio.imread(self.mask_paths[index]) > 0
        mask: onpt.NDArray[onp.bool_] = mask

        # Read RGB.
        rgb = iio.imread(self.rgb_paths[index])
        return Record3dFrame(
            K=self.K[index],
            rgb=rgb,
            depth=depth,
            mask=mask,
            conf=conf,
            T_world_camera=self.T_world_cameras[index],
            conf_threshold=self.conf_threshold,
            foreground_conf_threshold=self.foreground_conf_threshold,
        )


@dataclasses.dataclass
class Record3dFrame:
    """A single frame from a Record3D capture."""

    K: onpt.NDArray[onp.float32]
    rgb: onpt.NDArray[onp.uint8]
    depth: onpt.NDArray[onp.float32]
    mask: onpt.NDArray[onp.bool_]
    conf: onpt.NDArray[onp.float32]
    T_world_camera: onpt.NDArray[onp.float32]
    conf_threshold: float = 1.0
    foreground_conf_threshold: float = 0.1

    def get_point_cloud(
        self, downsample_factor: int = 1
    ) -> Tuple[onpt.NDArray[onp.float32], onpt.NDArray[onp.uint8]]:
        rgb = self.rgb[::downsample_factor, ::downsample_factor]
        depth = skimage.transform.resize(self.depth, rgb.shape[:2], order=0)
        mask = cast(
            onpt.NDArray[onp.bool_],
            skimage.transform.resize(self.mask, rgb.shape[:2], order=0),
        )
        assert depth.shape == rgb.shape[:2]

        K = self.K
        T_world_camera = self.T_world_camera

        img_wh = rgb.shape[:2][::-1]

        grid = (
            np.stack(np.meshgrid(np.arange(img_wh[0]), np.arange(img_wh[1])), 2) + 0.5
        )
        grid = grid * downsample_factor
        conf_mask = self.conf > self.conf_threshold
        fg_conf_mask = self.conf > self.foreground_conf_threshold

        homo_grid = np.pad(grid[fg_conf_mask * mask], np.array([[0, 0], [0, 1]]), constant_values=1)
        local_dirs = np.einsum("ij,bj->bi", np.linalg.inv(K), homo_grid)
        dirs = np.einsum("ij,bj->bi", T_world_camera[:3, :3], local_dirs)
        points = (T_world_camera[:, -1] + dirs * depth[fg_conf_mask * mask, None]).astype(np.float32)
        point_colors = rgb[fg_conf_mask * mask]
        
        bg_homo_grid = np.pad(grid[conf_mask * ~mask], np.array([[0, 0], [0, 1]]), constant_values=1)
        bg_local_dirs = np.einsum("ij,bj->bi", np.linalg.inv(K), bg_homo_grid)
        bg_dirs = np.einsum("ij,bj->bi", T_world_camera[:3, :3], bg_local_dirs)
        bg_points = (T_world_camera[:, -1] + bg_dirs * depth[conf_mask * ~mask, None]).astype(np.float32)
        bg_point_colors = rgb[conf_mask * ~mask]

        return points, point_colors, bg_points, bg_point_colors
