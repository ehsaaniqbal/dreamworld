"""Video writing utilities."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def write_video(frames: np.ndarray, path: str | Path, fps: int = 30) -> None:
    """Write RGB uint8 frames with shape [T, H, W, C] to an mp4 file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    array = np.asarray(frames)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"Expected frames with shape [T, H, W, 3], got {array.shape}")
    if array.shape[0] == 0:
        raise ValueError("Cannot write a video with zero frames")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    height, width = array.shape[1:3]
    if height % 2 or width % 2:
        pad_height = height % 2
        pad_width = width % 2
        array = np.pad(
            array,
            ((0, 0), (0, pad_height), (0, pad_width), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    imageio.mimsave(output_path, array, fps=fps, macro_block_size=1)
