"""Frame preprocessing utilities for CarRacing observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class FramePreprocessor:
    """Resize RGB frames and keep disk storage as uint8."""

    image_size: int = 64

    def __post_init__(self) -> None:
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")

    def to_uint8(self, frame: np.ndarray) -> np.ndarray:
        """Convert a raw RGB observation to a resized uint8 RGB frame."""
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(f"Expected RGB frame with shape [H, W, 3], got {array.shape}")

        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)

        image = Image.fromarray(array, mode="RGB")
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8)

    @staticmethod
    def to_float(frame: np.ndarray) -> np.ndarray:
        """Convert uint8 image data to float32 in [0, 1] for model inputs."""
        return np.asarray(frame, dtype=np.float32) / 255.0
