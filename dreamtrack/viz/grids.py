"""Image grid utilities for reconstruction visualizations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def make_reconstruction_grid(
    originals: np.ndarray,
    reconstructions: np.ndarray,
    *,
    max_items: int = 8,
    pad: int = 2,
) -> np.ndarray:
    """Create a two-row grid: originals on top, reconstructions below."""
    orig = _to_uint8(originals[:max_items])
    recon = _to_uint8(reconstructions[:max_items])
    if orig.shape != recon.shape:
        raise ValueError(
            f"Original and reconstruction shapes differ: {orig.shape} vs {recon.shape}"
        )

    count, height, width, channels = orig.shape
    grid = np.full((2 * height + pad, count * width + (count - 1) * pad, channels), 255, np.uint8)
    for index in range(count):
        x0 = index * (width + pad)
        grid[:height, x0 : x0 + width] = orig[index]
        grid[height + pad :, x0 : x0 + width] = recon[index]
    return grid


def save_image(array: np.ndarray, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_to_uint8(array)).save(output_path)


def make_token_grid(tokens: np.ndarray, *, max_items: int = 8, scale: int = 8) -> np.ndarray:
    """Visualize discrete token maps as colorized grids."""
    maps = np.asarray(tokens[:max_items])
    if maps.ndim != 3:
        raise ValueError(f"Expected token maps with shape [N, H, W], got {maps.shape}")

    max_token = max(int(maps.max()), 1)
    normalized = maps.astype(np.float32) / max_token
    red = (normalized * 255).astype(np.uint8)
    green = ((1.0 - normalized) * 180).astype(np.uint8)
    blue = ((0.5 + 0.5 * np.sin(normalized * np.pi * 4.0)) * 255).astype(np.uint8)
    color = np.stack([red, green, blue], axis=-1)

    tiles = []
    for token_map in color:
        image = Image.fromarray(token_map, mode="RGB").resize(
            (token_map.shape[1] * scale, token_map.shape[0] * scale),
            Image.Resampling.NEAREST,
        )
        tiles.append(np.asarray(image, dtype=np.uint8))

    pad = 2
    height, width, channels = tiles[0].shape
    grid = np.full((height, len(tiles) * width + (len(tiles) - 1) * pad, channels), 255, np.uint8)
    for index, tile in enumerate(tiles):
        x0 = index * (width + pad)
        grid[:, x0 : x0 + width] = tile
    return grid


def _to_uint8(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array)
    if value.dtype == np.uint8:
        return value
    if np.issubdtype(value.dtype, np.floating):
        value = np.clip(value, 0.0, 1.0) * 255.0
    return np.clip(value, 0, 255).astype(np.uint8)
