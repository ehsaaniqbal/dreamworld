"""Dataset utilities for saved CarRacing rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RolloutDataset:
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    episode_ids: np.ndarray
    episode_start_indices: np.ndarray
    episode_lengths: np.ndarray

    @property
    def num_frames(self) -> int:
        return int(self.obs.shape[0])

    @property
    def image_shape(self) -> tuple[int, int, int]:
        return tuple(int(dim) for dim in self.obs.shape[1:])


def load_rollout_dataset(path: str | Path) -> RolloutDataset:
    """Load a rollout dataset saved by dreamworld.data.collect_rollouts."""
    with np.load(Path(path)) as data:
        dataset = RolloutDataset(
            obs=data["obs"],
            actions=data["actions"],
            rewards=data["rewards"],
            dones=data["dones"],
            episode_ids=data["episode_ids"],
            episode_start_indices=data["episode_start_indices"],
            episode_lengths=data["episode_lengths"],
        )
    validate_rollout_dataset(dataset)
    return dataset


def validate_rollout_dataset(dataset: RolloutDataset, image_size: int | None = None) -> None:
    """Validate shapes, dtypes, finite values, and episode boundaries."""
    num_frames = dataset.obs.shape[0]
    if num_frames == 0:
        raise ValueError("Dataset contains no frames")

    if dataset.obs.ndim != 4 or dataset.obs.shape[-1] != 3:
        raise ValueError(f"obs must have shape [T, H, W, 3], got {dataset.obs.shape}")
    if dataset.obs.dtype != np.uint8:
        raise ValueError(f"obs must be uint8, got {dataset.obs.dtype}")
    if image_size is not None and tuple(dataset.obs.shape[1:3]) != (image_size, image_size):
        raise ValueError(f"Expected image size {image_size}, got {dataset.obs.shape[1:3]}")

    expected_1d = {
        "rewards": dataset.rewards,
        "dones": dataset.dones,
        "episode_ids": dataset.episode_ids,
    }
    for name, value in expected_1d.items():
        if value.shape != (num_frames,):
            raise ValueError(f"{name} must have shape [{num_frames}], got {value.shape}")

    if dataset.actions.shape != (num_frames, 3):
        raise ValueError(f"actions must have shape [{num_frames}, 3], got {dataset.actions.shape}")
    if dataset.actions.dtype != np.float32:
        raise ValueError(f"actions must be float32, got {dataset.actions.dtype}")
    if dataset.rewards.dtype != np.float32:
        raise ValueError(f"rewards must be float32, got {dataset.rewards.dtype}")
    if dataset.dones.dtype != np.bool_:
        raise ValueError(f"dones must be bool, got {dataset.dones.dtype}")

    if not np.isfinite(dataset.actions).all():
        raise ValueError("actions contain NaNs or infinities")
    if not np.isfinite(dataset.rewards).all():
        raise ValueError("rewards contain NaNs or infinities")

    if int(dataset.episode_lengths.sum()) != num_frames:
        raise ValueError("episode_lengths do not sum to the number of frames")
    if dataset.episode_start_indices.shape != dataset.episode_lengths.shape:
        raise ValueError("episode_start_indices and episode_lengths must have matching shapes")
