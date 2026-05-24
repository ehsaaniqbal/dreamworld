"""Train Phase 4 action-conditioned dynamics over VQ-VAE tokens."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from dreamtrack.config import load_config
from dreamtrack.data.dataset import RolloutDataset, load_rollout_dataset
from dreamtrack.models.dynamics_transformer import (
    DynamicsTransformerConfig,
    TokenDynamicsTransformer,
)
from dreamtrack.models.vqvae import VQVAE, VQVAEConfig
from dreamtrack.train.train_autoencoder import _nested_get, choose_device
from dreamtrack.viz.plots import save_loss_curve

DynamicsBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class TokenSequenceDataset(Dataset[DynamicsBatch]):
    """Episode-respecting token/action windows for next-token dynamics."""

    def __init__(
        self,
        *,
        tokens: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        episode_start_indices: np.ndarray,
        episode_lengths: np.ndarray,
        context_length: int,
    ) -> None:
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens [T, H, W], got {tokens.shape}")
        self.tokens = tokens.astype(np.int64, copy=False)
        self.actions = actions.astype(np.float32, copy=False)
        self.rewards = rewards.astype(np.float32, copy=False)
        self.dones = dones.astype(np.float32, copy=False)
        self.context_length = context_length
        self.starts: list[int] = []

        episode_ranges = zip(episode_start_indices, episode_lengths, strict=True)
        for episode_start, episode_length in episode_ranges:
            max_offset = int(episode_length) - context_length - 1
            for offset in range(max(0, max_offset + 1)):
                self.starts.append(int(episode_start) + offset)
        if not self.starts:
            raise ValueError(
                "No dynamics windows available. Collect longer episodes or reduce context_length."
            )

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        start = self.starts[index]
        stop = start + self.context_length
        token_window = self.tokens[start : stop + 1]
        return (
            torch.from_numpy(token_window[:-1]),
            torch.from_numpy(self.actions[start:stop]),
            torch.from_numpy(token_window[1:]),
            torch.from_numpy(self.rewards[start:stop]),
            torch.from_numpy(self.dones[start:stop]),
        )


def load_vqvae_checkpoint(path: str | Path, device: torch.device) -> VQVAE:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    if checkpoint.get("model_type") != "vqvae":
        raise ValueError(f"Expected a VQ-VAE checkpoint, got {checkpoint.get('model_type')!r}")
    model_config = VQVAEConfig(**checkpoint["model_config"])
    model = VQVAE(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


@torch.no_grad()
def encode_rollout_tokens(
    tokenizer: VQVAE,
    dataset: RolloutDataset,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Encode every rollout frame once into VQ token grids."""
    encoded: list[np.ndarray] = []
    for start in tqdm(range(0, dataset.num_frames, batch_size), desc="Encoding frames"):
        frames = dataset.obs[start : start + batch_size].astype(np.float32) / 255.0
        batch = torch.from_numpy(frames).permute(0, 3, 1, 2).to(device)
        _z_q, _vq_loss, tokens = tokenizer.encode(batch)
        encoded.append(tokens.detach().cpu().numpy().astype(np.int64))
    return np.concatenate(encoded, axis=0)


def train_dynamics(
    *,
    config_path: str | Path,
    data_path: str | Path,
    tokenizer_path: str | Path,
    out: str | Path,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    context_length: int | None = None,
    d_model: int | None = None,
    n_heads: int | None = None,
    n_layers: int | None = None,
    dropout: float | None = None,
    reward_loss_weight: float | None = None,
    done_loss_weight: float | None = None,
    device_name: str | None = None,
    max_frames: int | None = None,
    num_workers: int = 0,
) -> dict[str, Any]:
    config = load_config(config_path)
    seed = int(_nested_get(config, ("train", "seed"), 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = choose_device(device_name)
    rollout_dataset = load_rollout_dataset(data_path)
    if max_frames is not None:
        rollout_dataset = _truncate_dataset(rollout_dataset, max_frames)

    tokenizer = load_vqvae_checkpoint(tokenizer_path, device)
    tokens = encode_rollout_tokens(
        tokenizer,
        rollout_dataset,
        device=device,
        batch_size=batch_size or int(_nested_get(config, ("train", "batch_size"), 64)),
    )

    train_context = context_length or int(_nested_get(config, ("model", "context_length"), 32))
    sequence_dataset = TokenSequenceDataset(
        tokens=tokens,
        actions=rollout_dataset.actions,
        rewards=rollout_dataset.rewards,
        dones=rollout_dataset.dones,
        episode_start_indices=rollout_dataset.episode_start_indices,
        episode_lengths=rollout_dataset.episode_lengths,
        context_length=train_context,
    )

    val_size = max(1, int(len(sequence_dataset) * 0.1)) if len(sequence_dataset) > 10 else 1
    train_size = len(sequence_dataset) - val_size
    if train_size <= 0:
        raise ValueError("Need at least two dynamics windows to train and validate")
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        sequence_dataset,
        [train_size, val_size],
        generator=generator,
    )

    train_batch_size = batch_size or int(_nested_get(config, ("train", "batch_size"), 64))
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(train_batch_size, train_size),
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(train_batch_size, val_size),
        shuffle=False,
        num_workers=num_workers,
    )

    token_grid_size = int(tokens.shape[1])
    model_config = DynamicsTransformerConfig(
        context_length=train_context,
        token_grid_size=token_grid_size,
        num_embeddings=tokenizer.config.num_embeddings,
        token_embedding_dim=tokenizer.config.embedding_dim,
        d_model=d_model or int(_nested_get(config, ("model", "d_model"), 256)),
        n_heads=n_heads or int(_nested_get(config, ("model", "n_heads"), 4)),
        n_layers=n_layers or int(_nested_get(config, ("model", "n_layers"), 4)),
        dropout=(
            dropout
            if dropout is not None
            else float(_nested_get(config, ("model", "dropout"), 0.1))
        ),
    )
    model = TokenDynamicsTransformer(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate or float(_nested_get(config, ("train", "learning_rate"), 3e-4)),
    )
    train_epochs = epochs or int(_nested_get(config, ("train", "epochs"), 20))
    reward_weight = (
        reward_loss_weight
        if reward_loss_weight is not None
        else float(_nested_get(config, ("loss", "reward_weight"), 1.0))
    )
    done_weight = (
        done_loss_weight
        if done_loss_weight is not None
        else float(_nested_get(config, ("loss", "done_weight"), 0.2))
    )

    output_dir = Path(out)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_token_accuracy: list[float] = []
    val_reward_mae: list[float] = []
    best_val_loss = float("inf")

    for epoch in range(1, train_epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        progress = tqdm(train_loader, desc=f"Dynamics epoch {epoch}/{train_epochs}", leave=False)
        for batch in progress:
            inputs, actions, targets, rewards, dones = _batch_to_device(batch, device)
            token_logits, reward_pred, done_logits = model(inputs, actions)
            loss, parts = dynamics_loss(
                token_logits,
                reward_pred,
                done_logits,
                targets,
                rewards,
                dones,
                reward_weight=reward_weight,
                done_weight=done_weight,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(float(loss.detach().cpu()))
            progress.set_postfix(
                loss=f"{epoch_losses[-1]:.4f}",
                acc=f"{parts['token_accuracy']:.3f}",
            )

        train_loss = float(np.mean(epoch_losses))
        val_metrics = evaluate_dynamics(
            model,
            val_loader,
            device,
            reward_weight=reward_weight,
            done_weight=done_weight,
        )
        train_losses.append(train_loss)
        val_losses.append(val_metrics["loss"])
        val_token_accuracy.append(val_metrics["token_accuracy"])
        val_reward_mae.append(val_metrics["reward_mae"])

        checkpoint = {
            "model_type": "token_dynamics_transformer",
            "model_state": model.state_dict(),
            "model_config": asdict(model_config),
            "tokenizer_checkpoint": str(tokenizer_path),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, checkpoint_dir / f"epoch_{epoch:03d}.pt")
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(checkpoint, checkpoint_dir / "best.pt")

    save_loss_curve(train_losses, output_dir / "loss_curve.png", title="Dynamics Train Loss")
    save_loss_curve(val_losses, output_dir / "val_loss_curve.png", title="Dynamics Val Loss")
    save_loss_curve(
        val_token_accuracy,
        output_dir / "token_accuracy.png",
        title="Validation Token Accuracy",
    )
    save_loss_curve(val_reward_mae, output_dir / "reward_mae.png", title="Validation Reward MAE")

    metrics = {
        "data_path": str(data_path),
        "tokenizer_checkpoint": str(tokenizer_path),
        "frames": int(rollout_dataset.num_frames),
        "windows": int(len(sequence_dataset)),
        "train_windows": int(train_size),
        "val_windows": int(val_size),
        "device": str(device),
        "epochs": train_epochs,
        "batch_size": train_batch_size,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "reward_loss_weight": reward_weight,
        "done_loss_weight": done_weight,
        "model_config": asdict(model_config),
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_token_accuracy": val_token_accuracy,
        "val_reward_mae": val_reward_mae,
        "best_val_loss": best_val_loss,
        "best_checkpoint": str(checkpoint_dir / "best.pt"),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    return metrics


def dynamics_loss(
    token_logits: torch.Tensor,
    reward_pred: torch.Tensor,
    done_logits: torch.Tensor,
    targets: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    *,
    reward_weight: float,
    done_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    token_loss = F.cross_entropy(
        token_logits.reshape(-1, token_logits.shape[-1]),
        targets.reshape(-1),
    )
    reward_loss = F.mse_loss(reward_pred, rewards)
    done_loss = F.binary_cross_entropy_with_logits(done_logits, dones)
    loss = token_loss + reward_weight * reward_loss + done_weight * done_loss

    with torch.no_grad():
        predicted_tokens = token_logits.argmax(dim=-1)
        token_accuracy = (predicted_tokens == targets).float().mean()
        reward_mae = (reward_pred - rewards).abs().mean()
        done_accuracy = ((done_logits.sigmoid() >= 0.5) == (dones >= 0.5)).float().mean()
    return loss, {
        "token_loss": float(token_loss.detach().cpu()),
        "reward_loss": float(reward_loss.detach().cpu()),
        "done_loss": float(done_loss.detach().cpu()),
        "token_accuracy": float(token_accuracy.detach().cpu()),
        "reward_mae": float(reward_mae.detach().cpu()),
        "done_accuracy": float(done_accuracy.detach().cpu()),
    }


@torch.no_grad()
def evaluate_dynamics(
    model: TokenDynamicsTransformer,
    loader: DataLoader,
    device: torch.device,
    *,
    reward_weight: float,
    done_weight: float,
) -> dict[str, float]:
    model.eval()
    metrics: dict[str, list[float]] = {
        "loss": [],
        "token_loss": [],
        "reward_loss": [],
        "done_loss": [],
        "token_accuracy": [],
        "reward_mae": [],
        "done_accuracy": [],
    }
    for batch in loader:
        inputs, actions, targets, rewards, dones = _batch_to_device(batch, device)
        token_logits, reward_pred, done_logits = model(inputs, actions)
        loss, parts = dynamics_loss(
            token_logits,
            reward_pred,
            done_logits,
            targets,
            rewards,
            dones,
            reward_weight=reward_weight,
            done_weight=done_weight,
        )
        metrics["loss"].append(float(loss.detach().cpu()))
        for key, value in parts.items():
            metrics[key].append(value)
    return {key: float(np.mean(values)) for key, values in metrics.items()}


def _batch_to_device(
    batch: DynamicsBatch,
    device: torch.device,
) -> DynamicsBatch:
    inputs, actions, targets, rewards, dones = batch
    return (
        inputs.to(device),
        actions.to(device),
        targets.to(device),
        rewards.to(device),
        dones.to(device),
    )


def _truncate_dataset(dataset: RolloutDataset, max_frames: int) -> RolloutDataset:
    keep_frames = min(max_frames, dataset.num_frames)
    keep_episode_count = 0
    kept_lengths: list[int] = []
    for length in dataset.episode_lengths:
        remaining = keep_frames - sum(kept_lengths)
        if remaining <= 0:
            break
        kept_lengths.append(min(int(length), remaining))
        keep_episode_count += 1
    episode_lengths = np.asarray(kept_lengths, dtype=np.int32)
    episode_start_indices = np.concatenate(
        [np.asarray([0], dtype=np.int32), np.cumsum(episode_lengths[:-1], dtype=np.int32)]
    )
    episode_ids = np.concatenate(
        [
            np.full(length, episode_id, dtype=np.int32)
            for episode_id, length in enumerate(episode_lengths)
        ]
    )
    return RolloutDataset(
        obs=dataset.obs[:keep_frames],
        actions=dataset.actions[:keep_frames],
        rewards=dataset.rewards[:keep_frames],
        dones=dataset.dones[:keep_frames],
        episode_ids=episode_ids,
        episode_start_indices=episode_start_indices,
        episode_lengths=episode_lengths,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("dreamtrack/configs/dynamics_transformer.yaml"),
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--reward-loss-weight", type=float, default=None)
    parser.add_argument("--done-loss-weight", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_dynamics(
        config_path=args.config,
        data_path=args.data,
        tokenizer_path=args.tokenizer,
        out=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        context_length=args.context_length,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        reward_loss_weight=args.reward_loss_weight,
        done_loss_weight=args.done_loss_weight,
        device_name=args.device,
        max_frames=args.max_frames,
        num_workers=args.num_workers,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
