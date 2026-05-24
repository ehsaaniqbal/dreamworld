"""Train the Phase 2 convolutional autoencoder."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from dreamtrack.config import load_config
from dreamtrack.data.dataset import load_rollout_dataset
from dreamtrack.models.autoencoder import AutoencoderConfig, ConvAutoencoder
from dreamtrack.viz.grids import make_reconstruction_grid, save_image
from dreamtrack.viz.plots import save_loss_curve


class FrameDataset(Dataset[torch.Tensor]):
    """Read rollout frames as float tensors in CHW format."""

    def __init__(self, frames: np.ndarray) -> None:
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"Expected frames [T, H, W, 3], got {frames.shape}")
        self.frames = frames

    def __len__(self) -> int:
        return int(self.frames.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        frame = self.frames[index].astype(np.float32) / 255.0
        return torch.from_numpy(frame).permute(2, 0, 1)


def choose_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _nested_get(config: dict[str, Any], keys: tuple[str, ...], default: Any) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_nhwc_uint8(batch: torch.Tensor) -> np.ndarray:
    return (
        batch.detach()
        .cpu()
        .clamp(0.0, 1.0)
        .permute(0, 2, 3, 1)
        .numpy()
    )


def train_autoencoder(
    *,
    config_path: str | Path,
    data_path: str | Path,
    out: str | Path,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    latent_dim: int | None = None,
    device_name: str | None = None,
    max_frames: int | None = None,
    num_workers: int = 0,
) -> dict[str, Any]:
    config = load_config(config_path)
    seed = int(_nested_get(config, ("train", "seed"), 0))
    _seed_everything(seed)

    image_size = int(_nested_get(config, ("data", "image_size"), 64))
    model_config = AutoencoderConfig(
        image_size=image_size,
        latent_dim=latent_dim or int(_nested_get(config, ("model", "latent_dim"), 256)),
    )
    train_epochs = epochs or int(_nested_get(config, ("train", "epochs"), 10))
    train_batch_size = batch_size or int(_nested_get(config, ("train", "batch_size"), 128))
    lr = learning_rate or float(_nested_get(config, ("train", "learning_rate"), 3e-4))

    output_dir = Path(out)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    rollout_dataset = load_rollout_dataset(data_path)
    frames = rollout_dataset.obs
    if max_frames is not None:
        frames = frames[:max_frames]
    if tuple(frames.shape[1:3]) != (image_size, image_size):
        raise ValueError(f"Config image_size={image_size}, dataset image shape={frames.shape[1:3]}")

    frame_dataset = FrameDataset(frames)
    val_size = max(1, int(len(frame_dataset) * 0.1)) if len(frame_dataset) > 10 else 1
    train_size = len(frame_dataset) - val_size
    if train_size <= 0:
        raise ValueError("Need at least two frames to train and validate")

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        frame_dataset,
        [train_size, val_size],
        generator=generator,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(train_batch_size, train_size),
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(val_dataset, batch_size=min(train_batch_size, val_size), shuffle=False)

    device = choose_device(device_name)
    model = ConvAutoencoder(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")

    fixed_batch = next(iter(val_loader)).to(device)

    for epoch in range(1, train_epochs + 1):
        model.train()
        batch_losses: list[float] = []
        progress = tqdm(train_loader, desc=f"AE epoch {epoch}/{train_epochs}", leave=False)
        for batch in progress:
            batch = batch.to(device)
            reconstruction, _z = model(batch)
            loss = loss_fn(reconstruction, batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_losses.append(float(loss.detach().cpu()))
            progress.set_postfix(loss=f"{batch_losses[-1]:.5f}")

        train_loss = float(np.mean(batch_losses))
        val_loss = evaluate_loss(model, val_loader, loss_fn, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        with torch.no_grad():
            reconstructions, _z = model(fixed_batch)
        grid = make_reconstruction_grid(
            _to_nhwc_uint8(fixed_batch),
            _to_nhwc_uint8(reconstructions),
        )
        save_image(grid, output_dir / f"recon_epoch_{epoch:03d}.png")

        checkpoint = {
            "model_type": "autoencoder",
            "model_state": model.state_dict(),
            "model_config": model_config.__dict__,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        torch.save(checkpoint, checkpoint_dir / f"epoch_{epoch:03d}.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, checkpoint_dir / "best.pt")

    save_loss_curve(train_losses, output_dir / "loss_curve.png", title="Autoencoder Train MSE")
    save_loss_curve(val_losses, output_dir / "val_loss_curve.png", title="Autoencoder Val MSE")

    metrics = {
        "data_path": str(data_path),
        "frames": int(len(frame_dataset)),
        "train_frames": int(train_size),
        "val_frames": int(val_size),
        "device": str(device),
        "epochs": train_epochs,
        "batch_size": train_batch_size,
        "learning_rate": lr,
        "model_config": model_config.__dict__,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "best_val_loss": best_val_loss,
        "best_checkpoint": str(checkpoint_dir / "best.pt"),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    return metrics


def evaluate_loss(
    model: ConvAutoencoder,
    loader: DataLoader[torch.Tensor],
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            reconstruction, _z = model(batch)
            losses.append(float(loss_fn(reconstruction, batch).detach().cpu()))
    return float(np.mean(losses))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("dreamtrack/configs/tokenizer_ae.yaml"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_autoencoder(
        config_path=args.config,
        data_path=args.data,
        out=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        latent_dim=args.latent_dim,
        device_name=args.device,
        max_frames=args.max_frames,
        num_workers=args.num_workers,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
