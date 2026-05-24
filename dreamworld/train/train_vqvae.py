"""Train the Phase 3 VQ-VAE visual tokenizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dreamworld.config import load_config
from dreamworld.data.dataset import load_rollout_dataset
from dreamworld.models.vqvae import VQVAE, VQVAEConfig, codebook_metrics
from dreamworld.train.train_autoencoder import (
    FrameDataset,
    _nested_get,
    _seed_everything,
    _to_nhwc_uint8,
    choose_device,
)
from dreamworld.viz.grids import make_reconstruction_grid, make_token_grid, save_image
from dreamworld.viz.plots import save_loss_curve


def train_vqvae(
    *,
    config_path: str | Path,
    data_path: str | Path,
    out: str | Path,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    embedding_dim: int | None = None,
    num_embeddings: int | None = None,
    commitment_cost: float | None = None,
    device_name: str | None = None,
    max_frames: int | None = None,
    num_workers: int = 0,
) -> dict[str, Any]:
    config = load_config(config_path)
    seed = int(_nested_get(config, ("train", "seed"), 0))
    _seed_everything(seed)

    image_size = int(_nested_get(config, ("data", "image_size"), 64))
    model_config = VQVAEConfig(
        image_size=image_size,
        embedding_dim=embedding_dim or int(_nested_get(config, ("model", "embedding_dim"), 64)),
        num_embeddings=num_embeddings or int(_nested_get(config, ("model", "num_embeddings"), 512)),
        commitment_cost=(
            commitment_cost
            if commitment_cost is not None
            else float(_nested_get(config, ("model", "commitment_cost"), 0.25))
        ),
    )
    train_epochs = epochs or int(_nested_get(config, ("train", "epochs"), 20))
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
    model = VQVAE(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_losses: list[float] = []
    val_losses: list[float] = []
    train_recon_losses: list[float] = []
    train_vq_losses: list[float] = []
    perplexities: list[float] = []
    dead_code_fractions: list[float] = []
    best_val_loss = float("inf")
    best_codebook_score = float("-inf")

    fixed_batch = next(iter(val_loader)).to(device)

    for epoch in range(1, train_epochs + 1):
        model.train()
        total_batch_losses: list[float] = []
        recon_batch_losses: list[float] = []
        vq_batch_losses: list[float] = []
        epoch_tokens: list[torch.Tensor] = []

        progress = tqdm(train_loader, desc=f"VQ-VAE epoch {epoch}/{train_epochs}", leave=False)
        for batch in progress:
            batch = batch.to(device)
            reconstruction, vq_loss, tokens = model(batch)
            recon_loss = F.mse_loss(reconstruction, batch)
            loss = recon_loss + vq_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_batch_losses.append(float(loss.detach().cpu()))
            recon_batch_losses.append(float(recon_loss.detach().cpu()))
            vq_batch_losses.append(float(vq_loss.detach().cpu()))
            epoch_tokens.append(tokens.detach().cpu())
            progress.set_postfix(loss=f"{total_batch_losses[-1]:.5f}")

        train_loss = float(np.mean(total_batch_losses))
        train_recon_loss = float(np.mean(recon_batch_losses))
        train_vq_loss = float(np.mean(vq_batch_losses))
        val_metrics = evaluate_vqvae(model, val_loader, device)
        token_metrics = codebook_metrics(
            torch.cat(epoch_tokens, dim=0),
            model_config.num_embeddings,
        )

        train_losses.append(train_loss)
        val_losses.append(val_metrics["loss"])
        train_recon_losses.append(train_recon_loss)
        train_vq_losses.append(train_vq_loss)
        perplexities.append(token_metrics["codebook_perplexity"])
        dead_code_fractions.append(token_metrics["dead_code_fraction"])

        with torch.no_grad():
            reconstructions, _vq_loss, tokens = model(fixed_batch)
        recon_grid = make_reconstruction_grid(
            _to_nhwc_uint8(fixed_batch),
            _to_nhwc_uint8(reconstructions),
        )
        token_grid = make_token_grid(tokens.detach().cpu().numpy())
        save_image(recon_grid, output_dir / f"recon_epoch_{epoch:03d}.png")
        save_image(token_grid, output_dir / f"tokens_epoch_{epoch:03d}.png")

        checkpoint = {
            "model_type": "vqvae",
            "model_state": model.state_dict(),
            "model_config": model_config.__dict__,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "codebook_metrics": token_metrics,
        }
        torch.save(checkpoint, checkpoint_dir / f"epoch_{epoch:03d}.pt")
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(checkpoint, checkpoint_dir / "best.pt")
        codebook_score = (
            token_metrics["codebook_perplexity"]
            * (1.0 - token_metrics["dead_code_fraction"])
        )
        if codebook_score > best_codebook_score:
            best_codebook_score = codebook_score
            torch.save(checkpoint, checkpoint_dir / "best_codebook.pt")

    save_loss_curve(train_losses, output_dir / "loss_curve.png", title="VQ-VAE Train Loss")
    save_loss_curve(val_losses, output_dir / "val_loss_curve.png", title="VQ-VAE Val Loss")
    save_loss_curve(
        perplexities,
        output_dir / "codebook_perplexity.png",
        title="Codebook Perplexity",
    )

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
        "train_recon_loss": train_recon_losses,
        "train_vq_loss": train_vq_losses,
        "val_loss": val_losses,
        "codebook_perplexity": perplexities,
        "dead_code_fraction": dead_code_fractions,
        "best_val_loss": best_val_loss,
        "best_checkpoint": str(checkpoint_dir / "best.pt"),
        "best_codebook_checkpoint": str(checkpoint_dir / "best_codebook.pt"),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    return metrics


def evaluate_vqvae(
    model: VQVAE,
    loader: DataLoader[torch.Tensor],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    recon_losses: list[float] = []
    vq_losses: list[float] = []
    tokens: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            reconstruction, vq_loss, token_batch = model(batch)
            recon_loss = F.mse_loss(reconstruction, batch)
            loss = recon_loss + vq_loss
            losses.append(float(loss.detach().cpu()))
            recon_losses.append(float(recon_loss.detach().cpu()))
            vq_losses.append(float(vq_loss.detach().cpu()))
            tokens.append(token_batch.detach().cpu())

    usage = codebook_metrics(torch.cat(tokens, dim=0), model.config.num_embeddings)
    return {
        "loss": float(np.mean(losses)),
        "recon_loss": float(np.mean(recon_losses)),
        "vq_loss": float(np.mean(vq_losses)),
        **usage,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("dreamworld/configs/tokenizer_vqvae.yaml"),
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--num-embeddings", type=int, default=None)
    parser.add_argument("--commitment-cost", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_vqvae(
        config_path=args.config,
        data_path=args.data,
        out=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        embedding_dim=args.embedding_dim,
        num_embeddings=args.num_embeddings,
        commitment_cost=args.commitment_cost,
        device_name=args.device,
        max_frames=args.max_frames,
        num_workers=args.num_workers,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
