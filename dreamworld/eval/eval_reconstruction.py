"""Evaluate a trained autoencoder checkpoint on saved rollout frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dreamworld.data.dataset import load_rollout_dataset
from dreamworld.models.autoencoder import AutoencoderConfig, ConvAutoencoder
from dreamworld.models.vqvae import VQVAE, VQVAEConfig, codebook_metrics
from dreamworld.train.train_autoencoder import FrameDataset, choose_device
from dreamworld.viz.grids import make_reconstruction_grid, save_image


def _to_nhwc(batch: torch.Tensor) -> np.ndarray:
    return batch.detach().cpu().clamp(0.0, 1.0).permute(0, 2, 3, 1).numpy()


def eval_reconstruction(
    *,
    checkpoint_path: str | Path,
    data_path: str | Path,
    out: str | Path,
    batch_size: int = 64,
    max_batches: int | None = None,
    device_name: str | None = None,
) -> dict[str, float | str | int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    raw_config = checkpoint["model_config"]
    model_type = checkpoint.get("model_type")
    if model_type is None:
        model_type = "vqvae" if "embedding_dim" in raw_config else "autoencoder"

    if model_type == "vqvae":
        model = VQVAE(VQVAEConfig(**raw_config))
    elif model_type == "autoencoder":
        model = ConvAutoencoder(AutoencoderConfig(**raw_config))
    else:
        raise ValueError(f"Unsupported checkpoint model_type: {model_type}")
    model.load_state_dict(checkpoint["model_state"])

    device = choose_device(device_name)
    model.to(device)
    model.eval()

    dataset = load_rollout_dataset(data_path)
    frame_dataset = FrameDataset(dataset.obs)
    loader = DataLoader(frame_dataset, batch_size=batch_size, shuffle=False)

    mse_values: list[float] = []
    all_tokens: list[torch.Tensor] = []
    first_batch: torch.Tensor | None = None
    first_reconstruction: torch.Tensor | None = None

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = batch.to(device)
            output = model(batch)
            reconstruction = output[0]
            if model_type == "vqvae":
                all_tokens.append(output[2].detach().cpu())
            per_frame_mse = ((reconstruction - batch) ** 2).flatten(start_dim=1).mean(dim=1)
            mse_values.extend(float(value) for value in per_frame_mse.detach().cpu())
            if first_batch is None:
                first_batch = batch
                first_reconstruction = reconstruction

    if not mse_values:
        raise ValueError("No batches were evaluated")

    output_path = Path(out)
    output_path.mkdir(parents=True, exist_ok=True)
    if first_batch is not None and first_reconstruction is not None:
        grid = make_reconstruction_grid(_to_nhwc(first_batch), _to_nhwc(first_reconstruction))
        save_image(grid, output_path / "reconstruction_grid.png")

    metrics: dict[str, float | str | int] = {
        "checkpoint": str(checkpoint_path),
        "model_type": model_type,
        "data": str(data_path),
        "frames_evaluated": len(mse_values),
        "mse_mean": float(np.mean(mse_values)),
        "mse_std": float(np.std(mse_values)),
        "mse_min": float(np.min(mse_values)),
        "mse_max": float(np.max(mse_values)),
        "grid_path": str(output_path / "reconstruction_grid.png"),
    }
    if all_tokens and isinstance(model, VQVAE):
        metrics.update(codebook_metrics(torch.cat(all_tokens, dim=0), model.config.num_embeddings))
    with (output_path / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = eval_reconstruction(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        out=args.out,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        device_name=args.device,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
