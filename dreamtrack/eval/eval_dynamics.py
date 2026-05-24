"""Evaluate one-step token dynamics from a saved Phase 4 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dreamtrack.data.dataset import load_rollout_dataset
from dreamtrack.models.dynamics_transformer import (
    DynamicsTransformerConfig,
    TokenDynamicsTransformer,
)
from dreamtrack.train.train_autoencoder import choose_device
from dreamtrack.train.train_dynamics import (
    TokenSequenceDataset,
    encode_rollout_tokens,
    evaluate_dynamics,
    load_vqvae_checkpoint,
)


def load_dynamics_checkpoint(
    path: str | Path,
    device: torch.device,
) -> tuple[TokenDynamicsTransformer, dict]:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    if checkpoint.get("model_type") != "token_dynamics_transformer":
        raise ValueError(
            "Expected token_dynamics_transformer checkpoint, "
            f"got {checkpoint.get('model_type')!r}"
        )
    model = TokenDynamicsTransformer(DynamicsTransformerConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint


def eval_dynamics(
    *,
    checkpoint_path: str | Path,
    tokenizer_path: str | Path | None,
    data_path: str | Path,
    out: str | Path,
    batch_size: int = 64,
    device_name: str | None = None,
    num_workers: int = 0,
) -> dict[str, float | str]:
    device = choose_device(device_name)
    model, checkpoint = load_dynamics_checkpoint(checkpoint_path, device)
    tokenizer_checkpoint = tokenizer_path or checkpoint["tokenizer_checkpoint"]
    tokenizer = load_vqvae_checkpoint(tokenizer_checkpoint, device)
    dataset = load_rollout_dataset(data_path)
    tokens = encode_rollout_tokens(tokenizer, dataset, device=device, batch_size=batch_size)

    sequence_dataset = TokenSequenceDataset(
        tokens=tokens,
        actions=dataset.actions,
        rewards=dataset.rewards,
        dones=dataset.dones,
        episode_start_indices=dataset.episode_start_indices,
        episode_lengths=dataset.episode_lengths,
        context_length=model.config.context_length,
    )
    loader = DataLoader(
        sequence_dataset,
        batch_size=min(batch_size, len(sequence_dataset)),
        shuffle=False,
        num_workers=num_workers,
    )
    metrics = evaluate_dynamics(
        model,
        loader,
        device,
        reward_weight=1.0,
        done_weight=0.2,
    )
    output = {
        **metrics,
        "checkpoint": str(checkpoint_path),
        "tokenizer_checkpoint": str(tokenizer_checkpoint),
        "data_path": str(data_path),
        "windows": len(sequence_dataset),
    }
    output_dir = Path(out)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "one_step_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = eval_dynamics(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        data_path=args.data,
        out=args.out,
        batch_size=args.batch_size,
        device_name=args.device,
        num_workers=args.num_workers,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
