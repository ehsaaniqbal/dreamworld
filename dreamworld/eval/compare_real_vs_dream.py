"""Compare real rollout frames with decoded world-model dreams."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dreamworld.data.dataset import load_rollout_dataset
from dreamworld.eval.eval_dynamics import load_dynamics_checkpoint
from dreamworld.train.train_autoencoder import choose_device
from dreamworld.train.train_dynamics import encode_rollout_tokens, load_vqvae_checkpoint
from dreamworld.viz.video import write_video


@torch.no_grad()
def compare_real_vs_dream(
    *,
    data_path: str | Path,
    tokenizer_path: str | Path | None,
    dynamics_path: str | Path,
    out: str | Path,
    horizon: int = 30,
    start_index: int = 0,
    batch_size: int = 64,
    fps: int = 12,
    device_name: str | None = None,
) -> dict[str, object]:
    """Write videos and metrics for teacher-forced and open-loop dream rollouts."""
    device = choose_device(device_name)
    dynamics, dynamics_checkpoint = load_dynamics_checkpoint(dynamics_path, device)
    resolved_tokenizer = tokenizer_path or dynamics_checkpoint["tokenizer_checkpoint"]
    tokenizer = load_vqvae_checkpoint(resolved_tokenizer, device)
    dataset = load_rollout_dataset(data_path)

    context = dynamics.config.context_length
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    max_horizon = dataset.num_frames - start_index - context
    rollout_horizon = min(horizon, max_horizon)
    if rollout_horizon <= 0:
        raise ValueError(
            f"Need at least context+horizon frames, got {dataset.num_frames} frames, "
            f"context={context}, start_index={start_index}"
        )

    tokens_np = encode_rollout_tokens(
        tokenizer,
        dataset,
        device=device,
        batch_size=batch_size,
    )
    tokens = torch.from_numpy(tokens_np).long().to(device)
    actions = torch.from_numpy(dataset.actions).float().to(device)

    teacher_tokens, teacher_rewards, teacher_done_probs = _teacher_forced_tokens(
        dynamics=dynamics,
        tokens=tokens,
        actions=actions,
        start_index=start_index,
        context=context,
        horizon=rollout_horizon,
    )
    dream_tokens, dream_rewards, dream_done_probs = _open_loop_tokens(
        dynamics=dynamics,
        tokens=tokens,
        actions=actions,
        start_index=start_index,
        context=context,
        horizon=rollout_horizon,
    )

    real_future_tokens = tokens[start_index + context : start_index + context + rollout_horizon]
    real_frames = dataset.obs[start_index + context : start_index + context + rollout_horizon]
    teacher_frames = _decode_tokens(tokenizer, teacher_tokens, device)
    dream_frames = _decode_tokens(tokenizer, dream_tokens, device)

    teacher_video = _comparison_frames(real_frames, teacher_frames)
    dream_video = _comparison_frames(real_frames, dream_frames)

    output_dir = Path(out)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_video(teacher_video, output_dir / "teacher_forced_vs_real.mp4", fps=fps)
    write_video(dream_video, output_dir / "open_loop_dream_vs_real.mp4", fps=fps)

    pixel_mse = ((dream_frames.astype(np.float32) - real_frames.astype(np.float32)) / 255.0) ** 2
    teacher_pixel_mse = (
        (teacher_frames.astype(np.float32) - real_frames.astype(np.float32)) / 255.0
    ) ** 2
    token_accuracy = (
        dream_tokens.detach().cpu() == real_future_tokens.detach().cpu()
    ).float().mean(dim=(1, 2))
    teacher_token_accuracy = (
        teacher_tokens.detach().cpu() == real_future_tokens.detach().cpu()
    ).float().mean(dim=(1, 2))

    real_rewards = dataset.rewards[start_index + context : start_index + context + rollout_horizon]
    metrics = {
        "data_path": str(data_path),
        "tokenizer_checkpoint": str(resolved_tokenizer),
        "dynamics_checkpoint": str(dynamics_path),
        "start_index": start_index,
        "context_length": context,
        "horizon": rollout_horizon,
        "teacher_forced_video": str(output_dir / "teacher_forced_vs_real.mp4"),
        "open_loop_video": str(output_dir / "open_loop_dream_vs_real.mp4"),
        "dream_pixel_mse_by_step": pixel_mse.mean(axis=(1, 2, 3)).astype(float).tolist(),
        "teacher_pixel_mse_by_step": teacher_pixel_mse.mean(axis=(1, 2, 3)).astype(float).tolist(),
        "dream_token_accuracy_by_step": token_accuracy.numpy().astype(float).tolist(),
        "teacher_token_accuracy_by_step": teacher_token_accuracy.numpy().astype(float).tolist(),
        "dream_reward_prediction_by_step": (
            dream_rewards.detach().cpu().numpy().astype(float).tolist()
        ),
        "teacher_reward_prediction_by_step": (
            teacher_rewards.detach().cpu().numpy().astype(float).tolist()
        ),
        "real_reward_by_step": real_rewards.astype(float).tolist(),
        "dream_done_probability_by_step": (
            dream_done_probs.detach().cpu().numpy().astype(float).tolist()
        ),
        "teacher_done_probability_by_step": (
            teacher_done_probs.detach().cpu().numpy().astype(float).tolist()
        ),
        "mean_dream_pixel_mse": float(pixel_mse.mean()),
        "mean_teacher_pixel_mse": float(teacher_pixel_mse.mean()),
        "mean_dream_token_accuracy": float(token_accuracy.mean()),
        "mean_teacher_token_accuracy": float(teacher_token_accuracy.mean()),
        "mean_dream_reward_mae": float(
            np.mean(np.abs(dream_rewards.detach().cpu().numpy() - real_rewards))
        ),
        "mean_teacher_reward_mae": float(
            np.mean(np.abs(teacher_rewards.detach().cpu().numpy() - real_rewards))
        ),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    return metrics


@torch.no_grad()
def _teacher_forced_tokens(
    *,
    dynamics,
    tokens: torch.Tensor,
    actions: torch.Tensor,
    start_index: int,
    context: int,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    predictions: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    done_probs: list[torch.Tensor] = []
    for offset in range(horizon):
        start = start_index + offset
        input_tokens = tokens[start : start + context].unsqueeze(0)
        input_actions = actions[start : start + context].unsqueeze(0)
        next_tokens, reward, done_prob = dynamics.imagine_step(input_tokens, input_actions)
        predictions.append(next_tokens.squeeze(0))
        rewards.append(reward.squeeze(0))
        done_probs.append(done_prob.squeeze(0))
    return torch.stack(predictions), torch.stack(rewards), torch.stack(done_probs)


@torch.no_grad()
def _open_loop_tokens(
    *,
    dynamics,
    tokens: torch.Tensor,
    actions: torch.Tensor,
    start_index: int,
    context: int,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    context_tokens = tokens[start_index : start_index + context].clone()
    context_actions = actions[start_index : start_index + context].clone()
    predictions: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    done_probs: list[torch.Tensor] = []

    for offset in range(horizon):
        next_tokens, reward, done_prob = dynamics.imagine_step(
            context_tokens.unsqueeze(0),
            context_actions.unsqueeze(0),
        )
        next_grid = next_tokens.squeeze(0)
        predictions.append(next_grid)
        rewards.append(reward.squeeze(0))
        done_probs.append(done_prob.squeeze(0))

        action_index = start_index + context + offset
        next_action = actions[action_index].unsqueeze(0)
        context_tokens = torch.cat([context_tokens[1:], next_grid.unsqueeze(0)], dim=0)
        context_actions = torch.cat([context_actions[1:], next_action], dim=0)

    return torch.stack(predictions), torch.stack(rewards), torch.stack(done_probs)


@torch.no_grad()
def _decode_tokens(tokenizer, tokens: torch.Tensor, device: torch.device) -> np.ndarray:
    decoded = tokenizer.decode_tokens(tokens.to(device)).detach().cpu()
    decoded = decoded.clamp(0.0, 1.0).permute(0, 2, 3, 1).numpy()
    return np.clip(decoded * 255.0, 0, 255).astype(np.uint8)


def _comparison_frames(real_frames: np.ndarray, imagined_frames: np.ndarray) -> np.ndarray:
    if real_frames.shape != imagined_frames.shape:
        raise ValueError(f"Frame shape mismatch: {real_frames.shape} vs {imagined_frames.shape}")
    error = np.abs(real_frames.astype(np.int16) - imagined_frames.astype(np.int16)).astype(np.uint8)
    heat = _error_heatmap(error)
    pad = np.full((real_frames.shape[0], real_frames.shape[1], 3, 3), 255, dtype=np.uint8)
    return np.concatenate([real_frames, pad, imagined_frames, pad, heat], axis=2)


def _error_heatmap(error: np.ndarray) -> np.ndarray:
    gray = error.mean(axis=-1).astype(np.uint8)
    heat = np.zeros((*gray.shape, 3), dtype=np.uint8)
    heat[..., 0] = gray
    heat[..., 1] = np.clip(gray.astype(np.int16) * 2, 0, 255).astype(np.uint8)
    heat[..., 2] = 255 - gray
    return heat


def _write_contact_sheet(frames: np.ndarray, path: Path, max_items: int = 8) -> None:
    count = min(max_items, frames.shape[0])
    if count == 0:
        return
    pad = 2
    height, width, channels = frames.shape[1:]
    sheet = np.full((height, count * width + (count - 1) * pad, channels), 255, np.uint8)
    for index in range(count):
        x0 = index * (width + pad)
        sheet[:, x0 : x0 + width] = frames[index]
    Image.fromarray(sheet).save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--dynamics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = compare_real_vs_dream(
        data_path=args.data,
        tokenizer_path=args.tokenizer,
        dynamics_path=args.dynamics,
        out=args.out,
        horizon=args.horizon,
        start_index=args.start_index,
        batch_size=args.batch_size,
        fps=args.fps,
        device_name=args.device,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
