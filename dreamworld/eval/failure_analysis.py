"""Mine and export failure cases where open-loop dreams diverge from reality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from dreamworld.data.dataset import load_rollout_dataset
from dreamworld.eval.compare_real_vs_dream import (
    _comparison_frames,
    _decode_tokens,
    _open_loop_tokens,
)
from dreamworld.eval.eval_dynamics import load_dynamics_checkpoint
from dreamworld.train.train_autoencoder import choose_device
from dreamworld.train.train_dynamics import encode_rollout_tokens, load_vqvae_checkpoint
from dreamworld.viz.video import write_video


@torch.no_grad()
def mine_failures(
    *,
    data_path: str | Path,
    dynamics_path: str | Path,
    tokenizer_path: str | Path | None,
    out: str | Path,
    horizon: int = 20,
    top_k: int = 5,
    stride: int = 1,
    batch_size: int = 64,
    planner_metrics_path: str | Path | None = None,
    device_name: str | None = None,
    fps: int = 12,
) -> dict[str, Any]:
    """Scan rollout windows and export the worst open-loop model failures."""
    device = choose_device(device_name)
    dynamics, dynamics_checkpoint = load_dynamics_checkpoint(dynamics_path, device)
    resolved_tokenizer = tokenizer_path or dynamics_checkpoint["tokenizer_checkpoint"]
    tokenizer = load_vqvae_checkpoint(resolved_tokenizer, device)
    dataset = load_rollout_dataset(data_path)
    tokens_np = encode_rollout_tokens(tokenizer, dataset, device=device, batch_size=batch_size)

    context = dynamics.config.context_length
    max_start = dataset.num_frames - context - horizon
    if max_start < 0:
        raise ValueError(
            f"Need at least context+horizon frames, got frames={dataset.num_frames}, "
            f"context={context}, horizon={horizon}"
        )

    tokens = torch.from_numpy(tokens_np).long().to(device)
    actions = torch.from_numpy(dataset.actions).float().to(device)
    candidates: list[dict[str, Any]] = []
    for start_index in range(0, max_start + 1, stride):
        failure = _score_failure(
            dynamics=dynamics,
            tokenizer=tokenizer,
            dataset=dataset,
            tokens=tokens,
            actions=actions,
            start_index=start_index,
            context=context,
            horizon=horizon,
            device=device,
        )
        candidates.append(failure)

    ranked = sorted(candidates, key=lambda item: float(item["failure_score"]), reverse=True)
    selected = ranked[:top_k]

    output_dir = Path(out)
    failures_dir = output_dir / "failures"
    failures_dir.mkdir(parents=True, exist_ok=True)
    for index, failure in enumerate(selected, start=1):
        video_path = failures_dir / f"failure_{index:03d}.mp4"
        json_path = failures_dir / f"failure_{index:03d}.json"
        write_video(failure.pop("comparison_frames"), video_path, fps=fps)
        failure["video_path"] = str(video_path)
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(failure, handle, indent=2)
            handle.write("\n")

    planner_gap = _planner_exploitation_score(planner_metrics_path)
    summary = {
        "data_path": str(data_path),
        "dynamics_checkpoint": str(dynamics_path),
        "tokenizer_checkpoint": str(resolved_tokenizer),
        "context_length": context,
        "horizon": horizon,
        "stride": stride,
        "scanned_windows": len(candidates),
        "top_k": len(selected),
        "planner_exploitation_score": planner_gap,
        "failures": selected,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_gap_plot(ranked, output_dir / "dream_real_gap.png")
    _write_summary(summary, output_dir / "summary.md")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return summary


@torch.no_grad()
def _score_failure(
    *,
    dynamics,
    tokenizer,
    dataset,
    tokens: torch.Tensor,
    actions: torch.Tensor,
    start_index: int,
    context: int,
    horizon: int,
    device: torch.device,
) -> dict[str, Any]:
    dream_tokens, dream_rewards, dream_done_probs = _open_loop_tokens(
        dynamics=dynamics,
        tokens=tokens,
        actions=actions,
        start_index=start_index,
        context=context,
        horizon=horizon,
    )
    real_future_tokens = tokens[start_index + context : start_index + context + horizon]
    real_frames = dataset.obs[start_index + context : start_index + context + horizon]
    dream_frames = _decode_tokens(tokenizer, dream_tokens, device)
    comparison_frames = _comparison_frames(real_frames, dream_frames)

    pixel_mse_by_step = (
        ((dream_frames.astype(np.float32) - real_frames.astype(np.float32)) / 255.0) ** 2
    ).mean(axis=(1, 2, 3))
    token_accuracy_by_step = (
        dream_tokens.detach().cpu() == real_future_tokens.detach().cpu()
    ).float().mean(dim=(1, 2))
    real_rewards = dataset.rewards[start_index + context : start_index + context + horizon]
    real_dones = dataset.dones[start_index + context : start_index + context + horizon]
    predicted_rewards = dream_rewards.detach().cpu().numpy()
    predicted_done_probs = dream_done_probs.detach().cpu().numpy()
    predicted_return = float(predicted_rewards.sum())
    real_return = float(real_rewards.sum())
    reward_prediction_error = float(abs(predicted_return - real_return))
    done_prediction_error = float(np.mean(np.abs(predicted_done_probs - real_dones.astype(float))))
    latent_drift_at_horizon = float(1.0 - token_accuracy_by_step[-1].item())
    mean_pixel_mse = float(pixel_mse_by_step.mean())
    failure_score = (
        mean_pixel_mse
        + 0.2 * reward_prediction_error
        + 0.5 * latent_drift_at_horizon
        + 0.1 * done_prediction_error
    )

    return {
        "start_index": start_index,
        "failure_score": float(failure_score),
        "predicted_return": predicted_return,
        "real_return": real_return,
        "predicted_return_minus_real_return": float(predicted_return - real_return),
        "reward_prediction_error": reward_prediction_error,
        "done_prediction_error": done_prediction_error,
        "latent_drift_at_horizon": latent_drift_at_horizon,
        "mean_pixel_mse": mean_pixel_mse,
        "final_pixel_mse": float(pixel_mse_by_step[-1]),
        "pixel_mse_by_step": pixel_mse_by_step.astype(float).tolist(),
        "token_accuracy_by_step": token_accuracy_by_step.numpy().astype(float).tolist(),
        "dream_done_probability_by_step": predicted_done_probs.astype(float).tolist(),
        "real_done_by_step": real_dones.astype(bool).tolist(),
        "comparison_frames": comparison_frames,
    }


def _planner_exploitation_score(path: str | Path | None) -> float | None:
    if path is None:
        return None
    metrics_path = Path(path)
    if not metrics_path.exists():
        return None
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    results = metrics.get("results", {})
    if not isinstance(results, dict) or "random_policy" not in results:
        return None
    random_return = float(results["random_policy"].get("mean_return", 0.0))
    dream_returns = [
        float(value.get("mean_return", 0.0))
        for key, value in results.items()
        if key.startswith("dream_") and isinstance(value, dict)
    ]
    if not dream_returns:
        return None
    return float(max(dream_returns) - random_return)


def _write_gap_plot(failures: list[dict[str, Any]], path: Path) -> None:
    starts = [int(item["start_index"]) for item in failures]
    scores = [float(item["failure_score"]) for item in failures]
    pixel_mse = [float(item["mean_pixel_mse"]) for item in failures]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(starts, scores, label="failure score", linewidth=1.5)
    ax.plot(starts, pixel_mse, label="mean pixel MSE", linewidth=1.5)
    ax.set_xlabel("Start index")
    ax.set_ylabel("Gap")
    ax.set_title("Dream-Real Failure Ranking")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_summary(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Failure Analysis",
        "",
        f"- Scanned windows: {summary['scanned_windows']}",
        f"- Horizon: {summary['horizon']}",
        f"- Planner exploitation score: {summary['planner_exploitation_score']}",
        "",
        "## Top Failures",
        "",
    ]
    for index, failure in enumerate(summary["failures"], start=1):
        lines.extend(
            [
                f"### Failure {index:03d}",
                "",
                f"- Start index: {failure['start_index']}",
                f"- Failure score: {failure['failure_score']:.5f}",
                f"- Predicted return - real return: "
                f"{failure['predicted_return_minus_real_return']:.5f}",
                f"- Mean pixel MSE: {failure['mean_pixel_mse']:.5f}",
                f"- Latent drift at horizon: {failure['latent_drift_at_horizon']:.5f}",
                f"- Video: {failure['video_path']}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dynamics", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--planner-metrics", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--fps", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = mine_failures(
        data_path=args.data,
        dynamics_path=args.dynamics,
        tokenizer_path=args.tokenizer,
        out=args.out,
        horizon=args.horizon,
        top_k=args.top_k,
        stride=args.stride,
        batch_size=args.batch_size,
        planner_metrics_path=args.planner_metrics,
        device_name=args.device,
        fps=args.fps,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
