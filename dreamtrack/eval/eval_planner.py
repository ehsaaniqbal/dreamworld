"""Evaluate random, heuristic, and dream-planning agents in real CarRacing."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from dreamtrack.data.collect_rollouts import heuristic_policy, random_policy
from dreamtrack.plan.mpc_agent import DreamMPCAgent
from dreamtrack.viz.video import write_video

PolicyFn = Callable[[gym.Env, np.ndarray, np.random.Generator], np.ndarray]


def eval_planner(
    *,
    env_id: str,
    tokenizer_path: str | Path | None,
    dynamics_path: str | Path,
    planner: str,
    episodes: int,
    max_steps: int,
    horizon: int,
    candidates: int,
    elites: int,
    iterations: int,
    done_penalty: float,
    brake_penalty: float,
    steering_penalty: float,
    low_gas_penalty: float,
    smoothness_penalty: float,
    out: str | Path,
    seed: int = 0,
    device_name: str | None = None,
    video_fps: int = 30,
) -> dict[str, object]:
    """Run baseline policies plus the requested dream planner."""
    output_dir = Path(out)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, object]] = {}
    results["random_policy"] = _eval_policy(
        env_id=env_id,
        name="random_policy",
        policy_fn=random_policy,
        episodes=episodes,
        max_steps=max_steps,
        out=output_dir,
        seed=seed,
        video_fps=video_fps,
    )
    results["heuristic_policy"] = _eval_policy(
        env_id=env_id,
        name="heuristic_policy",
        policy_fn=heuristic_policy,
        episodes=episodes,
        max_steps=max_steps,
        out=output_dir,
        seed=seed + 10_000,
        video_fps=video_fps,
    )

    dream_planners = ["random", "cem"] if planner == "all" else [planner]
    for planner_name in dream_planners:
        key = f"dream_{planner_name}"
        results[key] = _eval_dream_agent(
            env_id=env_id,
            name=key,
            tokenizer_path=tokenizer_path,
            dynamics_path=dynamics_path,
            planner=planner_name,
            episodes=episodes,
            max_steps=max_steps,
            horizon=horizon,
            candidates=candidates,
            elites=elites,
            iterations=iterations,
            done_penalty=done_penalty,
            brake_penalty=brake_penalty,
            steering_penalty=steering_penalty,
            low_gas_penalty=low_gas_penalty,
            smoothness_penalty=smoothness_penalty,
            out=output_dir,
            seed=seed + 20_000,
            device_name=device_name,
            video_fps=video_fps,
        )

    summary = {
        "env_id": env_id,
        "dynamics_checkpoint": str(dynamics_path),
        "tokenizer_checkpoint": str(tokenizer_path) if tokenizer_path else None,
        "episodes": episodes,
        "max_steps": max_steps,
        "horizon": horizon,
        "candidates": candidates,
        "done_penalty": done_penalty,
        "brake_penalty": brake_penalty,
        "steering_penalty": steering_penalty,
        "low_gas_penalty": low_gas_penalty,
        "smoothness_penalty": smoothness_penalty,
        "results": results,
    }
    _write_reward_plot(summary, output_dir / "reward_curve.png")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return summary


def _eval_policy(
    *,
    env_id: str,
    name: str,
    policy_fn: PolicyFn,
    episodes: int,
    max_steps: int,
    out: Path,
    seed: int,
    video_fps: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    env = gym.make(env_id, render_mode="rgb_array")
    env.action_space.seed(seed)
    returns: list[float] = []
    lengths: list[int] = []
    first_episode_frames: list[np.ndarray] = []
    try:
        for episode in range(episodes):
            observation, _info = env.reset(seed=seed + episode)
            total_reward = 0.0
            length = 0
            for _step in range(max_steps):
                if episode == 0:
                    first_episode_frames.append(np.asarray(observation, dtype=np.uint8))
                action = policy_fn(env, observation, rng)
                observation, reward, terminated, truncated, _step_info = env.step(action)
                total_reward += float(reward)
                length += 1
                if terminated or truncated:
                    break
            returns.append(total_reward)
            lengths.append(length)
    finally:
        env.close()

    if first_episode_frames:
        write_video(np.stack(first_episode_frames), out / f"{name}_episode_000.mp4", fps=video_fps)
    return _policy_metrics(name, returns, lengths, out / f"{name}_episode_000.mp4")


def _eval_dream_agent(
    *,
    env_id: str,
    name: str,
    tokenizer_path: str | Path | None,
    dynamics_path: str | Path,
    planner: str,
    episodes: int,
    max_steps: int,
    horizon: int,
    candidates: int,
    elites: int,
    iterations: int,
    done_penalty: float,
    brake_penalty: float,
    steering_penalty: float,
    low_gas_penalty: float,
    smoothness_penalty: float,
    out: Path,
    seed: int,
    device_name: str | None,
    video_fps: int,
) -> dict[str, object]:
    returns: list[float] = []
    lengths: list[int] = []
    first_episode_frames: list[np.ndarray] = []
    debug: list[dict[str, object]] = []
    candidate_debug_path = out / f"{name}_candidate_plan_debug.mp4"
    env = gym.make(env_id, render_mode="rgb_array")
    env.action_space.seed(seed)
    try:
        for episode in range(episodes):
            agent = DreamMPCAgent(
                tokenizer_path=tokenizer_path,
                dynamics_path=dynamics_path,
                planner=planner,
                horizon=horizon,
                candidates=candidates,
                elites=elites,
                iterations=iterations,
                done_penalty=done_penalty,
                brake_penalty=brake_penalty,
                steering_penalty=steering_penalty,
                low_gas_penalty=low_gas_penalty,
                smoothness_penalty=smoothness_penalty,
                device_name=device_name,
                seed=seed + episode,
            )
            observation, _info = env.reset(seed=seed + episode)
            total_reward = 0.0
            for step in range(max_steps):
                if episode == 0:
                    first_episode_frames.append(np.asarray(observation, dtype=np.uint8))
                action = agent.act(observation)
                if episode == 0 and step < 20:
                    debug.append({"step": step, **agent.last_debug})
                if episode == 0 and step == 0:
                    candidate_video = agent.candidate_plan_video()
                    write_video(candidate_video, candidate_debug_path, fps=video_fps)
                    canonical_path = out / "candidate_plan_debug.mp4"
                    if not canonical_path.exists() or canonical_path.stat().st_size == 0:
                        write_video(candidate_video, canonical_path, fps=video_fps)
                observation, reward, terminated, truncated, _step_info = env.step(action)
                total_reward += float(reward)
                if terminated or truncated:
                    break
            returns.append(total_reward)
            lengths.append(step + 1)
    finally:
        env.close()

    if first_episode_frames:
        write_video(np.stack(first_episode_frames), out / f"{name}_episode_000.mp4", fps=video_fps)
    metrics = _policy_metrics(name, returns, lengths, out / f"{name}_episode_000.mp4")
    metrics["planner_debug_first_episode"] = debug
    metrics["candidate_plan_debug_video"] = str(candidate_debug_path)
    return metrics


def _policy_metrics(
    name: str,
    returns: list[float],
    lengths: list[int],
    video_path: Path,
) -> dict[str, object]:
    return {
        "name": name,
        "episode_returns": returns,
        "episode_lengths": lengths,
        "mean_return": float(np.mean(returns)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
        "mean_length": float(np.mean(lengths)),
        "video_path": str(video_path),
    }


def _write_reward_plot(summary: dict[str, object], path: Path) -> None:
    results = summary["results"]
    assert isinstance(results, dict)
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = []
    values = []
    for name, metrics in results.items():
        assert isinstance(metrics, dict)
        labels.append(name)
        values.append(float(metrics["mean_return"]))
    ax.bar(labels, values)
    ax.set_ylabel("Mean Episode Return")
    ax.set_title("CarRacing Planner Comparison")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", type=str, default="CarRacing-v3")
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--dynamics", type=Path, required=True)
    parser.add_argument("--planner", choices=["random", "cem", "all"], default="cem")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--candidates", type=int, default=128)
    parser.add_argument("--elites", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--done-penalty", type=float, default=10.0)
    parser.add_argument("--brake-penalty", type=float, default=1.5)
    parser.add_argument("--steering-penalty", type=float, default=0.05)
    parser.add_argument("--low-gas-penalty", type=float, default=0.05)
    parser.add_argument("--smoothness-penalty", type=float, default=0.1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--video-fps", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = eval_planner(
        env_id=args.env,
        tokenizer_path=args.tokenizer,
        dynamics_path=args.dynamics,
        planner=args.planner,
        episodes=args.episodes,
        max_steps=args.max_steps,
        horizon=args.horizon,
        candidates=args.candidates,
        elites=args.elites,
        iterations=args.iterations,
        done_penalty=args.done_penalty,
        brake_penalty=args.brake_penalty,
        steering_penalty=args.steering_penalty,
        low_gas_penalty=args.low_gas_penalty,
        smoothness_penalty=args.smoothness_penalty,
        out=args.out,
        seed=args.seed,
        device_name=args.device,
        video_fps=args.video_fps,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
