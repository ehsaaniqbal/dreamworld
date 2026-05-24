"""Collect CarRacing rollouts for world-model training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from dreamworld.config import load_config
from dreamworld.data.dataset import load_rollout_dataset, validate_rollout_dataset
from dreamworld.data.preprocess import FramePreprocessor
from dreamworld.viz.video import write_video


def random_policy(env: gym.Env, _observation: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample a continuous CarRacing action."""
    if hasattr(env.action_space, "sample"):
        return np.asarray(env.action_space.sample(), dtype=np.float32)
    return rng.uniform([-1.0, 0.0, 0.0], [1.0, 1.0, 1.0]).astype(np.float32)


def heuristic_policy(
    _env: gym.Env,
    observation: np.ndarray,
    _rng: np.random.Generator,
) -> np.ndarray:
    """A tiny visual heuristic that steers toward green track pixels near the car."""
    frame = np.asarray(observation)
    lower = frame[int(frame.shape[0] * 0.55) :, :, :]
    green_mask = (lower[..., 1] > lower[..., 0] + 15) & (lower[..., 1] > lower[..., 2] + 15)

    if green_mask.any():
        columns = np.nonzero(green_mask)[1]
        track_center = float(columns.mean()) / max(frame.shape[1] - 1, 1)
        steer = np.clip((track_center - 0.5) * 2.5, -1.0, 1.0)
        gas = 0.55
        brake = 0.0
    else:
        steer = 0.0
        gas = 0.15
        brake = 0.25
    return np.asarray([steer, gas, brake], dtype=np.float32)


def hybrid_policy(env: gym.Env, observation: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Heuristic driving with enough random exploration for dynamics learning."""
    if rng.random() < 0.2:
        return random_policy(env, observation, rng)

    action = heuristic_policy(env, observation, rng)
    action[0] = np.clip(action[0] + rng.normal(0.0, 0.18), -1.0, 1.0)
    action[1] = np.clip(action[1] + rng.normal(0.0, 0.12), 0.0, 1.0)
    if rng.random() < 0.05:
        action[2] = np.clip(rng.uniform(0.0, 0.35), 0.0, 1.0)
    return action.astype(np.float32)


POLICIES = {
    "random": random_policy,
    "heuristic": heuristic_policy,
    "hybrid": hybrid_policy,
}


def _nested_get(config: dict[str, Any], keys: tuple[str, ...], default: Any) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _stats(
    rewards: np.ndarray,
    actions: np.ndarray,
    episode_lengths: np.ndarray,
    episode_returns: list[float],
    image_size: int,
    policy: str,
) -> dict[str, Any]:
    return {
        "policy": policy,
        "image_size": image_size,
        "num_episodes": int(len(episode_lengths)),
        "num_frames": int(rewards.shape[0]),
        "mean_reward": float(rewards.mean()),
        "min_reward": float(rewards.min()),
        "max_reward": float(rewards.max()),
        "episode_lengths": [int(value) for value in episode_lengths.tolist()],
        "episode_returns": [float(value) for value in episode_returns],
        "mean_episode_return": float(np.mean(episode_returns)),
        "action_mean": actions.mean(axis=0).astype(float).tolist(),
        "action_std": actions.std(axis=0).astype(float).tolist(),
        "action_min": actions.min(axis=0).astype(float).tolist(),
        "action_max": actions.max(axis=0).astype(float).tolist(),
    }


def collect_rollouts(
    *,
    episodes: int,
    policy: str,
    image_size: int,
    out: str | Path,
    env_id: str = "CarRacing-v3",
    seed: int = 0,
    max_steps: int | None = None,
    video_fps: int = 30,
    video_frames: int = 600,
) -> dict[str, Any]:
    """Collect rollouts and save rollouts.npz, sample.mp4, and stats.json."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy '{policy}'. Choices: {sorted(POLICIES)}")

    output_dir = Path(out)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    preprocessor = FramePreprocessor(image_size=image_size)
    policy_fn = POLICIES[policy]

    obs_frames: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    dones: list[bool] = []
    episode_ids: list[int] = []
    episode_start_indices: list[int] = []
    episode_lengths: list[int] = []
    episode_returns: list[float] = []

    env = gym.make(env_id, render_mode="rgb_array")
    env.action_space.seed(seed)
    try:
        for episode_id in tqdm(range(episodes), desc="Collecting episodes"):
            observation, _info = env.reset(seed=seed + episode_id)
            episode_start_indices.append(len(obs_frames))
            episode_return = 0.0
            steps = 0

            while True:
                action = policy_fn(env, observation, rng)
                obs_frames.append(preprocessor.to_uint8(observation))
                actions.append(action.astype(np.float32))
                episode_ids.append(episode_id)

                observation, reward, terminated, truncated, _step_info = env.step(action)
                done = bool(terminated or truncated)
                if max_steps is not None and steps + 1 >= max_steps:
                    done = True

                rewards.append(float(reward))
                dones.append(done)
                episode_return += float(reward)
                steps += 1

                if done:
                    episode_lengths.append(steps)
                    episode_returns.append(episode_return)
                    break
    finally:
        env.close()

    obs_array = np.stack(obs_frames).astype(np.uint8)
    action_array = np.stack(actions).astype(np.float32)
    reward_array = np.asarray(rewards, dtype=np.float32)
    done_array = np.asarray(dones, dtype=np.bool_)
    episode_id_array = np.asarray(episode_ids, dtype=np.int32)
    episode_start_array = np.asarray(episode_start_indices, dtype=np.int32)
    episode_length_array = np.asarray(episode_lengths, dtype=np.int32)

    validate_rollout_dataset(
        dataset=load_rollout_dataset_from_arrays(
            obs_array,
            action_array,
            reward_array,
            done_array,
            episode_id_array,
            episode_start_array,
            episode_length_array,
        ),
        image_size=image_size,
    )

    dataset_path = output_dir / "rollouts.npz"
    np.savez_compressed(
        dataset_path,
        obs=obs_array,
        actions=action_array,
        rewards=reward_array,
        dones=done_array,
        episode_ids=episode_id_array,
        episode_start_indices=episode_start_array,
        episode_lengths=episode_length_array,
    )

    dataset = load_rollout_dataset(dataset_path)
    validate_rollout_dataset(dataset, image_size=image_size)

    sample_frames = dataset.obs[: min(video_frames, dataset.num_frames)]
    write_video(sample_frames, output_dir / "sample.mp4", fps=video_fps)

    stats = _stats(
        rewards=dataset.rewards,
        actions=dataset.actions,
        episode_lengths=dataset.episode_lengths,
        episode_returns=episode_returns,
        image_size=image_size,
        policy=policy,
    )
    stats["dataset_path"] = str(dataset_path)
    stats["video_path"] = str(output_dir / "sample.mp4")

    with (output_dir / "stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
        handle.write("\n")

    return stats


def load_rollout_dataset_from_arrays(
    obs: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    episode_ids: np.ndarray,
    episode_start_indices: np.ndarray,
    episode_lengths: np.ndarray,
):
    """Create a RolloutDataset without a temporary file."""
    from dreamworld.data.dataset import RolloutDataset

    return RolloutDataset(
        obs=obs,
        actions=actions,
        rewards=rewards,
        dones=dones,
        episode_ids=episode_ids,
        episode_start_indices=episode_start_indices,
        episode_lengths=episode_lengths,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("dreamworld/configs/collect.yaml"))
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--policy", choices=sorted(POLICIES), default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--video-frames", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config) if args.config else {}

    stats = collect_rollouts(
        episodes=args.episodes or int(_nested_get(config, ("data", "episodes"), 10)),
        policy=args.policy or str(_nested_get(config, ("data", "policy"), "random")),
        image_size=args.image_size or int(_nested_get(config, ("data", "image_size"), 64)),
        out=args.out or Path(str(_nested_get(config, ("data", "out"), "data/car_racing_debug"))),
        env_id=str(_nested_get(config, ("env", "id"), "CarRacing-v3")),
        seed=args.seed if args.seed is not None else int(_nested_get(config, ("seed",), 0)),
        max_steps=args.max_steps,
        video_fps=args.video_fps,
        video_frames=args.video_frames,
    )

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
