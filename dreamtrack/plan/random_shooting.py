"""Random-shooting action search inside the learned token world."""

from __future__ import annotations

import torch

ACTION_LOW = torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float32)
ACTION_HIGH = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
ACTION_PRIOR_MEAN = torch.tensor([0.0, 0.55, 0.0], dtype=torch.float32)
ACTION_PRIOR_STD = torch.tensor([0.45, 0.22, 0.08], dtype=torch.float32)


def sample_action_sequences(
    *,
    candidates: int,
    horizon: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample drivable CarRacing action sequences [N, H, 3]."""
    low = ACTION_LOW.to(device)
    high = ACTION_HIGH.to(device)
    mean = ACTION_PRIOR_MEAN.to(device)
    std = ACTION_PRIOR_STD.to(device)
    noise = torch.randn(
        candidates,
        horizon,
        3,
        device=device,
        generator=generator,
    )
    actions = torch.clamp(mean + noise * std, low, high)

    explore_count = max(1, candidates // 8)
    unit = torch.rand(
        explore_count,
        horizon,
        3,
        device=device,
        generator=generator,
    )
    actions[:explore_count] = low + unit * (high - low)
    brake_mask = torch.rand(
        candidates,
        horizon,
        device=device,
        generator=generator,
    ) < 0.9
    actions[..., 2] = torch.where(brake_mask, actions[..., 2].mul(0.1), actions[..., 2])
    return actions


@torch.no_grad()
def score_action_sequences(
    *,
    dynamics,
    context_tokens: torch.Tensor,
    context_actions: torch.Tensor,
    action_sequences: torch.Tensor,
    done_penalty: float = 10.0,
    brake_penalty: float = 1.5,
    steering_penalty: float = 0.05,
    low_gas_penalty: float = 0.05,
    smoothness_penalty: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll candidate action sequences in imagination and return scores and rewards."""
    candidates, horizon, _action_dim = action_sequences.shape
    token_context = context_tokens.unsqueeze(0).repeat(candidates, 1, 1, 1)
    action_context = context_actions.unsqueeze(0).repeat(candidates, 1, 1)
    scores = torch.zeros(candidates, device=action_sequences.device)
    reward_rollout = torch.zeros(candidates, horizon, device=action_sequences.device)

    for step in range(horizon):
        planned_action = action_sequences[:, step]
        action_context = torch.cat([action_context[:, 1:], planned_action.unsqueeze(1)], dim=1)
        next_tokens, rewards, done_probs = dynamics.imagine_step(token_context, action_context)
        reward_rollout[:, step] = rewards
        action_cost = (
            brake_penalty * planned_action[:, 2]
            + steering_penalty * planned_action[:, 0].abs()
            + low_gas_penalty * (0.45 - planned_action[:, 1]).clamp_min(0.0)
        )
        if step > 0:
            action_delta = planned_action - action_sequences[:, step - 1]
            action_cost = action_cost + smoothness_penalty * action_delta.abs().mean(dim=-1)
        scores = scores + rewards - done_penalty * done_probs - action_cost
        token_context = torch.cat([token_context[:, 1:], next_tokens.unsqueeze(1)], dim=1)

    return scores, reward_rollout


@torch.no_grad()
def random_shooting_plan(
    *,
    dynamics,
    context_tokens: torch.Tensor,
    context_actions: torch.Tensor,
    candidates: int,
    horizon: int,
    generator: torch.Generator | None = None,
    done_penalty: float = 10.0,
    brake_penalty: float = 1.5,
    steering_penalty: float = 0.05,
    low_gas_penalty: float = 0.05,
    smoothness_penalty: float = 0.1,
) -> dict[str, torch.Tensor | float | int]:
    """Choose the best first action from random imagined futures."""
    actions = sample_action_sequences(
        candidates=candidates,
        horizon=horizon,
        device=context_tokens.device,
        generator=generator,
    )
    scores, reward_rollout = score_action_sequences(
        dynamics=dynamics,
        context_tokens=context_tokens,
        context_actions=context_actions,
        action_sequences=actions,
        done_penalty=done_penalty,
        brake_penalty=brake_penalty,
        steering_penalty=steering_penalty,
        low_gas_penalty=low_gas_penalty,
        smoothness_penalty=smoothness_penalty,
    )
    best_index = int(scores.argmax().item())
    top_count = min(8, candidates)
    top = scores.topk(top_count)
    return {
        "action": actions[best_index, 0],
        "sequence": actions[best_index],
        "score": float(scores[best_index].detach().cpu()),
        "candidate_scores": scores.detach().cpu(),
        "candidate_rewards": reward_rollout.detach().cpu(),
        "top_sequences": actions[top.indices].detach().cpu(),
        "top_scores": top.values.detach().cpu(),
        "best_index": best_index,
    }
