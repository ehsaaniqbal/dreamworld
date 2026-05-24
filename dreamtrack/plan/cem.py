"""Cross-entropy method action search inside the learned token world."""

from __future__ import annotations

import torch

from dreamtrack.plan.random_shooting import (
    ACTION_HIGH,
    ACTION_LOW,
    ACTION_PRIOR_MEAN,
    ACTION_PRIOR_STD,
    score_action_sequences,
)


@torch.no_grad()
def cem_plan(
    *,
    dynamics,
    context_tokens: torch.Tensor,
    context_actions: torch.Tensor,
    candidates: int,
    horizon: int,
    elites: int,
    iterations: int,
    generator: torch.Generator | None = None,
    done_penalty: float = 10.0,
    brake_penalty: float = 1.5,
    steering_penalty: float = 0.05,
    low_gas_penalty: float = 0.05,
    smoothness_penalty: float = 0.1,
    min_std: float = 0.05,
) -> dict[str, torch.Tensor | float | int]:
    """Choose an action sequence by repeatedly fitting elites."""
    device = context_tokens.device
    low = ACTION_LOW.to(device)
    high = ACTION_HIGH.to(device)
    mean = ACTION_PRIOR_MEAN.to(device).repeat(horizon, 1)
    std = ACTION_PRIOR_STD.to(device).repeat(horizon, 1)
    elite_count = min(elites, candidates)

    best_actions: torch.Tensor | None = None
    best_scores: torch.Tensor | None = None
    best_index = 0

    for _iteration in range(iterations):
        noise = torch.randn(candidates, horizon, 3, device=device, generator=generator)
        actions = torch.clamp(mean.unsqueeze(0) + std.unsqueeze(0) * noise, low, high)
        scores, _reward_rollout = score_action_sequences(
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
        elite_indices = scores.topk(elite_count).indices
        elite_actions = actions[elite_indices]
        mean = elite_actions.mean(dim=0)
        std = elite_actions.std(dim=0).clamp_min(min_std)
        best_actions = actions
        best_scores = scores
        best_index = int(scores.argmax().item())

    if best_actions is None or best_scores is None:
        raise ValueError("CEM requires at least one iteration")
    top_count = min(8, candidates)
    top = best_scores.topk(top_count)
    return {
        "action": best_actions[best_index, 0],
        "sequence": best_actions[best_index],
        "score": float(best_scores[best_index].detach().cpu()),
        "candidate_scores": best_scores.detach().cpu(),
        "top_sequences": best_actions[top.indices].detach().cpu(),
        "top_scores": top.values.detach().cpu(),
        "best_index": best_index,
    }
