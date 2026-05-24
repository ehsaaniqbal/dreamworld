"""Model-predictive controller backed by learned VQ-VAE token dynamics."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch

from dreamtrack.data.preprocess import FramePreprocessor
from dreamtrack.eval.eval_dynamics import load_dynamics_checkpoint
from dreamtrack.plan.cem import cem_plan
from dreamtrack.plan.random_shooting import random_shooting_plan
from dreamtrack.train.train_autoencoder import choose_device
from dreamtrack.train.train_dynamics import load_vqvae_checkpoint


class DreamMPCAgent:
    """Encode real frames, imagine candidate futures, and execute the best first action."""

    def __init__(
        self,
        *,
        tokenizer_path: str | Path | None,
        dynamics_path: str | Path,
        planner: str = "random",
        horizon: int = 10,
        candidates: int = 128,
        elites: int = 16,
        iterations: int = 3,
        image_size: int = 64,
        done_penalty: float = 10.0,
        brake_penalty: float = 1.5,
        steering_penalty: float = 0.05,
        low_gas_penalty: float = 0.05,
        smoothness_penalty: float = 0.1,
        device_name: str | None = None,
        seed: int = 0,
    ) -> None:
        self.device = choose_device(device_name)
        self.dynamics, checkpoint = load_dynamics_checkpoint(dynamics_path, self.device)
        resolved_tokenizer = tokenizer_path or checkpoint["tokenizer_checkpoint"]
        self.tokenizer = load_vqvae_checkpoint(resolved_tokenizer, self.device)
        self.planner = planner
        self.horizon = horizon
        self.candidates = candidates
        self.elites = elites
        self.iterations = iterations
        self.done_penalty = done_penalty
        self.brake_penalty = brake_penalty
        self.steering_penalty = steering_penalty
        self.low_gas_penalty = low_gas_penalty
        self.smoothness_penalty = smoothness_penalty
        self.preprocessor = FramePreprocessor(image_size=image_size)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.context_tokens: deque[torch.Tensor] = deque(maxlen=self.dynamics.config.context_length)
        self.context_actions: deque[torch.Tensor] = deque(
            maxlen=self.dynamics.config.context_length
        )
        self.last_debug: dict[str, object] = {}
        self.last_top_sequences: torch.Tensor | None = None
        self.last_plan_context_tokens: torch.Tensor | None = None
        self.last_plan_context_actions: torch.Tensor | None = None

    @torch.no_grad()
    def act(self, observation: np.ndarray) -> np.ndarray:
        token_grid = self.encode_observation(observation)
        if not self.context_tokens:
            zero_action = torch.zeros(3, device=self.device)
            for _ in range(self.dynamics.config.context_length):
                self.context_tokens.append(token_grid)
                self.context_actions.append(zero_action)
        else:
            self.context_tokens.append(token_grid)

        tokens = torch.stack(tuple(self.context_tokens), dim=0)
        actions = torch.stack(tuple(self.context_actions), dim=0)
        if self.planner == "cem":
            plan = cem_plan(
                dynamics=self.dynamics,
                context_tokens=tokens,
                context_actions=actions,
                candidates=self.candidates,
                horizon=self.horizon,
                elites=self.elites,
                iterations=self.iterations,
                generator=self.generator,
                done_penalty=self.done_penalty,
                brake_penalty=self.brake_penalty,
                steering_penalty=self.steering_penalty,
                low_gas_penalty=self.low_gas_penalty,
                smoothness_penalty=self.smoothness_penalty,
            )
        elif self.planner == "random":
            plan = random_shooting_plan(
                dynamics=self.dynamics,
                context_tokens=tokens,
                context_actions=actions,
                candidates=self.candidates,
                horizon=self.horizon,
                generator=self.generator,
                done_penalty=self.done_penalty,
                brake_penalty=self.brake_penalty,
                steering_penalty=self.steering_penalty,
                low_gas_penalty=self.low_gas_penalty,
                smoothness_penalty=self.smoothness_penalty,
            )
        else:
            raise ValueError("planner must be 'random' or 'cem'")

        action_tensor = plan["action"].detach().to(self.device)
        self.last_plan_context_tokens = tokens.detach().clone()
        self.last_plan_context_actions = actions.detach().clone()
        self.context_actions.append(action_tensor)
        self.last_top_sequences = plan["top_sequences"].to(self.device)
        self.last_debug = {
            "score": plan["score"],
            "best_index": plan["best_index"],
            "sequence": plan["sequence"].detach().cpu().numpy().astype(float).tolist(),
            "top_scores": plan["top_scores"].detach().cpu().numpy().astype(float).tolist(),
        }
        return action_tensor.detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def encode_observation(self, observation: np.ndarray) -> torch.Tensor:
        frame = self.preprocessor.to_uint8(observation).astype(np.float32) / 255.0
        batch = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(self.device)
        _z_q, _vq_loss, tokens = self.tokenizer.encode(batch)
        return tokens.squeeze(0)

    @torch.no_grad()
    def candidate_plan_video(self, max_candidates: int = 4) -> np.ndarray:
        """Render the latest top imagined candidate plans as a side-by-side video."""
        if self.last_top_sequences is None:
            raise ValueError("No planner candidates available. Call act() first.")
        if self.last_plan_context_tokens is None or self.last_plan_context_actions is None:
            raise ValueError("No planner context available. Call act() first.")
        sequences = self.last_top_sequences[:max_candidates].to(self.device)
        token_context = self.last_plan_context_tokens
        action_context = self.last_plan_context_actions
        token_context = token_context.unsqueeze(0).repeat(sequences.shape[0], 1, 1, 1)
        action_context = action_context.unsqueeze(0).repeat(sequences.shape[0], 1, 1)
        frames: list[np.ndarray] = []
        pad = np.full((self.tokenizer.config.image_size, 3, 3), 255, dtype=np.uint8)

        for step in range(sequences.shape[1]):
            planned_action = sequences[:, step]
            action_context = torch.cat([action_context[:, 1:], planned_action.unsqueeze(1)], dim=1)
            next_tokens, _rewards, _done_probs = self.dynamics.imagine_step(
                token_context,
                action_context,
            )
            decoded = self.tokenizer.decode_tokens(next_tokens).detach().cpu()
            decoded = decoded.clamp(0.0, 1.0).permute(0, 2, 3, 1).numpy()
            decoded_uint8 = np.clip(decoded * 255.0, 0, 255).astype(np.uint8)
            tiles: list[np.ndarray] = []
            for index, frame in enumerate(decoded_uint8):
                if index:
                    tiles.append(pad)
                tiles.append(frame)
            frames.append(np.concatenate(tiles, axis=1))
            token_context = torch.cat([token_context[:, 1:], next_tokens.unsqueeze(1)], dim=1)

        return np.stack(frames)
