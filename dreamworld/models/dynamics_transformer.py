"""Action-conditioned transformer dynamics for VQ-VAE token worlds."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class DynamicsTransformerConfig:
    context_length: int = 32
    token_grid_size: int = 8
    num_embeddings: int = 512
    token_embedding_dim: int = 64
    action_dim: int = 3
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1


class TokenDynamicsTransformer(nn.Module):
    """Predict next visual tokens, reward, and done from token/action history."""

    def __init__(self, config: DynamicsTransformerConfig) -> None:
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.config = config
        self.grid_tokens = config.token_grid_size * config.token_grid_size

        self.token_embedding = nn.Embedding(config.num_embeddings, config.token_embedding_dim)
        self.frame_projection = nn.Linear(
            self.grid_tokens * config.token_embedding_dim,
            config.d_model,
        )
        self.action_projection = nn.Linear(config.action_dim, config.d_model)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.context_length, config.d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=4 * config.d_model,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.norm = nn.LayerNorm(config.d_model)
        self.token_head = nn.Linear(config.d_model, self.grid_tokens * config.num_embeddings)
        self.reward_head = nn.Linear(config.d_model, 1)
        self.done_head = nn.Linear(config.d_model, 1)

        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return next-token logits, reward predictions, and done logits.

        Args:
            tokens: Long tensor [B, L, H, W] containing current frame tokens.
            actions: Float tensor [B, L, 3] containing actions taken from each frame.
        """
        if tokens.ndim != 4:
            raise ValueError(f"tokens must have shape [B, L, H, W], got {tokens.shape}")
        if actions.ndim != 3:
            raise ValueError(f"actions must have shape [B, L, A], got {actions.shape}")
        batch, length, height, width = tokens.shape
        if length > self.config.context_length:
            raise ValueError(
                f"Input length {length} exceeds context_length={self.config.context_length}"
            )
        if (height, width) != (self.config.token_grid_size, self.config.token_grid_size):
            raise ValueError(
                f"Expected token grid {self.config.token_grid_size}x"
                f"{self.config.token_grid_size}, got {height}x{width}"
            )

        embedded_tokens = self.token_embedding(tokens).flatten(start_dim=2)
        frame_features = self.frame_projection(embedded_tokens)
        action_features = self.action_projection(actions)
        sequence = frame_features + action_features + self.position_embedding[:, :length]

        causal_mask = torch.triu(
            torch.ones(length, length, device=tokens.device, dtype=torch.bool),
            diagonal=1,
        )
        hidden = self.transformer(sequence, mask=causal_mask)
        hidden = self.norm(hidden)

        token_logits = self.token_head(hidden).view(
            batch,
            length,
            height,
            width,
            self.config.num_embeddings,
        )
        rewards = self.reward_head(hidden).squeeze(-1)
        done_logits = self.done_head(hidden).squeeze(-1)
        return token_logits, rewards, done_logits

    @torch.no_grad()
    def imagine_step(
        self,
        tokens: torch.Tensor,
        actions: torch.Tensor,
        *,
        sample: bool = False,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict the next token grid from the last position in a context window."""
        token_logits, rewards, done_logits = self(tokens, actions)
        last_logits = token_logits[:, -1]
        if sample:
            flat_logits = last_logits.flatten(end_dim=-2) / max(temperature, 1e-6)
            next_tokens = torch.multinomial(flat_logits.softmax(dim=-1), num_samples=1)
            next_tokens = next_tokens.view(
                tokens.shape[0],
                self.config.token_grid_size,
                self.config.token_grid_size,
            )
        else:
            next_tokens = last_logits.argmax(dim=-1)
        next_rewards = rewards[:, -1]
        next_done_probs = done_logits[:, -1].sigmoid()
        return next_tokens, next_rewards, next_done_probs
