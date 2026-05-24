"""VQ-VAE tokenizer for discrete visual latents."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class VQVAEConfig:
    image_size: int = 64
    embedding_dim: int = 64
    num_embeddings: int = 512
    commitment_cost: float = 0.25
    channels: int = 3


class VectorQuantizer(nn.Module):
    """Nearest-neighbor vector quantization with straight-through gradients."""

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # BCHW -> BHWC so each spatial vector is a code lookup.
        z_e_bhwc = z_e.permute(0, 2, 3, 1).contiguous()
        flat = z_e_bhwc.view(-1, self.embedding_dim)

        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1)
        )
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(encoding_indices).view_as(z_e_bhwc)

        codebook_loss = F.mse_loss(quantized, z_e_bhwc.detach())
        commitment_loss = F.mse_loss(z_e_bhwc, quantized.detach())
        loss = codebook_loss + self.commitment_cost * commitment_loss

        quantized = z_e_bhwc + (quantized - z_e_bhwc).detach()
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        indices = encoding_indices.view(z_e.shape[0], z_e.shape[2], z_e.shape[3])
        return quantized, loss, indices


class VQVAE(nn.Module):
    """Convolutional VQ-VAE with an 8x8 token grid for 64x64 frames."""

    def __init__(self, config: VQVAEConfig) -> None:
        super().__init__()
        if config.image_size != 64:
            raise ValueError("VQVAE currently supports image_size=64")
        self.config = config

        self.encoder = nn.Sequential(
            nn.Conv2d(config.channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, config.embedding_dim, kernel_size=3, stride=1, padding=1),
        )
        self.quantizer = VectorQuantizer(
            num_embeddings=config.num_embeddings,
            embedding_dim=config.embedding_dim,
            commitment_cost=config.commitment_cost,
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(config.embedding_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, config.channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_e = self.encoder(x)
        return self.quantizer(z_e)

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_q)

    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        z_q = self.quantizer.embedding(tokens).permute(0, 3, 1, 2).contiguous()
        return self.decode(z_q)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_q, vq_loss, tokens = self.encode(x)
        reconstruction = self.decode(z_q)
        return reconstruction, vq_loss, tokens


def codebook_metrics(tokens: torch.Tensor, num_embeddings: int) -> dict[str, float]:
    """Compute perplexity and usage metrics from token indices."""
    flat = tokens.detach().flatten().cpu()
    counts = torch.bincount(flat, minlength=num_embeddings).float()
    probs = counts / counts.sum().clamp_min(1.0)
    nonzero = probs > 0
    entropy = -(probs[nonzero] * torch.log(probs[nonzero])).sum()
    perplexity = torch.exp(entropy)
    used_codes = int(nonzero.sum().item())
    return {
        "codebook_perplexity": float(perplexity.item()),
        "used_codes": float(used_codes),
        "dead_code_fraction": float(1.0 - used_codes / num_embeddings),
    }
