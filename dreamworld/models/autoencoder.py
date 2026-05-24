"""Convolutional autoencoder for 64x64 RGB frame reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class AutoencoderConfig:
    image_size: int = 64
    latent_dim: int = 256
    channels: int = 3


class ConvAutoencoder(nn.Module):
    """Small convolutional autoencoder for CarRacing frames."""

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        if config.image_size != 64:
            raise ValueError("ConvAutoencoder currently supports image_size=64")
        self.config = config

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(config.channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.encoder_fc = nn.Linear(256 * 4 * 4, config.latent_dim)

        self.decoder_fc = nn.Linear(config.latent_dim, 256 * 4 * 4)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, config.channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder_conv(x)
        return self.encoder_fc(features.flatten(start_dim=1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        features = self.decoder_fc(z).view(z.shape[0], 256, 4, 4)
        return self.decoder_conv(features)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction, z
