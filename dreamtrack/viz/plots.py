"""Plotting utilities for training metrics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def save_loss_curve(losses: list[float], path: str | Path, *, title: str = "Training Loss") -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(losses) + 1), losses, marker="o", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
