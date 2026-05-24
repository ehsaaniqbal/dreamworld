"""Modal jobs and dashboard for dreamtrack.

Examples:

    modal run infra/modal_app.py::collect_rollouts_remote
    modal run infra/modal_app.py::train_vqvae_remote
    modal serve infra/modal_app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import modal

APP_NAME = "dreamtrack"
VOLUME_NAME = "dreamtrack-runs"
VOLUME_ROOT = Path("/dreamtrack")
RUNS_ROOT = VOLUME_ROOT / "runs"
DATA_ROOT = VOLUME_ROOT / "datasets"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("swig")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_python_source("dreamtrack")
    .add_local_dir("dreamtrack/configs", remote_path="/root/dreamtrack/configs")
)


@app.function(
    image=image,
    volumes={str(VOLUME_ROOT): volume},
    timeout=60 * 60,
)
def collect_rollouts_remote(
    *,
    name: str = "collect-debug",
    episodes: int = 2,
    policy: str = "hybrid",
    image_size: int = 64,
    max_steps: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    """Collect CarRacing rollouts into the shared Modal Volume."""
    from dreamtrack.data.collect_rollouts import collect_rollouts
    from dreamtrack.experiments.artifacts import (
        append_log,
        init_run,
        mark_failed,
        update_run,
        write_metrics,
    )

    config = {
        "episodes": episodes,
        "policy": policy,
        "image_size": image_size,
        "max_steps": max_steps,
        "seed": seed,
    }
    run_path = init_run(RUNS_ROOT, name=name, kind="collect", config=config)
    output_path = DATA_ROOT / run_path.name
    try:
        update_run(run_path, status="running", phase="collecting")
        append_log(run_path, f"writing dataset to {output_path}")
        metrics = collect_rollouts(
            out=output_path,
            episodes=episodes,
            policy=policy,
            image_size=image_size,
            max_steps=max_steps,
            seed=seed,
        )
        write_metrics(run_path, metrics)
        update_run(
            run_path,
            status="succeeded",
            phase="complete",
            artifact_path=str(output_path),
            dataset_path=str(output_path / "rollouts.npz"),
        )
        volume.commit()
        return {
            "run_id": run_path.name,
            "dataset_path": str(output_path / "rollouts.npz"),
            **metrics,
        }
    except BaseException as exc:
        mark_failed(run_path, exc)
        volume.commit()
        raise


@app.function(
    image=image,
    volumes={str(VOLUME_ROOT): volume},
    gpu="A10G",
    timeout=6 * 60 * 60,
)
def train_dynamics_remote(
    *,
    name: str = "dynamics-debug",
    data_path: str,
    tokenizer_path: str,
    epochs: int = 8,
    batch_size: int = 64,
    context_length: int = 16,
    d_model: int = 256,
    n_heads: int = 4,
    n_layers: int = 4,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Train token dynamics on a Modal GPU and write artifacts to the shared Volume."""
    from dreamtrack.eval.eval_dynamics import eval_dynamics
    from dreamtrack.experiments.artifacts import (
        append_log,
        init_run,
        mark_failed,
        update_run,
        write_metrics,
    )
    from dreamtrack.train.train_dynamics import train_dynamics

    config = {
        "data_path": data_path,
        "tokenizer_path": tokenizer_path,
        "epochs": epochs,
        "batch_size": batch_size,
        "context_length": context_length,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "max_frames": max_frames,
    }
    run_path = init_run(RUNS_ROOT, name=name, kind="train_dynamics", config=config, gpu="A10G")
    output_path = run_path / "dynamics"
    try:
        update_run(run_path, status="running", phase="training")
        append_log(run_path, f"training dynamics from {data_path}")
        metrics = train_dynamics(
            config_path=Path("dreamtrack/configs/dynamics_transformer.yaml"),
            data_path=Path(data_path),
            tokenizer_path=Path(tokenizer_path),
            out=output_path,
            epochs=epochs,
            batch_size=batch_size,
            context_length=context_length,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_frames=max_frames,
            device_name="cuda",
        )
        update_run(run_path, phase="evaluating")
        eval_metrics = eval_dynamics(
            checkpoint_path=output_path / "checkpoints" / "best.pt",
            tokenizer_path=None,
            data_path=Path(data_path),
            out=run_path / "eval",
            batch_size=batch_size,
            device_name="cuda",
        )
        combined = {"train": metrics, "eval": eval_metrics}
        write_metrics(run_path, combined)
        update_run(
            run_path,
            status="succeeded",
            phase="complete",
            checkpoint_path=str(output_path / "checkpoints" / "best.pt"),
        )
        volume.commit()
        return {"run_id": run_path.name, **combined}
    except BaseException as exc:
        mark_failed(run_path, exc)
        volume.commit()
        raise


@app.function(
    image=image,
    volumes={str(VOLUME_ROOT): volume},
    gpu="A10G",
    timeout=4 * 60 * 60,
)
def train_vqvae_remote(
    *,
    name: str = "vqvae-debug",
    data_path: str,
    epochs: int = 8,
    batch_size: int = 128,
    embedding_dim: int = 32,
    num_embeddings: int = 128,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Train a VQ-VAE on a Modal GPU and write artifacts to the shared Volume."""
    from dreamtrack.eval.eval_reconstruction import eval_reconstruction
    from dreamtrack.experiments.artifacts import (
        append_log,
        init_run,
        mark_failed,
        update_run,
        write_metrics,
    )
    from dreamtrack.train.train_vqvae import train_vqvae

    config = {
        "data_path": data_path,
        "epochs": epochs,
        "batch_size": batch_size,
        "embedding_dim": embedding_dim,
        "num_embeddings": num_embeddings,
        "max_frames": max_frames,
    }
    run_path = init_run(RUNS_ROOT, name=name, kind="train_vqvae", config=config, gpu="A10G")
    output_path = run_path / "vqvae"
    try:
        update_run(run_path, status="running", phase="training")
        append_log(run_path, f"training VQ-VAE from {data_path}")
        metrics = train_vqvae(
            config_path=Path("dreamtrack/configs/tokenizer_vqvae.yaml"),
            data_path=Path(data_path),
            out=output_path,
            epochs=epochs,
            batch_size=batch_size,
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            max_frames=max_frames,
            device_name="cuda",
        )
        update_run(run_path, phase="evaluating")
        eval_metrics = eval_reconstruction(
            checkpoint_path=output_path / "checkpoints" / "best_codebook.pt",
            data_path=Path(data_path),
            out=run_path / "eval_codebook",
            device_name="cuda",
        )
        combined = {"train": metrics, "eval": eval_metrics}
        write_metrics(run_path, combined)
        update_run(
            run_path,
            status="succeeded",
            phase="complete",
            checkpoint_path=str(output_path / "checkpoints" / "best_codebook.pt"),
        )
        volume.commit()
        return {"run_id": run_path.name, **combined}
    except BaseException as exc:
        mark_failed(run_path, exc)
        volume.commit()
        raise


@app.function(
    image=image,
    volumes={str(VOLUME_ROOT): volume},
    timeout=60 * 60,
)
@modal.asgi_app()
def dashboard():
    """Serve the experiment dashboard from the shared Modal Volume."""
    from dreamtrack.experiments.dashboard import create_app

    return create_app(RUNS_ROOT)
