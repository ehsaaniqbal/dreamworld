"""Modal jobs and dashboard for dreamworld.

Examples:

    modal run infra/modal_app.py::collect_rollouts_remote
    modal run infra/modal_app.py::train_vqvae_remote
    modal serve infra/modal_app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import modal

APP_NAME = "dreamworld"
VOLUME_NAME = "dreamworld-runs"
VOLUME_ROOT = Path("/dreamworld")
RUNS_ROOT = VOLUME_ROOT / "runs"
DATA_ROOT = VOLUME_ROOT / "datasets"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("swig")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_python_source("dreamworld")
    .add_local_dir("dreamworld/configs", remote_path="/root/dreamworld/configs")
    .add_local_dir("experiments", remote_path="/root/experiments")
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
    from dreamworld.data.collect_rollouts import collect_rollouts
    from dreamworld.experiments.artifacts import (
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
    from dreamworld.eval.eval_dynamics import eval_dynamics
    from dreamworld.experiments.artifacts import (
        append_log,
        init_run,
        mark_failed,
        update_run,
        write_metrics,
    )
    from dreamworld.train.train_dynamics import train_dynamics

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
            config_path=Path("dreamworld/configs/dynamics_transformer.yaml"),
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
    from dreamworld.eval.eval_reconstruction import eval_reconstruction
    from dreamworld.experiments.artifacts import (
        append_log,
        init_run,
        mark_failed,
        update_run,
        write_metrics,
    )
    from dreamworld.train.train_vqvae import train_vqvae

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
            config_path=Path("dreamworld/configs/tokenizer_vqvae.yaml"),
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
    gpu="A10G",
    timeout=4 * 60 * 60,
)
def eval_planner_remote(
    *,
    name: str = "planner-debug",
    dynamics_path: str,
    tokenizer_path: str | None = None,
    planner: str = "cem",
    episodes: int = 1,
    max_steps: int = 160,
    horizon: int = 8,
    candidates: int = 96,
    elites: int = 16,
    iterations: int = 2,
    seed: int = 0,
) -> dict[str, Any]:
    """Evaluate the dream planner in real CarRacing from Modal."""
    from dreamworld.eval.eval_planner import eval_planner
    from dreamworld.experiments.artifacts import (
        append_log,
        init_run,
        mark_failed,
        update_run,
        write_metrics,
    )

    config = {
        "dynamics_path": dynamics_path,
        "tokenizer_path": tokenizer_path,
        "planner": planner,
        "episodes": episodes,
        "max_steps": max_steps,
        "horizon": horizon,
        "candidates": candidates,
        "elites": elites,
        "iterations": iterations,
        "seed": seed,
    }
    run_path = init_run(RUNS_ROOT, name=name, kind="eval_planner", config=config, gpu="A10G")
    try:
        update_run(run_path, status="running", phase="planning")
        append_log(run_path, f"evaluating planner with dynamics {dynamics_path}")
        metrics = eval_planner(
            env_id="CarRacing-v3",
            tokenizer_path=tokenizer_path,
            dynamics_path=dynamics_path,
            planner=planner,
            episodes=episodes,
            max_steps=max_steps,
            horizon=horizon,
            candidates=candidates,
            elites=elites,
            iterations=iterations,
            done_penalty=10.0,
            brake_penalty=1.5,
            steering_penalty=0.05,
            low_gas_penalty=0.05,
            smoothness_penalty=0.1,
            out=run_path / "planner",
            seed=seed,
            device_name="cuda",
        )
        write_metrics(run_path, metrics)
        update_run(run_path, status="succeeded", phase="complete")
        volume.commit()
        return {"run_id": run_path.name, **metrics}
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
def failure_analysis_remote(
    *,
    name: str = "failure-debug",
    data_path: str,
    dynamics_path: str,
    tokenizer_path: str | None = None,
    planner_metrics_path: str | None = None,
    horizon: int = 20,
    top_k: int = 5,
    stride: int = 5,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Mine dream-model failures from Modal."""
    from dreamworld.eval.failure_analysis import mine_failures
    from dreamworld.experiments.artifacts import (
        append_log,
        init_run,
        mark_failed,
        update_run,
        write_metrics,
    )

    config = {
        "data_path": data_path,
        "dynamics_path": dynamics_path,
        "tokenizer_path": tokenizer_path,
        "planner_metrics_path": planner_metrics_path,
        "horizon": horizon,
        "top_k": top_k,
        "stride": stride,
        "batch_size": batch_size,
    }
    run_path = init_run(RUNS_ROOT, name=name, kind="failure_analysis", config=config, gpu="A10G")
    try:
        update_run(run_path, status="running", phase="mining failures")
        append_log(run_path, f"mining failures with dynamics {dynamics_path}")
        metrics = mine_failures(
            data_path=data_path,
            dynamics_path=dynamics_path,
            tokenizer_path=tokenizer_path,
            out=run_path / "failures",
            horizon=horizon,
            top_k=top_k,
            stride=stride,
            batch_size=batch_size,
            planner_metrics_path=planner_metrics_path,
            device_name="cuda",
        )
        write_metrics(run_path, metrics)
        update_run(run_path, status="succeeded", phase="complete")
        volume.commit()
        return {"run_id": run_path.name, **metrics}
    except BaseException as exc:
        mark_failed(run_path, exc)
        volume.commit()
        raise


@app.function(
    image=image,
    volumes={str(VOLUME_ROOT): volume},
    gpu="A10G",
    timeout=12 * 60 * 60,
)
def run_pipeline_remote(
    *,
    spec_name: str = "baseline_debug",
) -> dict[str, Any]:
    """Run a full tracked experiment from an experiment spec."""
    import yaml

    from dreamworld.data.collect_rollouts import collect_rollouts
    from dreamworld.eval.compare_real_vs_dream import compare_real_vs_dream
    from dreamworld.eval.eval_dynamics import eval_dynamics
    from dreamworld.eval.eval_planner import eval_planner
    from dreamworld.eval.eval_reconstruction import eval_reconstruction
    from dreamworld.eval.failure_analysis import mine_failures
    from dreamworld.experiments.artifacts import (
        append_log,
        init_run,
        mark_failed,
        update_run,
        write_metrics,
    )
    from dreamworld.train.train_dynamics import train_dynamics
    from dreamworld.train.train_vqvae import train_vqvae

    volume.reload()
    spec_path = _resolve_spec_path(spec_name)
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
    name = str(spec.get("name") or spec_path.stem.replace("_", "-"))
    run_path = init_run(RUNS_ROOT, name=name, kind="pipeline", config=spec, gpu="A10G")
    dataset_path = DATA_ROOT / run_path.name / "rollouts.npz"
    metrics: dict[str, Any] = {"spec_path": str(spec_path)}

    try:
        update_run(run_path, status="running", phase="collect")
        collect_cfg = dict(spec.get("collect") or {})
        append_log(run_path, "collecting rollouts")
        metrics["collect"] = collect_rollouts(
            out=dataset_path.parent,
            episodes=int(collect_cfg.get("episodes", 2)),
            policy=str(collect_cfg.get("policy", "hybrid")),
            image_size=int(collect_cfg.get("image_size", 64)),
            max_steps=int(collect_cfg.get("max_steps", 200)),
            seed=int(collect_cfg.get("seed", 0)),
        )
        update_run(run_path, dataset_path=str(dataset_path))
        volume.commit()

        update_run(run_path, phase="train_vqvae")
        vq_cfg = dict(spec.get("vqvae") or {})
        append_log(run_path, "training VQ-VAE")
        vqvae_path = run_path / "vqvae"
        metrics["vqvae"] = train_vqvae(
            config_path=Path("dreamworld/configs/tokenizer_vqvae.yaml"),
            data_path=dataset_path,
            out=vqvae_path,
            epochs=int(vq_cfg.get("epochs", 8)),
            batch_size=int(vq_cfg.get("batch_size", 128)),
            embedding_dim=int(vq_cfg.get("embedding_dim", 32)),
            num_embeddings=int(vq_cfg.get("num_embeddings", 128)),
            max_frames=vq_cfg.get("max_frames"),
            device_name="cuda",
        )
        tokenizer_path = vqvae_path / "checkpoints" / "best_codebook.pt"
        metrics["vqvae_eval"] = eval_reconstruction(
            checkpoint_path=tokenizer_path,
            data_path=dataset_path,
            out=run_path / "vqvae_eval",
            device_name="cuda",
        )
        volume.commit()

        update_run(run_path, phase="train_dynamics")
        dynamics_cfg = dict(spec.get("dynamics") or {})
        append_log(run_path, "training dynamics")
        dynamics_path = run_path / "dynamics"
        metrics["dynamics"] = train_dynamics(
            config_path=Path("dreamworld/configs/dynamics_transformer.yaml"),
            data_path=dataset_path,
            tokenizer_path=tokenizer_path,
            out=dynamics_path,
            epochs=int(dynamics_cfg.get("epochs", 8)),
            batch_size=int(dynamics_cfg.get("batch_size", 64)),
            learning_rate=_optional_float(dynamics_cfg.get("learning_rate")),
            context_length=int(dynamics_cfg.get("context_length", 16)),
            d_model=int(dynamics_cfg.get("d_model", 256)),
            n_heads=int(dynamics_cfg.get("n_heads", 4)),
            n_layers=int(dynamics_cfg.get("n_layers", 4)),
            dropout=_optional_float(dynamics_cfg.get("dropout")),
            reward_loss_weight=_optional_float(dynamics_cfg.get("reward_loss_weight")),
            done_loss_weight=_optional_float(dynamics_cfg.get("done_loss_weight")),
            max_frames=dynamics_cfg.get("max_frames"),
            device_name="cuda",
        )
        dynamics_checkpoint = dynamics_path / "checkpoints" / "best.pt"
        metrics["dynamics_eval"] = eval_dynamics(
            checkpoint_path=dynamics_checkpoint,
            tokenizer_path=None,
            data_path=dataset_path,
            out=run_path / "dynamics_eval",
            batch_size=int(dynamics_cfg.get("batch_size", 64)),
            device_name="cuda",
        )
        volume.commit()

        update_run(run_path, phase="dream_vs_real")
        dream_cfg = dict(spec.get("dream") or {})
        metrics["dream"] = compare_real_vs_dream(
            data_path=dataset_path,
            tokenizer_path=None,
            dynamics_path=dynamics_checkpoint,
            out=run_path / "dream_vs_real",
            horizon=int(dream_cfg.get("horizon", 30)),
            batch_size=int(dream_cfg.get("batch_size", 64)),
            device_name="cuda",
        )
        volume.commit()

        update_run(run_path, phase="planner")
        planner_cfg = dict(spec.get("planner") or {})
        metrics["planner"] = eval_planner(
            env_id="CarRacing-v3",
            tokenizer_path=None,
            dynamics_path=dynamics_checkpoint,
            planner=str(planner_cfg.get("planner", "cem")),
            episodes=int(planner_cfg.get("episodes", 1)),
            max_steps=int(planner_cfg.get("max_steps", 160)),
            horizon=int(planner_cfg.get("horizon", 8)),
            candidates=int(planner_cfg.get("candidates", 96)),
            elites=int(planner_cfg.get("elites", 16)),
            iterations=int(planner_cfg.get("iterations", 2)),
            done_penalty=float(planner_cfg.get("done_penalty", 10.0)),
            brake_penalty=float(planner_cfg.get("brake_penalty", 1.5)),
            steering_penalty=float(planner_cfg.get("steering_penalty", 0.05)),
            low_gas_penalty=float(planner_cfg.get("low_gas_penalty", 0.05)),
            smoothness_penalty=float(planner_cfg.get("smoothness_penalty", 0.1)),
            out=run_path / "planner",
            seed=int(planner_cfg.get("seed", 0)),
            device_name="cuda",
        )
        volume.commit()

        update_run(run_path, phase="failure_analysis")
        failure_cfg = dict(spec.get("failure") or {})
        metrics["failure"] = mine_failures(
            data_path=dataset_path,
            dynamics_path=dynamics_checkpoint,
            tokenizer_path=None,
            out=run_path / "failure_analysis",
            horizon=int(failure_cfg.get("horizon", 20)),
            top_k=int(failure_cfg.get("top_k", 5)),
            stride=int(failure_cfg.get("stride", 5)),
            batch_size=int(failure_cfg.get("batch_size", 64)),
            planner_metrics_path=run_path / "planner" / "metrics.json",
            device_name="cuda",
        )

        write_metrics(run_path, metrics)
        update_run(
            run_path,
            status="succeeded",
            phase="complete",
            dataset_path=str(dataset_path),
            tokenizer_path=str(tokenizer_path),
            dynamics_path=str(dynamics_checkpoint),
        )
        volume.commit()
        return {"run_id": run_path.name, **metrics}
    except BaseException as exc:
        write_metrics(run_path, metrics)
        mark_failed(run_path, exc)
        volume.commit()
        raise


def _resolve_spec_path(spec_name: str) -> Path:
    raw = Path(spec_name)
    candidates = [
        raw,
        Path("experiments/specs") / spec_name,
        Path("experiments/specs") / f"{spec_name}.yaml",
        Path("experiments/specs") / f"{spec_name}.yml",
        VOLUME_ROOT / "research" / "queue" / "pending" / spec_name,
        VOLUME_ROOT / "research" / "queue" / "pending" / f"{spec_name}.yaml",
        VOLUME_ROOT / "research" / "queue" / "pending" / f"{spec_name}.yml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Experiment spec not found: {spec_name}")


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


@app.function(
    image=image,
    volumes={str(VOLUME_ROOT): volume},
    timeout=10 * 60,
)
def propose_next_experiment_remote(
    *,
    base_spec_name: str = "baseline_debug",
    name: str | None = None,
) -> dict[str, Any]:
    """Write the next config-only experiment spec into the Modal Volume."""
    from dreamworld.research.loop import propose_next_spec

    result = propose_next_spec(
        research_root=VOLUME_ROOT,
        runs_root=RUNS_ROOT,
        base_spec_path=_resolve_spec_path(base_spec_name),
        name=name,
    )
    volume.commit()
    return result


@app.function(
    image=image,
    volumes={str(VOLUME_ROOT): volume},
    timeout=10 * 60,
)
def evaluate_completed_run_remote(
    *,
    run_id: str,
) -> dict[str, Any]:
    """Evaluate a completed run and write its research note."""
    from dreamworld.research.loop import evaluate_completed_run

    volume.reload()
    result = evaluate_completed_run(
        research_root=VOLUME_ROOT,
        runs_root=RUNS_ROOT,
        run_id=run_id,
    )
    volume.commit()
    return result


@app.function(
    image=image,
    volumes={str(VOLUME_ROOT): volume},
    timeout=14 * 60 * 60,
)
def run_autoresearch_once_remote(
    *,
    base_spec_name: str = "baseline_debug",
    name: str | None = None,
) -> dict[str, Any]:
    """Plan, run, and evaluate one config-only autoresearch iteration."""
    from dreamworld.research.loop import evaluate_completed_run, propose_next_spec

    proposal = propose_next_spec(
        research_root=VOLUME_ROOT,
        runs_root=RUNS_ROOT,
        base_spec_path=_resolve_spec_path(base_spec_name),
        name=name,
    )
    volume.commit()
    run_result = run_pipeline_remote.remote(spec_name=proposal["spec_path"])
    volume.reload()
    decision = evaluate_completed_run(
        research_root=VOLUME_ROOT,
        runs_root=RUNS_ROOT,
        run_id=str(run_result["run_id"]),
    )
    volume.commit()
    return {"proposal": proposal, "run": run_result, "decision": decision}


@app.function(
    image=image,
    volumes={str(VOLUME_ROOT): volume},
    timeout=24 * 60 * 60,
)
def run_autoresearch_loop_remote(
    *,
    base_spec_name: str = "baseline_debug",
    max_iterations: int = 4,
) -> dict[str, Any]:
    """Run config-only autoresearch until the stop rule or iteration budget fires."""
    from dreamworld.experiments.artifacts import load_json, write_json
    from dreamworld.research.loop import (
        evaluate_completed_run,
        propose_next_spec,
        research_paths,
        should_stop,
    )

    paths = research_paths(VOLUME_ROOT, runs_root=RUNS_ROOT)
    state = load_json(paths.state_path, default={})
    state["status"] = "running"
    state["max_iterations"] = max_iterations
    write_json(paths.state_path, state)
    volume.commit()

    completed: list[dict[str, Any]] = []
    for _index in range(max_iterations):
        state = load_json(paths.state_path, default={})
        if should_stop(state):
            break
        proposal = propose_next_spec(
            research_root=VOLUME_ROOT,
            runs_root=RUNS_ROOT,
            base_spec_path=_resolve_spec_path(base_spec_name),
        )
        volume.commit()
        run_result = run_pipeline_remote.remote(spec_name=proposal["spec_path"])
        volume.reload()
        decision = evaluate_completed_run(
            research_root=VOLUME_ROOT,
            runs_root=RUNS_ROOT,
            run_id=str(run_result["run_id"]),
        )
        volume.commit()
        completed.append(
            {
                "proposal": proposal["spec_path"],
                "run_id": run_result["run_id"],
                "decision": decision["decision"]["decision"],
            }
        )

    state = load_json(paths.state_path, default={})
    state["status"] = "stopped" if should_stop(state) else "ready"
    write_json(paths.state_path, state)
    volume.commit()
    return {"completed": completed, "state": state}


@app.function(
    image=image,
    volumes={str(VOLUME_ROOT): volume},
    timeout=60 * 60,
)
@modal.asgi_app()
def dashboard():
    """Serve the experiment dashboard from the shared Modal Volume."""
    from dreamworld.experiments.dashboard import create_app

    return create_app(RUNS_ROOT)
