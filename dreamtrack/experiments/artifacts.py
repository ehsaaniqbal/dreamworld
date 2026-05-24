"""File-backed experiment run metadata and artifact helpers."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RUN_FILE = "run.json"
METRICS_FILE = "metrics.json"
LOG_FILE = "logs.txt"


def now_iso() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat()


def default_runs_root() -> Path:
    return Path(os.environ.get("DREAMTRACK_RUNS_ROOT", "runs/research"))


def safe_slug(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in value)
    parts = [part for part in cleaned.split("-") if part]
    return "-".join(parts)[:80] or "run"


def git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def make_run_id(name: str) -> str:
    stamp = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{safe_slug(name)}"


def run_dir(root: str | Path, run_id: str) -> Path:
    return Path(root) / run_id


def init_run(
    root: str | Path,
    *,
    name: str,
    kind: str,
    config: dict[str, Any] | None = None,
    run_id: str | None = None,
    status: str = "queued",
    gpu: str | list[str] | None = None,
) -> Path:
    output_dir = run_dir(root, run_id or make_run_id(name))
    output_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "run_id": output_dir.name,
        "name": name,
        "kind": kind,
        "status": status,
        "phase": "created",
        "gpu": gpu,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "started_at": None,
        "finished_at": None,
        "git_sha": git_sha(),
        "config": config or {},
        "metrics_path": METRICS_FILE,
        "log_path": LOG_FILE,
    }
    write_json(output_dir / RUN_FILE, record)
    append_log(output_dir, f"created {kind} run '{name}'")
    return output_dir


def load_json(path: str | Path, default: Any = None) -> Any:
    json_path = Path(path)
    if not json_path.exists():
        return default
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, data: Any) -> None:
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = json_path.with_suffix(json_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
    tmp_path.replace(json_path)


def update_run(output_dir: str | Path, **updates: Any) -> dict[str, Any]:
    path = Path(output_dir)
    record = load_json(path / RUN_FILE, default={})
    record.update(updates)
    record["updated_at"] = now_iso()
    if updates.get("status") == "running" and not record.get("started_at"):
        record["started_at"] = now_iso()
    if updates.get("status") in {"succeeded", "failed", "cancelled"}:
        record["finished_at"] = now_iso()
    write_json(path / RUN_FILE, record)
    return record


def write_metrics(output_dir: str | Path, metrics: dict[str, Any]) -> None:
    write_json(Path(output_dir) / METRICS_FILE, metrics)


def append_log(output_dir: str | Path, message: str) -> None:
    path = Path(output_dir) / LOG_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now_iso()}] {message}\n")


def mark_failed(output_dir: str | Path, exc: BaseException) -> None:
    append_log(output_dir, f"failed: {type(exc).__name__}: {exc}")
    update_run(output_dir, status="failed", phase="failed", error=f"{type(exc).__name__}: {exc}")


def discover_runs(root: str | Path) -> list[dict[str, Any]]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    runs: list[dict[str, Any]] = []
    for child in sorted(root_path.iterdir()):
        if not child.is_dir():
            continue
        run_record = load_json(child / RUN_FILE, default=None)
        metrics = load_json(child / METRICS_FILE, default=None)
        if run_record is None and metrics is None:
            metrics_files = list(child.rglob(METRICS_FILE))
            if not metrics_files:
                continue
            run_record = inferred_run_record(child, metrics_files[0])
            metrics = load_json(metrics_files[0], default={})
        elif run_record is None:
            run_record = inferred_run_record(child, child / METRICS_FILE)
        run_record = dict(run_record)
        run_record["path"] = str(child)
        run_record["metrics"] = metrics or load_json(child / METRICS_FILE, default={}) or {}
        run_record["score"] = score_run(run_record["metrics"])
        run_record["media"] = list_media(child)
        runs.append(run_record)
    return sorted(runs, key=lambda item: item.get("updated_at") or item.get("run_id"), reverse=True)


def inferred_run_record(path: Path, metrics_path: Path) -> dict[str, Any]:
    return {
        "run_id": path.name,
        "name": path.name.replace("_", " ").replace("-", " "),
        "kind": "legacy",
        "status": "succeeded",
        "phase": "complete",
        "gpu": None,
        "created_at": None,
        "updated_at": _mtime_iso(metrics_path),
        "started_at": None,
        "finished_at": _mtime_iso(metrics_path),
        "git_sha": None,
        "config": {},
        "metrics_path": str(metrics_path.relative_to(path)),
        "log_path": LOG_FILE,
    }


def _mtime_iso(path: Path) -> str | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).replace(microsecond=0).isoformat()


def score_run(metrics: dict[str, Any]) -> float | None:
    planner = _nested(metrics, ("results", "dream_cem", "mean_return"))
    if isinstance(planner, int | float):
        return float(planner)
    for key in (
        "mean_return",
        "token_accuracy",
        "best_val_loss",
        "mse_mean",
        "mean_dream_token_accuracy",
    ):
        value = metrics.get(key)
        if isinstance(value, int | float):
            return float(value)
    return None


def list_media(path: Path) -> list[dict[str, str]]:
    media: list[dict[str, str]] = []
    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file():
            continue
        suffix = candidate.suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4"}:
            continue
        relative = candidate.relative_to(path).as_posix()
        media.append(
            {
                "path": relative,
                "name": candidate.stem.replace("_", " ").replace("-", " "),
                "type": "video" if suffix == ".mp4" else "image",
            }
        )
    return media


def flatten_metrics(metrics: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in metrics.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_metrics(value, name))
        elif isinstance(value, list):
            if value and all(isinstance(item, int | float) for item in value):
                flat[name] = value[-1]
            else:
                flat[name] = f"{len(value)} items"
        else:
            flat[name] = value
    return flat


def metric_series(metrics: dict[str, Any], prefix: str = "") -> dict[str, list[float]]:
    series: dict[str, list[float]] = {}
    for key, value in metrics.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            series.update(metric_series(value, name))
        elif (
            isinstance(value, list)
            and len(value) >= 2
            and all(isinstance(item, int | float) for item in value)
        ):
            series[name] = [float(item) for item in value]
    return series


def _nested(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current
