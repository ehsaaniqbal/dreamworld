"""Small config-only autoresearch loop for world-model experiments."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from dreamworld.experiments.artifacts import (
    discover_runs,
    now_iso,
    safe_slug,
    write_json,
)

STATE_FILE = "state.json"
GOAL_FILE = "goal.md"


@dataclass(frozen=True)
class ResearchPaths:
    root: Path
    runs_root: Path
    research_root: Path
    queue_root: Path
    pending_root: Path
    running_root: Path
    done_root: Path
    failed_root: Path
    notes_root: Path
    state_path: Path
    goal_path: Path
    active_call_path: Path


def research_paths(root: str | Path, runs_root: str | Path | None = None) -> ResearchPaths:
    base = Path(root)
    research_root = base / "research"
    queue_root = research_root / "queue"
    return ResearchPaths(
        root=base,
        runs_root=Path(runs_root) if runs_root is not None else base / "runs",
        research_root=research_root,
        queue_root=queue_root,
        pending_root=queue_root / "pending",
        running_root=queue_root / "running",
        done_root=queue_root / "done",
        failed_root=queue_root / "failed",
        notes_root=research_root / "notes",
        state_path=research_root / STATE_FILE,
        goal_path=research_root / GOAL_FILE,
        active_call_path=research_root / "active_call.json",
    )


def ensure_research_dirs(paths: ResearchPaths) -> None:
    for path in (
        paths.research_root,
        paths.pending_root,
        paths.running_root,
        paths.done_root,
        paths.failed_root,
        paths.notes_root,
    ):
        path.mkdir(parents=True, exist_ok=True)
    if not paths.goal_path.exists():
        paths.goal_path.write_text(default_goal(), encoding="utf-8")
    if not paths.state_path.exists():
        write_json(paths.state_path, default_state())


def default_goal() -> str:
    return """# dreamworld autoresearch goal

Run config-only world-model experiments.

Stop when planner mean return improves by 25 percent over the current best, or when the
experiment budget is exhausted.

Allowed changes:
- rollout count, policy mix, and max steps
- VQ-VAE size and training budget
- dynamics context, width, depth, learning rate, and reward/done loss weights
- planner horizon, candidates, elites, iterations, and penalties

Do not edit Python model code in this loop.
"""


def default_state() -> dict[str, Any]:
    return {
        "status": "ready",
        "requested_status": "running",
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "iterations": 0,
        "max_iterations": 8,
        "target_improvement": 0.25,
        "min_confirm_episodes": 3,
        "best_run_id": None,
        "best_score": None,
        "candidate_best": None,
        "last_proposal": None,
        "last_decision": None,
    }


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return data


def write_yaml(path: str | Path, data: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def summarize_runs(runs_root: str | Path) -> dict[str, Any]:
    runs = discover_runs(runs_root)
    scored = [run for run in runs if isinstance(run.get("score"), int | float)]
    best = max(scored, key=lambda run: float(run["score"]), default=None)
    succeeded = [run for run in runs if run.get("status") == "succeeded"]
    failed = [run for run in runs if run.get("status") == "failed"]
    return {
        "total": len(runs),
        "succeeded": len(succeeded),
        "failed": len(failed),
        "best_run_id": best.get("run_id") if best else None,
        "best_name": best.get("name") if best else None,
        "best_score": float(best["score"]) if best else None,
        "recent": runs[:8],
    }


def propose_next_spec(
    *,
    research_root: str | Path,
    runs_root: str | Path,
    base_spec_path: str | Path,
    name: str | None = None,
) -> dict[str, Any]:
    paths = research_paths(research_root, runs_root=runs_root)
    ensure_research_dirs(paths)
    state = _load_state(paths)
    base_spec = load_yaml(base_spec_path)
    summary = summarize_runs(paths.runs_root)
    if "baseline_score" not in state and summary["best_score"] is not None:
        state["baseline_score"] = summary["best_score"]
        state["best_score"] = summary["best_score"]
        state["best_run_id"] = summary["best_run_id"]
    iteration = int(state.get("iterations") or 0) + 1
    proposal = mutate_spec(base_spec, iteration=iteration, summary=summary)
    proposal_name = name or proposal["name"]
    proposal["name"] = proposal_name
    proposal.setdefault("research", {})
    proposal["research"].update(
        {
            "iteration": iteration,
            "created_at": now_iso(),
            "base_spec": str(base_spec_path),
            "best_run_id": summary["best_run_id"],
            "best_score": summary["best_score"],
            "hypothesis": hypothesis_for(proposal, iteration),
        }
    )

    spec_path = paths.pending_root / f"{safe_slug(proposal_name)}.yaml"
    note_path = paths.notes_root / f"{safe_slug(proposal_name)}.md"
    write_yaml(spec_path, proposal)
    note_path.write_text(render_proposal_note(proposal, summary), encoding="utf-8")

    state.update(
        {
            "status": "proposed",
            "updated_at": now_iso(),
            "last_proposal": {
                "name": proposal_name,
                "spec_path": str(spec_path),
                "note_path": str(note_path),
                "iteration": iteration,
            },
        }
    )
    write_json(paths.state_path, state)
    return {
        "spec_path": str(spec_path),
        "note_path": str(note_path),
        "spec": proposal,
        "summary": _public_summary(summary),
    }


def mark_spec_running(research_root: str | Path, spec_path: str | Path) -> str:
    paths = research_paths(research_root)
    ensure_research_dirs(paths)
    source = Path(spec_path)
    if not source.exists():
        raise FileNotFoundError(f"Spec not found: {spec_path}")
    target = paths.running_root / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), target)
    return str(target)


def finalize_spec(
    *,
    research_root: str | Path,
    spec_name: str,
    decision: str,
) -> str | None:
    paths = research_paths(research_root)
    ensure_research_dirs(paths)
    source = find_queue_spec(paths, spec_name)
    if source is None:
        return None
    final_root = paths.failed_root if decision in {"crash", "inconclusive"} else paths.done_root
    target = final_root / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), target)
    return str(target)


def find_queue_spec(paths: ResearchPaths, spec_name: str) -> Path | None:
    slug = safe_slug(spec_name)
    candidates = [
        paths.running_root / f"{slug}.yaml",
        paths.pending_root / f"{slug}.yaml",
        paths.done_root / f"{slug}.yaml",
        paths.failed_root / f"{slug}.yaml",
    ]
    return next((path for path in candidates if path.exists()), None)


def mutate_spec(
    base_spec: dict[str, Any],
    *,
    iteration: int,
    summary: dict[str, Any],
) -> dict[str, Any]:
    spec = copy.deepcopy(base_spec)
    family = (iteration - 1) % 8
    best_score = summary.get("best_score")
    score_slug = "cold" if best_score is None else f"best-{float(best_score):.1f}"
    spec["name"] = f"auto-{iteration:03d}-{_family_name(family)}-{safe_slug(score_slug)}"

    collect = spec.setdefault("collect", {})
    vqvae = spec.setdefault("vqvae", {})
    dynamics = spec.setdefault("dynamics", {})
    planner = spec.setdefault("planner", {})
    dream = spec.setdefault("dream", {})
    failure = spec.setdefault("failure", {})

    if family == 0:
        collect["episodes"] = int(collect.get("episodes", 2)) + 2
        collect["max_steps"] = max(int(collect.get("max_steps", 120)), 160)
        vqvae["max_frames"] = None
        dynamics["max_frames"] = None
    elif family == 1:
        vqvae["num_embeddings"] = 128
        vqvae["embedding_dim"] = max(int(vqvae.get("embedding_dim", 32)), 48)
        dynamics["d_model"] = max(int(dynamics.get("d_model", 128)), 192)
    elif family == 2:
        dynamics["context_length"] = max(int(dynamics.get("context_length", 8)), 12)
        dynamics["n_layers"] = max(int(dynamics.get("n_layers", 2)), 3)
    elif family == 3:
        dynamics["reward_loss_weight"] = 2.0
        dynamics["done_loss_weight"] = 0.5
        planner["done_penalty"] = 15.0
    elif family == 4:
        planner["horizon"] = max(int(planner.get("horizon", 6)), 10)
        planner["candidates"] = max(int(planner.get("candidates", 48)), 96)
        planner["elites"] = max(int(planner.get("elites", 8)), 16)
    elif family == 5:
        planner["brake_penalty"] = 0.8
        planner["low_gas_penalty"] = 0.02
        planner["smoothness_penalty"] = 0.2
    elif family == 6:
        vqvae["epochs"] = int(vqvae.get("epochs", 2)) + 2
        dynamics["epochs"] = int(dynamics.get("epochs", 2)) + 2
        dynamics["learning_rate"] = 0.0002
    else:
        collect["policy"] = "hybrid"
        collect["episodes"] = max(int(collect.get("episodes", 2)), 6)
        dynamics["context_length"] = max(int(dynamics.get("context_length", 8)), 16)
        planner["horizon"] = max(int(planner.get("horizon", 6)), 12)

    dream["horizon"] = max(int(dream.get("horizon", 20)), int(planner.get("horizon", 8)) * 2)
    failure["horizon"] = max(int(failure.get("horizon", 15)), int(planner.get("horizon", 8)) * 2)
    planner.setdefault("planner", "cem")
    planner.setdefault("episodes", 1)
    planner.setdefault("max_steps", 100)
    return spec


def evaluate_completed_run(
    *,
    research_root: str | Path,
    runs_root: str | Path,
    run_id: str,
) -> dict[str, Any]:
    paths = research_paths(research_root, runs_root=runs_root)
    ensure_research_dirs(paths)
    state = _load_state(paths)
    runs = {str(run.get("run_id")): run for run in discover_runs(paths.runs_root)}
    run = runs.get(run_id)
    if run is None:
        raise ValueError(f"Run not found: {run_id}")

    score = run.get("score")
    old_best = state.get("best_score")
    planner_episodes = planner_episode_count(run)
    min_confirm_episodes = int(state.get("min_confirm_episodes") or 3)
    decision = decide_run(
        score,
        old_best,
        run.get("status"),
        planner_episodes=planner_episodes,
        min_confirm_episodes=min_confirm_episodes,
    )
    if decision == "keep":
        state["best_score"] = float(score)
        state["best_run_id"] = run_id
        state["candidate_best"] = None
    elif decision == "candidate_best":
        state["candidate_best"] = {
            "run_id": run_id,
            "score": float(score),
            "previous_best": old_best,
            "planner_episodes": planner_episodes,
            "min_confirm_episodes": min_confirm_episodes,
        }
    state["iterations"] = int(state.get("iterations") or 0) + 1
    state["status"] = "ready"
    state["updated_at"] = now_iso()
    state["last_decision"] = {
        "run_id": run_id,
        "score": score,
        "previous_best": old_best,
        "decision": decision,
        "reason": decision_reason(
            decision,
            score,
            old_best,
            run,
            planner_episodes=planner_episodes,
            min_confirm_episodes=min_confirm_episodes,
        ),
    }
    spec_target = finalize_spec(
        research_root=paths.root,
        spec_name=str(run.get("name") or run_id),
        decision=decision,
    )
    state["last_decision"]["spec_path"] = spec_target
    write_json(paths.state_path, state)

    note_path = Path(run["path"]) / "note.md"
    note_path.write_text(render_decision_note(state["last_decision"], run), encoding="utf-8")
    return {"decision": state["last_decision"], "note_path": str(note_path), "state": state}


def decide_run(
    score: Any,
    previous_best: Any,
    status: Any,
    *,
    planner_episodes: int | None = None,
    min_confirm_episodes: int = 3,
) -> str:
    if status != "succeeded":
        return "crash"
    if not isinstance(score, int | float):
        return "inconclusive"
    if not isinstance(previous_best, int | float):
        return "keep"
    if float(score) > float(previous_best):
        if planner_episodes is not None and planner_episodes < min_confirm_episodes:
            return "candidate_best"
        return "keep"
    return "discard"


def should_stop(state: dict[str, Any]) -> bool:
    requested = state.get("requested_status")
    if requested in {"paused", "stopped"}:
        return True
    if int(state.get("iterations") or 0) >= int(state.get("max_iterations") or 8):
        return True
    best = state.get("best_score")
    baseline = state.get("baseline_score")
    target = float(state.get("target_improvement") or 0.25)
    if isinstance(best, int | float) and isinstance(baseline, int | float):
        return float(best) >= float(baseline) * (1.0 + target)
    return False


def should_pause(state: dict[str, Any]) -> bool:
    return state.get("requested_status") == "paused"


def should_stop_requested(state: dict[str, Any]) -> bool:
    return state.get("requested_status") == "stopped"


def _load_state(paths: ResearchPaths) -> dict[str, Any]:
    ensure_research_dirs(paths)
    with paths.state_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _public_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "recent"}


def _family_name(family: int) -> str:
    return (
        "more-data",
        "larger-codebook",
        "longer-context",
        "reward-done",
        "planner-search",
        "action-prior",
        "train-longer",
        "larger-loop",
    )[family]


def hypothesis_for(spec: dict[str, Any], iteration: int) -> str:
    family = _family_name((iteration - 1) % 8).replace("-", " ")
    return f"{family} should improve planner return without making dream rollouts less stable"


def render_proposal_note(spec: dict[str, Any], summary: dict[str, Any]) -> str:
    research = spec.get("research", {})
    return f"""# {spec.get("name")}

Hypothesis: {research.get("hypothesis")}

Current best: {summary.get("best_run_id") or "none"} / {summary.get("best_score")}

Key config:

```yaml
{yaml.safe_dump(spec, sort_keys=False).strip()}
```
"""


def render_decision_note(decision: dict[str, Any], run: dict[str, Any]) -> str:
    return f"""# {run.get("name") or run.get("run_id")}

Decision: {decision["decision"]}

Score: {decision.get("score")}
Previous best: {decision.get("previous_best")}

Reason: {decision["reason"]}
"""


def planner_episode_count(run: dict[str, Any]) -> int | None:
    config = run.get("config")
    if not isinstance(config, dict):
        return None
    planner = config.get("planner")
    if not isinstance(planner, dict):
        return None
    episodes = planner.get("episodes")
    return int(episodes) if isinstance(episodes, int | float) else None


def decision_reason(
    decision: str,
    score: Any,
    old_best: Any,
    run: dict[str, Any],
    *,
    planner_episodes: int | None,
    min_confirm_episodes: int,
) -> str:
    if decision == "crash":
        return f"run ended with status {run.get('status')}"
    if decision == "inconclusive":
        return "run did not produce a comparable planner score"
    if decision == "keep" and old_best is None:
        return "first comparable run"
    if decision == "keep":
        return f"score improved from {old_best} to {score}"
    if decision == "candidate_best":
        return (
            f"score improved from {old_best} to {score}, but only "
            f"{planner_episodes} planner episodes ran; needs {min_confirm_episodes}"
        )
    return f"score {score} did not beat current best {old_best}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    propose = subparsers.add_parser("propose")
    propose.add_argument("--research-root", type=Path, default=Path("runs"))
    propose.add_argument("--runs-root", type=Path, default=Path("runs/research"))
    propose.add_argument(
        "--base-spec",
        type=Path,
        default=Path("experiments/specs/baseline_debug.yaml"),
    )
    propose.add_argument("--name", default=None)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--research-root", type=Path, default=Path("runs"))
    evaluate.add_argument("--runs-root", type=Path, default=Path("runs/research"))
    evaluate.add_argument("--run-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "propose":
        result = propose_next_spec(
            research_root=args.research_root,
            runs_root=args.runs_root,
            base_spec_path=args.base_spec,
            name=args.name,
        )
    else:
        result = evaluate_completed_run(
            research_root=args.research_root,
            runs_root=args.runs_root,
            run_id=args.run_id,
        )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
