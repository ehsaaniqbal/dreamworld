"""Minimal web dashboard for dreamtrack experiment runs."""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse

from dreamtrack.experiments.artifacts import (
    discover_runs,
    flatten_metrics,
    metric_series,
)

STATUS_LABELS = {
    "queued": "Queued",
    "running": "Running",
    "succeeded": "Succeeded",
    "failed": "Failed",
    "cancelled": "Cancelled",
}


def create_app(runs_root: str | Path) -> FastAPI:
    root = Path(runs_root)
    app = FastAPI(title="dreamtrack research")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        runs = discover_runs(root)
        return page("Runs", render_index(runs, root))

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def run_detail(run_id: str) -> str:
        runs = {run["run_id"]: run for run in discover_runs(root)}
        run = runs.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return page(str(run.get("name") or run_id), render_run_detail(run, root))

    @app.get("/artifacts/{run_id}/{artifact_path:path}")
    def artifact(run_id: str, artifact_path: str) -> FileResponse:
        run_path = safe_child(root, run_id)
        path = safe_child(run_path, artifact_path)
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Artifact not found")
        return FileResponse(path)

    @app.get("/logs/{run_id}", response_class=PlainTextResponse)
    def logs(run_id: str) -> str:
        run_path = safe_child(root, run_id)
        path = run_path / "logs.txt"
        if not path.exists():
            return "No logs written for this run yet.\n"
        return path.read_text(encoding="utf-8", errors="replace")

    @app.get("/api/runs")
    def api_runs() -> list[dict[str, Any]]:
        return discover_runs(root)

    return app


def safe_child(root: Path, relative: str) -> Path:
    base = root.resolve()
    path = (base / relative).resolve()
    if base != path and base not in path.parents:
        raise HTTPException(status_code=400, detail="Invalid path")
    return path


def page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="30">
  <title>{escape(title)} &middot; dreamtrack</title>
  <style>{styles()}</style>
</head>
<body>
  <header>
    <div>
      <a class="brand" href="/">dreamtrack</a>
      <span class="muted">world-model research runs</span>
    </div>
    <span class="refresh">auto-refresh 30s</span>
  </header>
  <main>{body}</main>
</body>
</html>"""


def render_index(runs: list[dict[str, Any]], root: Path) -> str:
    if not runs:
        return f"""
<section class="hero">
  <h1>No runs yet</h1>
  <p>Point the dashboard at a run directory or launch a Modal job. Current root:
  <code>{escape(str(root))}</code></p>
</section>"""

    completed = sum(1 for run in runs if run.get("status") == "succeeded")
    running = sum(1 for run in runs if run.get("status") == "running")
    failed = sum(1 for run in runs if run.get("status") == "failed")
    best = next((run for run in sorted_runs_by_score(runs) if run.get("score") is not None), None)

    rows = "\n".join(render_run_row(run) for run in runs)
    best_label = format_score(best.get("score")) if best else "n/a"
    best_name = escape(str(best.get("name") or best.get("run_id"))) if best else "No scored run"
    return f"""
<section class="summary-grid">
  {stat_card("Total Runs", str(len(runs)), "tracked in this root")}
  {stat_card("Running", str(running), "active experiments")}
  {stat_card("Succeeded", str(completed), f"{failed} failed")}
  {stat_card("Best Score", best_label, best_name)}
</section>
<section>
  <div class="section-title">
    <h1>Experiment Runs</h1>
    <span>{escape(str(root))}</span>
  </div>
  <div class="table">
    <div class="thead">
      <span>Run</span><span>Status</span><span>Phase</span><span>Score</span><span>Updated</span>
    </div>
    {rows}
  </div>
</section>"""


def render_run_row(run: dict[str, Any]) -> str:
    run_id = str(run.get("run_id"))
    status = str(run.get("status") or "unknown")
    name = str(run.get("name") or run_id)
    kind = str(run.get("kind") or "run")
    phase = str(run.get("phase") or "n/a")
    updated = str(run.get("updated_at") or "n/a")
    return f"""
<a class="trow" href="/runs/{escape(run_id)}">
  <span>
    <strong>{escape(name)}</strong>
    <small>{escape(run_id)} &middot; {escape(kind)}</small>
  </span>
  <span>{status_badge(status)}</span>
  <span>{escape(phase)}</span>
  <span>{format_score(run.get("score"))}</span>
  <span>{escape(short_time(updated))}</span>
</a>"""


def render_run_detail(run: dict[str, Any], root: Path) -> str:
    run_id = str(run.get("run_id"))
    name = str(run.get("name") or run_id)
    metrics = run.get("metrics") if isinstance(run.get("metrics"), dict) else {}
    flat = flatten_metrics(metrics)
    series = metric_series(metrics)
    media = run.get("media") if isinstance(run.get("media"), list) else []
    config = run.get("config") if isinstance(run.get("config"), dict) else {}

    scalar_items = "\n".join(
        metric_item(key, value)
        for key, value in sorted(flat.items())
        if isinstance(value, int | float | str | bool) and not isinstance(value, list | dict)
    )
    charts = "\n".join(render_chart(name, values) for name, values in pick_series(series))
    gallery = "\n".join(render_media(run_id, item) for item in media[:18])
    config_html = escape(json.dumps(config, indent=2)) if config else "No config recorded."
    log_href = f"/logs/{escape(run_id)}"
    updated_card = stat_card(
        "Updated",
        escape(short_time(str(run.get("updated_at") or "n/a"))),
        "last metadata write",
    )
    return f"""
<section class="detail-head">
  <a href="/" class="back">&larr; Runs</a>
  <div>
    <h1>{escape(name)}</h1>
    <p>{escape(run_id)} &middot; {escape(str(run.get("kind") or "run"))}</p>
  </div>
  {status_badge(str(run.get("status") or "unknown"))}
</section>
<section class="summary-grid">
  {stat_card("Score", format_score(run.get("score")), "primary comparison value")}
  {stat_card("Phase", escape(str(run.get("phase") or "n/a")), "latest reported stage")}
  {stat_card("GPU", escape(format_gpu(run.get("gpu"))), "requested hardware")}
  {updated_card}
</section>
<section>
  <div class="section-title"><h2>Curves</h2><span>{len(series)} numeric series</span></div>
  <div class="chart-grid">{charts or empty_state("No numeric metric series found.")}</div>
</section>
<section>
  <div class="section-title"><h2>Metrics</h2><span>{len(flat)} scalar values</span></div>
  <div class="metrics-grid">{scalar_items or empty_state("No metrics found.")}</div>
</section>
<section>
  <div class="section-title"><h2>Artifacts</h2><span>{len(media)} media files</span></div>
  <div class="media-grid">{gallery or empty_state("No images or videos found.")}</div>
</section>
<section class="two-col">
  <div>
    <div class="section-title"><h2>Config</h2></div>
    <pre>{config_html}</pre>
  </div>
  <div>
    <div class="section-title"><h2>Logs</h2><a href="{log_href}">open raw</a></div>
    <iframe src="{log_href}" title="logs"></iframe>
  </div>
</section>
<footer>Run root: <code>{escape(str(root))}</code></footer>"""


def stat_card(label: str, value: str, hint: str) -> str:
    return f"""
<div class="stat">
  <small>{escape(label)}</small>
  <strong>{value}</strong>
  <span>{escape(hint)}</span>
</div>"""


def status_badge(status: str) -> str:
    label = STATUS_LABELS.get(status, status.title())
    return f'<span class="status {escape(status)}">{escape(label)}</span>'


def metric_item(key: str, value: Any) -> str:
    rendered = format_score(value) if isinstance(value, int | float) else escape(str(value))
    return f"""
<div class="metric">
  <small>{escape(key)}</small>
  <strong>{rendered}</strong>
</div>"""


def pick_series(series: dict[str, list[float]], limit: int = 8) -> list[tuple[str, list[float]]]:
    preferred = (
        "loss",
        "val_loss",
        "train_loss",
        "token_accuracy",
        "reward_mae",
        "codebook_perplexity",
        "episode_returns",
    )
    items = list(series.items())
    items.sort(key=lambda item: (not any(key in item[0] for key in preferred), item[0]))
    return items[:limit]


def sorted_runs_by_score(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        runs,
        key=lambda run: (
            run.get("score") is None,
            -(float(run.get("score")) if isinstance(run.get("score"), int | float) else 0.0),
        ),
    )


def render_chart(name: str, values: list[float]) -> str:
    width = 420
    height = 180
    pad = 20
    if not values:
        return ""
    min_value = min(values)
    max_value = max(values)
    span = max(max_value - min_value, 1e-9)
    points = []
    for index, value in enumerate(values):
        x = pad + (width - 2 * pad) * index / max(len(values) - 1, 1)
        y = height - pad - (height - 2 * pad) * (value - min_value) / span
        points.append(f"{x:.2f},{y:.2f}")
    last = values[-1]
    range_label = f"{format_score(min_value)} to {format_score(max_value)}"
    return f"""
<div class="chart">
  <div><strong>{escape(name)}</strong><span>{format_score(last)}</span></div>
  <svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(name)} chart">
    <path class="gridline" d="M {pad} {height - pad} H {width - pad}" />
    <path class="gridline" d="M {pad} {pad} H {width - pad}" />
    <polyline points="{' '.join(points)}" />
  </svg>
  <small>{range_label} &middot; {len(values)} points</small>
</div>"""


def render_media(run_id: str, item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    path = str(item.get("path") or "")
    name = str(item.get("name") or path)
    media_type = str(item.get("type") or "image")
    src = f"/artifacts/{escape(run_id)}/{escape(path)}"
    if media_type == "video":
        media = f'<video src="{src}" controls muted playsinline></video>'
    else:
        media = f'<img src="{src}" alt="{escape(name)}">'
    return f"""
<figure>
  {media}
  <figcaption>{escape(name)}</figcaption>
</figure>"""


def empty_state(message: str) -> str:
    return f'<div class="empty">{escape(message)}</div>'


def format_score(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int | float):
        if abs(float(value)) >= 100:
            return f"{float(value):.1f}"
        if abs(float(value)) >= 10:
            return f"{float(value):.2f}"
        return f"{float(value):.4f}"
    return escape(str(value))


def format_gpu(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def short_time(value: str) -> str:
    if not value or value == "n/a":
        return "n/a"
    return value.replace("T", " ").replace("+00:00", " UTC")


def styles() -> str:
    return """
:root {
  color-scheme: light;
  --bg: #f7f8fa;
  --panel: #ffffff;
  --ink: #20242a;
  --muted: #65717f;
  --line: #dde3ea;
  --blue: #2864d9;
  --green: #13845a;
  --red: #bd2d3a;
  --amber: #a3630a;
  --shadow: 0 18px 45px rgba(31, 41, 55, 0.08);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 14px/1.5 Inter, ui-sans-serif, system-ui, -apple-system,
    BlinkMacSystemFont, "Segoe UI", sans-serif;
}
header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 20px;
  padding: 18px clamp(18px, 4vw, 56px);
  border-bottom: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.86);
  backdrop-filter: blur(16px);
  position: sticky;
  top: 0;
  z-index: 10;
}
main {
  width: min(1240px, calc(100vw - 32px));
  margin: 28px auto 56px;
}
section { margin: 0 0 28px; }
h1, h2, p { margin: 0; }
h1 { font-size: clamp(26px, 4vw, 42px); line-height: 1.05; letter-spacing: 0; }
h2 { font-size: 19px; letter-spacing: 0; }
a { color: inherit; text-decoration: none; }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
pre {
  margin: 0;
  min-height: 260px;
  overflow: auto;
  background: #111827;
  color: #d9e2ef;
  border-radius: 8px;
  padding: 18px;
}
iframe {
  width: 100%;
  min-height: 260px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: white;
}
.brand { font-size: 18px; font-weight: 800; margin-right: 12px; }
.muted, .refresh, small, figcaption, footer { color: var(--muted); }
.hero {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 34px;
  box-shadow: var(--shadow);
}
.summary-grid, .chart-grid, .metrics-grid, .media-grid, .two-col {
  display: grid;
  gap: 14px;
}
.summary-grid { grid-template-columns: repeat(4, minmax(0, 1fr)); }
.chart-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.metrics-grid { grid-template-columns: repeat(4, minmax(0, 1fr)); }
.media-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.two-col { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); }
.stat, .metric, .chart, figure, .empty {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: var(--shadow);
}
.stat, .metric { padding: 16px; min-width: 0; }
.stat small, .metric small {
  display: block;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.stat strong, .metric strong {
  display: block;
  margin: 5px 0 2px;
  font-size: 25px;
  line-height: 1.1;
  overflow-wrap: anywhere;
}
.stat span { color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }
.section-title {
  display: flex;
  align-items: end;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 12px;
}
.section-title span, .section-title a { color: var(--muted); font-size: 13px; }
.table {
  overflow: hidden;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
  box-shadow: var(--shadow);
}
.thead, .trow {
  display: grid;
  grid-template-columns: minmax(260px, 2fr) 130px 150px 110px 190px;
  gap: 14px;
  align-items: center;
  padding: 13px 16px;
}
.thead {
  color: var(--muted);
  font-size: 12px;
  border-bottom: 1px solid var(--line);
  background: #f1f4f8;
  text-transform: uppercase;
}
.trow { border-bottom: 1px solid var(--line); }
.trow:last-child { border-bottom: 0; }
.trow:hover { background: #f8fbff; }
.trow small { display: block; margin-top: 2px; }
.status {
  display: inline-flex;
  align-items: center;
  min-height: 26px;
  padding: 2px 10px;
  border-radius: 999px;
  border: 1px solid var(--line);
  color: var(--muted);
  background: #f6f7f9;
  font-size: 12px;
  font-weight: 700;
}
.status.running { color: var(--blue); background: #edf4ff; border-color: #bcd3ff; }
.status.succeeded { color: var(--green); background: #ecf8f2; border-color: #b9e4ce; }
.status.failed { color: var(--red); background: #fff0f1; border-color: #f4bbc2; }
.status.queued { color: var(--amber); background: #fff6e8; border-color: #f0d3a6; }
.detail-head {
  display: flex;
  align-items: center;
  gap: 18px;
}
.detail-head div { flex: 1; }
.detail-head p { color: var(--muted); margin-top: 6px; }
.back {
  display: inline-flex;
  align-items: center;
  min-height: 34px;
  padding: 0 12px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
}
.chart { padding: 14px; }
.chart > div {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 8px;
}
.chart svg { width: 100%; height: 160px; display: block; }
.chart polyline {
  fill: none;
  stroke: var(--blue);
  stroke-width: 3;
  stroke-linejoin: round;
  stroke-linecap: round;
}
.gridline { stroke: #e3e8ef; stroke-width: 1; }
figure { margin: 0; overflow: hidden; }
figure img, figure video {
  display: block;
  width: 100%;
  aspect-ratio: 16 / 10;
  object-fit: contain;
  background: #111827;
}
figcaption { padding: 10px 12px; font-size: 12px; }
.empty { padding: 24px; color: var(--muted); grid-column: 1 / -1; }
footer { margin-top: 32px; }
@media (max-width: 920px) {
  .summary-grid, .chart-grid, .metrics-grid, .media-grid, .two-col {
    grid-template-columns: 1fr;
  }
  .thead { display: none; }
  .trow {
    grid-template-columns: 1fr;
    gap: 8px;
  }
  header { align-items: flex-start; flex-direction: column; }
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs/research"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    return parser.parse_args()


def main() -> None:
    import uvicorn

    args = parse_args()
    uvicorn.run(create_app(args.runs_root), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
