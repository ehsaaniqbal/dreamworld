"""Build a static HTML demo page from dreamtrack run artifacts."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from html import escape
from pathlib import Path
from typing import Any


def build_dashboard(
    *,
    out: str | Path,
    ae_run: str | Path | None = None,
    vqvae_run: str | Path | None = None,
    dream_run: str | Path | None = None,
    planner_run: str | Path | None = None,
    failure_run: str | Path | None = None,
    title: str = "dreamtrack CarRacing World Model Demo",
) -> Path:
    """Create a self-contained static demo directory with copied artifacts."""
    output_dir = Path(out)
    asset_dir = output_dir / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    for stale_asset in asset_dir.iterdir():
        if stale_asset.is_file():
            stale_asset.unlink()

    sections: list[str] = []
    sections.append(_hero_section(title))

    if ae_run:
        sections.append(_image_section("Autoencoder Reconstruction", Path(ae_run), asset_dir))
    if vqvae_run:
        sections.append(_vqvae_section(Path(vqvae_run), asset_dir))
    if dream_run:
        sections.append(_dream_section(Path(dream_run), asset_dir))
    if planner_run:
        sections.append(_planner_section(Path(planner_run), asset_dir))
    if failure_run:
        sections.append(_failure_section(Path(failure_run), asset_dir))

    html = _page(title=title, body="\n".join(sections))
    index_path = output_dir / "index.html"
    index_path.write_text(html, encoding="utf-8")
    return index_path


def _hero_section(title: str) -> str:
    return f"""
    <section class="hero">
      <div>
        <p class="eyebrow">CarRacing-v3</p>
        <h1>{escape(title)}</h1>
      </div>
    </section>
    """


def _image_section(title: str, run_dir: Path, asset_dir: Path) -> str:
    image = _first_existing(
        [
            run_dir / "eval_latest" / "reconstruction_grid.png",
            run_dir / "eval" / "reconstruction_grid.png",
            run_dir / "recon_epoch_003.png",
            run_dir / "recon_epoch_002.png",
            run_dir / "recon_epoch_001.png",
        ]
    )
    metrics = _load_json(run_dir / "metrics.json")
    image_tag = _image_tag(image, asset_dir) if image else "<p>No reconstruction image found.</p>"
    return f"""
    <section>
      <h2>{escape(title)}</h2>
      {image_tag}
      {_metrics_table(metrics, ["best_val_loss", "frames", "epochs", "device"])}
    </section>
    """


def _vqvae_section(run_dir: Path, asset_dir: Path) -> str:
    recon = _first_existing(
        [
            run_dir / "eval_codebook" / "reconstruction_grid.png",
            run_dir / "recon_epoch_003.png",
            run_dir / "recon_epoch_002.png",
            run_dir / "recon_epoch_001.png",
        ]
    )
    tokens = _first_existing(
        [
            run_dir / "tokens_epoch_003.png",
            run_dir / "tokens_epoch_002.png",
            run_dir / "tokens_epoch_001.png",
        ]
    )
    metrics = _load_json(run_dir / "metrics.json")
    return f"""
    <section>
      <h2>VQ-VAE Visual Tokens</h2>
      <div class="grid two">
        <div>{_image_tag(recon, asset_dir) if recon else "<p>No reconstruction found.</p>"}</div>
        <div>{_image_tag(tokens, asset_dir) if tokens else "<p>No token grid found.</p>"}</div>
      </div>
      {_metrics_table(metrics, ["best_val_loss", "frames", "epochs", "best_codebook_checkpoint"])}
    </section>
    """


def _dream_section(run_dir: Path, asset_dir: Path) -> str:
    metrics = _load_json(run_dir / "metrics.json")
    teacher_video = _first_existing([run_dir / "teacher_forced_vs_real.mp4"])
    open_loop_video = _first_existing([run_dir / "open_loop_dream_vs_real.mp4"])
    failure_note = _failure_summary(metrics)
    return f"""
    <section>
      <h2>Real Future vs Imagined Future</h2>
      <div class="grid two">
        <div>
          <h3>Teacher Forced</h3>
          {_video_tag(teacher_video, asset_dir)}
        </div>
        <div>
          <h3>Open Loop Dream</h3>
          {_video_tag(open_loop_video, asset_dir)}
        </div>
      </div>
      {_metrics_table(metrics, [
          "horizon",
          "mean_teacher_pixel_mse",
          "mean_dream_pixel_mse",
          "mean_teacher_token_accuracy",
          "mean_dream_token_accuracy",
          "mean_dream_reward_mae",
      ])}
      <p class="failure">{escape(failure_note)}</p>
    </section>
    """


def _planner_section(run_dir: Path, asset_dir: Path) -> str:
    metrics = _load_json(run_dir / "metrics.json")
    reward_plot = _first_existing([run_dir / "reward_curve.png"])
    videos = sorted(run_dir.glob("*_episode_000.mp4"))
    candidate_videos = sorted(run_dir.glob("*candidate_plan_debug.mp4"))
    video_cards = "\n".join(
        f"<div><h3>{escape(video.stem)}</h3>{_video_tag(video, asset_dir)}</div>"
        for video in videos
    )
    candidate_cards = "\n".join(
        f"<div><h3>{escape(video.stem)}</h3>{_video_tag(video, asset_dir)}</div>"
        for video in candidate_videos
    )
    debug = _planner_debug(metrics)
    return f"""
    <section>
      <h2>Planner Evaluation</h2>
      {_image_tag(reward_plot, asset_dir) if reward_plot else "<p>No reward plot found.</p>"}
      <div class="grid three">{video_cards}</div>
      <h3>Candidate Imagined Plans</h3>
      <div class="grid two">{candidate_cards}</div>
      {debug}
    </section>
    """


def _failure_section(run_dir: Path, asset_dir: Path) -> str:
    metrics = _load_json(run_dir / "metrics.json")
    gap_plot = _first_existing([run_dir / "dream_real_gap.png"])
    failure_videos = sorted((run_dir / "failures").glob("failure_*.mp4"))[:6]
    video_cards = "\n".join(
        f"<div><h3>{escape(video.stem)}</h3>{_video_tag(video, asset_dir)}</div>"
        for video in failure_videos
    )
    summary_path = run_dir / "summary.md"
    summary = ""
    if summary_path.exists():
        summary = f"<pre>{escape(summary_path.read_text(encoding='utf-8'))}</pre>"
    return f"""
    <section>
      <h2>Failure Cases</h2>
      {_image_tag(gap_plot, asset_dir) if gap_plot else "<p>No failure gap plot found.</p>"}
      <div class="grid three">{video_cards}</div>
      {_metrics_table(metrics, [
          "scanned_windows",
          "horizon",
          "planner_exploitation_score",
      ])}
      {summary}
    </section>
    """


def _planner_debug(metrics: dict[str, Any]) -> str:
    results = metrics.get("results", {})
    if not isinstance(results, dict):
        return ""
    rows = []
    for name, result in results.items():
        if not isinstance(result, dict):
            continue
        rows.append(
            "<tr>"
            f"<td>{escape(name)}</td>"
            f"<td>{float(result.get('mean_return', 0.0)):.3f}</td>"
            f"<td>{float(result.get('mean_length', 0.0)):.1f}</td>"
            "</tr>"
        )
    return f"""
    <table>
      <thead><tr><th>Agent</th><th>Mean Return</th><th>Mean Length</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    """


def _failure_summary(metrics: dict[str, Any]) -> str:
    mse = metrics.get("dream_pixel_mse_by_step")
    if not isinstance(mse, list) or not mse:
        return "Failure analysis: dream-real gap metrics were not available."
    worst_index = max(range(len(mse)), key=lambda index: float(mse[index]))
    return (
        "Failure case: the largest open-loop dream-real pixel gap occurs at "
        f"step {worst_index + 1} with MSE {float(mse[worst_index]):.4f}."
    )


def _metrics_table(metrics: dict[str, Any], keys: list[str]) -> str:
    if not metrics:
        return ""
    rows = []
    for key in keys:
        if key not in metrics:
            continue
        value = metrics[key]
        if isinstance(value, float):
            rendered = f"{value:.5f}"
        else:
            rendered = escape(str(value))
        rows.append(f"<tr><th>{escape(key)}</th><td>{rendered}</td></tr>")
    if not rows:
        return ""
    return f"<table><tbody>{''.join(rows)}</tbody></table>"


def _image_tag(path: Path | None, asset_dir: Path) -> str:
    if path is None:
        return "<p>Missing image artifact.</p>"
    rel = _copy_asset(path, asset_dir)
    return f'<img src="{escape(rel)}" alt="{escape(path.stem)}">'


def _video_tag(path: Path | None, asset_dir: Path) -> str:
    if path is None:
        return "<p>Missing video artifact.</p>"
    rel = _copy_asset(path, asset_dir)
    return f'<video src="{escape(rel)}" controls muted loop playsinline></video>'


def _copy_asset(path: Path, asset_dir: Path) -> str:
    prefix = "_".join(path.parts[-3:-1])
    safe_prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", prefix).strip("_")
    target_name = f"{safe_prefix}_{path.name}" if safe_prefix else path.name
    target = asset_dir / target_name
    if path.resolve() != target.resolve():
        shutil.copy2(path, target)
    return f"assets/{target.name}"


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _page(*, title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #18212f;
      --muted: #5f6b7a;
      --line: #d7dde5;
      --paper: #f6f8fb;
      --accent: #0f766e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system,
        BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--paper);
    }}
    .hero {{
      min-height: 32vh;
      display: flex;
      align-items: end;
      padding: 56px clamp(20px, 5vw, 72px);
      background: #101820;
      color: white;
    }}
    .eyebrow {{
      margin: 0 0 10px;
      color: #86efac;
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0;
      font-weight: 700;
    }}
    h1 {{
      margin: 0;
      max-width: 980px;
      font-size: clamp(36px, 7vw, 72px);
      line-height: 0.96;
      letter-spacing: 0;
    }}
    section {{
      padding: 34px clamp(20px, 5vw, 72px);
      border-bottom: 1px solid var(--line);
      background: white;
    }}
    section:nth-child(odd) {{ background: #f9fbfd; }}
    h2 {{ margin: 0 0 18px; font-size: 24px; letter-spacing: 0; }}
    h3 {{ margin: 0 0 10px; font-size: 15px; color: var(--muted); letter-spacing: 0; }}
    img, video {{
      display: block;
      width: 100%;
      max-height: 520px;
      object-fit: contain;
      background: #0d1117;
      border: 1px solid var(--line);
    }}
    .grid {{
      display: grid;
      gap: 16px;
      align-items: start;
    }}
    .two {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .three {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    table {{
      width: 100%;
      margin-top: 16px;
      border-collapse: collapse;
      font-size: 14px;
      background: white;
    }}
    pre {{
      overflow-x: auto;
      white-space: pre-wrap;
      background: #101820;
      color: white;
      padding: 14px;
      font-size: 13px;
      line-height: 1.45;
    }}
    th, td {{
      text-align: left;
      border: 1px solid var(--line);
      padding: 8px 10px;
      vertical-align: top;
    }}
    th {{ color: var(--muted); font-weight: 650; }}
    .failure {{
      margin: 16px 0 0;
      color: var(--accent);
      font-weight: 650;
    }}
    @media (max-width: 900px) {{
      .two, .three {{ grid-template-columns: 1fr; }}
      .hero {{ min-height: 26vh; }}
    }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ae-run", type=Path, default=None)
    parser.add_argument("--vqvae-run", type=Path, default=None)
    parser.add_argument("--dream-run", type=Path, default=None)
    parser.add_argument("--planner-run", type=Path, default=None)
    parser.add_argument("--failure-run", type=Path, default=None)
    parser.add_argument("--title", type=str, default="dreamtrack CarRacing World Model Demo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index = build_dashboard(
        out=args.out,
        ae_run=args.ae_run,
        vqvae_run=args.vqvae_run,
        dream_run=args.dream_run,
        planner_run=args.planner_run,
        failure_run=args.failure_run,
        title=args.title,
    )
    print(index)


if __name__ == "__main__":
    main()
