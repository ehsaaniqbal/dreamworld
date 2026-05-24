# dreamtrack

Modern world-model experiments for Gymnasium `CarRacing-v3`.

The project is intentionally incremental: collect pixel rollouts, learn visual
representations, learn action-conditioned dynamics, imagine futures, and use the
learned model for planning.

## Setup

```bash
uv sync
```

## Research Dashboard and Modal Jobs

The project includes a file-backed experiment dashboard. It can inspect local
`runs/` artifacts today and the same app is served from Modal against the
shared `dreamtrack-runs` Volume.

Local dashboard:

```bash
uv run python -m dreamtrack.experiments.dashboard \
  --runs-root runs \
  --port 8787
```

Then open `http://127.0.0.1:8787`.

Modal entrypoints:

```bash
modal run infra/modal_app.py::collect_rollouts_remote

modal run infra/modal_app.py::train_vqvae_remote \
  --data-path /dreamtrack/datasets/<dataset-run>/rollouts.npz

modal run infra/modal_app.py::train_dynamics_remote \
  --data-path /dreamtrack/datasets/<dataset-run>/rollouts.npz \
  --tokenizer-path /dreamtrack/runs/<vqvae-run>/vqvae/checkpoints/best_codebook.pt

modal serve infra/modal_app.py
```

Each Modal job writes a `run.json`, `metrics.json`, `logs.txt`, plots, videos,
and checkpoints under `/dreamtrack/runs/<run-id>/`. The dashboard is read-only:
jobs own training state; the webpage only renders the files they write.

## Quick Environment Check

```bash
uv run python -m dreamtrack.data.collect_rollouts \
  --episodes 1 \
  --policy random \
  --image-size 64 \
  --out data/car_racing_debug \
  --max-steps 40
```

This creates a `CarRacing-v3` environment, takes random actions, preprocesses
frames, and writes a tiny dataset plus sample video. It is the smallest useful
end-to-end check for the current codebase.

## Phase 1 Rollout Collection

```bash
uv run python -m dreamtrack.data.collect_rollouts \
  --episodes 10 \
  --policy hybrid \
  --image-size 64 \
  --out data/car_racing_train_small
```

For a quick end-to-end check:

```bash
uv run python -m dreamtrack.data.collect_rollouts \
  --episodes 1 \
  --policy random \
  --image-size 64 \
  --out data/car_racing_debug \
  --max-steps 40
```

Outputs:

```text
data/car_racing_debug/
  rollouts.npz
  sample.mp4
  stats.json
```

## Phase 2 Autoencoder

```bash
uv run python -m dreamtrack.train.train_autoencoder \
  --config dreamtrack/configs/tokenizer_ae.yaml \
  --data data/car_racing_debug/rollouts.npz \
  --out runs/ae_debug
```

For a quick CPU debug run:

```bash
uv run python -m dreamtrack.train.train_autoencoder \
  --config dreamtrack/configs/tokenizer_ae.yaml \
  --data data/car_racing_debug/rollouts.npz \
  --out runs/ae_debug \
  --epochs 3 \
  --batch-size 16 \
  --latent-dim 64 \
  --device cpu
```

Evaluate reconstruction quality:

```bash
uv run python -m dreamtrack.eval.eval_reconstruction \
  --checkpoint runs/ae_debug/checkpoints/best.pt \
  --data data/car_racing_debug/rollouts.npz \
  --out runs/ae_debug/eval
```

Outputs include checkpoints, reconstruction grids, loss curves, and metrics JSON.

## Phase 3 VQ-VAE Tokenizer

```bash
uv run python -m dreamtrack.train.train_vqvae \
  --config dreamtrack/configs/tokenizer_vqvae.yaml \
  --data data/car_racing_debug/rollouts.npz \
  --out runs/vqvae_debug
```

For a quick CPU debug run:

```bash
uv run python -m dreamtrack.train.train_vqvae \
  --config dreamtrack/configs/tokenizer_vqvae.yaml \
  --data data/car_racing_debug/rollouts.npz \
  --out runs/vqvae_debug \
  --epochs 3 \
  --batch-size 16 \
  --embedding-dim 32 \
  --num-embeddings 64 \
  --device cpu
```

The trainer saves both `checkpoints/best.pt` for reconstruction loss and
`checkpoints/best_codebook.pt` for codebook usage. On tiny debug datasets,
codebook collapse is common; inspect `codebook_perplexity.png`,
`tokens_epoch_*.png`, and `metrics.json`.

```bash
uv run python -m dreamtrack.eval.eval_reconstruction \
  --checkpoint runs/vqvae_debug/checkpoints/best_codebook.pt \
  --data data/car_racing_debug/rollouts.npz \
  --out runs/vqvae_debug/eval_codebook
```

## Phase 4 Token Dynamics

Train action-conditioned dynamics over VQ-VAE token grids:

```bash
uv run python -m dreamtrack.train.train_dynamics \
  --config dreamtrack/configs/dynamics_transformer.yaml \
  --data data/car_racing_debug/rollouts.npz \
  --tokenizer runs/vqvae_debug/checkpoints/best_codebook.pt \
  --out runs/dynamics_debug
```

Quick CPU debug run:

```bash
uv run python -m dreamtrack.train.train_dynamics \
  --config dreamtrack/configs/dynamics_transformer.yaml \
  --data data/car_racing_debug/rollouts.npz \
  --tokenizer runs/vqvae_debug/checkpoints/best_codebook.pt \
  --out runs/dynamics_debug \
  --epochs 1 \
  --batch-size 8 \
  --context-length 4 \
  --d-model 64 \
  --n-heads 4 \
  --n-layers 1 \
  --device cpu
```

Evaluate one-step token/reward/done prediction:

```bash
uv run python -m dreamtrack.eval.eval_dynamics \
  --checkpoint runs/dynamics_debug/checkpoints/best.pt \
  --data data/car_racing_debug/rollouts.npz \
  --out runs/dynamics_debug/eval
```

## Phase 5 Dream Rollout Visualization

Generate teacher-forced and open-loop dream-vs-real videos:

```bash
uv run python -m dreamtrack.eval.compare_real_vs_dream \
  --data data/car_racing_debug/rollouts.npz \
  --dynamics runs/dynamics_debug/checkpoints/best.pt \
  --out runs/dynamics_debug/dream_vs_real \
  --horizon 30
```

Outputs:

```text
runs/dynamics_debug/dream_vs_real/
  teacher_forced_vs_real.mp4
  open_loop_dream_vs_real.mp4
  metrics.json
```

## Phase 6 Dream Planning

Compare random, heuristic, and learned-world MPC planners in the real
CarRacing environment:

```bash
uv run python -m dreamtrack.eval.eval_planner \
  --dynamics runs/dynamics_debug/checkpoints/best.pt \
  --planner cem \
  --episodes 5 \
  --max-steps 500 \
  --horizon 15 \
  --candidates 512 \
  --elites 64 \
  --iterations 4 \
  --out runs/planner_eval
```

For a fast wiring check:

```bash
uv run python -m dreamtrack.eval.eval_planner \
  --dynamics runs/dynamics_debug/checkpoints/best.pt \
  --planner random \
  --episodes 1 \
  --max-steps 8 \
  --horizon 3 \
  --candidates 4 \
  --out runs/planner_eval_debug \
  --device cpu
```

Outputs include per-agent gameplay videos, `candidate_plan_debug.mp4`,
`reward_curve.png`, `metrics.json`, and candidate action debug data for the
first dream-planning episode.

## Phase 7 Failure Analysis

Mine open-loop dream failures and export ranked clips:

```bash
uv run python -m dreamtrack.eval.failure_analysis \
  --data data/car_racing_debug/rollouts.npz \
  --dynamics runs/dynamics_debug/checkpoints/best.pt \
  --out runs/failure_analysis \
  --horizon 20 \
  --top-k 5 \
  --planner-metrics runs/planner_eval/metrics.json
```

Outputs:

```text
runs/failure_analysis/
  failures/
    failure_001.mp4
    failure_001.json
  dream_real_gap.png
  metrics.json
  summary.md
```

## Static Demo Dashboard

Build a local HTML demo from the generated artifacts:

```bash
uv run python -m dreamtrack.viz.dashboard \
  --ae-run runs/ae_debug \
  --vqvae-run runs/vqvae_debug \
  --dream-run runs/dynamics_debug/dream_vs_real \
  --planner-run runs/planner_eval \
  --failure-run runs/failure_analysis \
  --out runs/demo
```

For the current debug artifacts:

```bash
uv run python -m dreamtrack.viz.dashboard \
  --ae-run runs/ae_debug \
  --vqvae-run runs/vqvae_debug \
  --dream-run runs/dynamics_debug/dream_vs_real \
  --planner-run runs/planner_eval_debug \
  --failure-run runs/failure_analysis_debug \
  --out runs/demo_debug
```

Open `runs/demo_debug/index.html` in a browser to view the demo.

## Current Runs

The repository includes a small debug-scale run and a larger local run produced
from 20 hybrid-policy episodes:

```text
data/car_racing_train_small/
runs/vqvae_train_small/
runs/dynamics_train_small/
runs/planner_eval_train_small/
runs/failure_analysis_train_small/
runs/demo_train_small/

data/car_racing_train/
runs/vqvae_train/
runs/dynamics_train/
runs/planner_eval_train_prior/
runs/failure_analysis_train/
runs/demo_train/
```

Key observed metrics from the small run:

```text
dynamics val token accuracy: 0.8548
40-step open-loop mean token accuracy: 0.8676
planner eval, 1 episode x 120 steps:
  random: 0.539
  heuristic: 68.128
  dream CEM: 3.625
  dream CEM + action prior: 11.437
```

Key observed metrics from the larger run:

```text
dataset: 7,917 frames from 20 hybrid episodes
VQ-VAE eval MSE: 0.001317
VQ-VAE codebook perplexity: 23.4027
dynamics one-step token accuracy: 0.9342
dynamics one-step reward MAE: 0.0240
50-step open-loop mean token accuracy: 0.4984
50-step teacher-forced mean token accuracy: 0.9103
planner eval, 1 episode x 160 steps:
  random: -0.326
  heuristic: 64.128
  dream CEM + action prior: 11.344
failure analysis windows scanned: 788
```

Open `runs/demo_train/index.html` in a browser to view the latest static demo.

These runs satisfy the wiring-level planning gate because dream CEM beats
random in the real environment, and the full loop now produces reconstructions,
dream rollouts, planner videos, candidate plans, reward curves, gap plots, and
failure clips. It is not yet a strong CarRacing agent because the learned-world
planner remains far below the heuristic baseline and still underpredicts crash
or terminal penalties in the worst failure windows.

Suggested next scale commands:

```bash
uv run python -m dreamtrack.data.collect_rollouts \
  --episodes 30 \
  --policy hybrid \
  --image-size 64 \
  --out data/car_racing_train \
  --max-steps 500

uv run python -m dreamtrack.train.train_vqvae \
  --config dreamtrack/configs/tokenizer_vqvae.yaml \
  --data data/car_racing_train/rollouts.npz \
  --out runs/vqvae_train \
  --epochs 20 \
  --batch-size 128 \
  --embedding-dim 32 \
  --num-embeddings 128

uv run python -m dreamtrack.train.train_dynamics \
  --config dreamtrack/configs/dynamics_transformer.yaml \
  --data data/car_racing_train/rollouts.npz \
  --tokenizer runs/vqvae_train/checkpoints/best_codebook.pt \
  --out runs/dynamics_train \
  --epochs 20 \
  --batch-size 64 \
  --context-length 16 \
  --d-model 256 \
  --n-heads 4 \
  --n-layers 4

uv run python -m dreamtrack.eval.eval_planner \
  --dynamics runs/dynamics_train/checkpoints/best.pt \
  --planner cem \
  --episodes 5 \
  --max-steps 1000 \
  --horizon 15 \
  --candidates 512 \
  --elites 64 \
  --iterations 4 \
  --out runs/planner_eval
```
