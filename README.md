# dreamtrack

Autoresearch infrastructure for pixel-based world models.

`dreamtrack` trains a world model for Gymnasium `CarRacing-v3`: collect pixel
rollouts, learn a visual tokenizer, learn action-conditioned dynamics, imagine
future rollouts, and use those imagined futures for MPC planning.

The current focus is not a polished benchmark result. It is a clean research
loop that can run experiments on Modal GPUs and track outputs in a minimal web
dashboard.

## What is included

- CarRacing rollout collection from random, heuristic, and hybrid policies
- Convolutional autoencoder and VQ-VAE visual tokenizers
- Transformer dynamics model over VQ token grids
- Dream-vs-real rollout visualization
- Random shooting and CEM MPC planners
- Failure analysis for model drift and bad imagined futures
- Static demo builder for generated artifacts
- Modal jobs for remote collection and GPU training
- Read-only experiment dashboard for metrics, logs, plots, videos, and run status

## Setup

```bash
uv sync
```

Quick local environment check:

```bash
uv run python -m dreamtrack.data.collect_rollouts \
  --episodes 1 \
  --policy random \
  --image-size 64 \
  --out data/car_racing_debug \
  --max-steps 40
```

## Local dashboard

```bash
uv run python -m dreamtrack.experiments.dashboard \
  --runs-root runs \
  --port 8787
```

Open `http://127.0.0.1:8787`.

## Modal

Authenticate first:

```bash
modal token set
```

Run a small remote collection job:

```bash
modal run infra/modal_app.py::collect_rollouts_remote \
  --name collect-debug \
  --episodes 1 \
  --max-steps 40
```

Train the VQ-VAE remotely:

```bash
modal run infra/modal_app.py::train_vqvae_remote \
  --data-path /dreamtrack/datasets/<dataset-run>/rollouts.npz
```

Train dynamics remotely:

```bash
modal run infra/modal_app.py::train_dynamics_remote \
  --data-path /dreamtrack/datasets/<dataset-run>/rollouts.npz \
  --tokenizer-path /dreamtrack/runs/<vqvae-run>/vqvae/checkpoints/best_codebook.pt
```

Deploy the dashboard:

```bash
modal deploy infra/modal_app.py
```

Each Modal job writes `run.json`, `metrics.json`, `logs.txt`, plots, videos, and
checkpoints to the `dreamtrack-runs` Modal Volume.

## Local training commands

Train a VQ-VAE:

```bash
uv run python -m dreamtrack.train.train_vqvae \
  --config dreamtrack/configs/tokenizer_vqvae.yaml \
  --data data/car_racing_debug/rollouts.npz \
  --out runs/vqvae_debug
```

Train dynamics:

```bash
uv run python -m dreamtrack.train.train_dynamics \
  --config dreamtrack/configs/dynamics_transformer.yaml \
  --data data/car_racing_debug/rollouts.npz \
  --tokenizer runs/vqvae_debug/checkpoints/best_codebook.pt \
  --out runs/dynamics_debug
```

Evaluate planning:

```bash
uv run python -m dreamtrack.eval.eval_planner \
  --dynamics runs/dynamics_debug/checkpoints/best.pt \
  --planner cem \
  --episodes 1 \
  --max-steps 120 \
  --horizon 8 \
  --candidates 96 \
  --elites 16 \
  --iterations 2 \
  --out runs/planner_eval_debug
```

## Status

The end-to-end baseline is functional: the learned planner can run in the real
environment and beat random on short evaluations. It is not yet a strong
CarRacing agent. The next major work is to add full remote pipeline jobs and a
config-only autoresearch loop that proposes, runs, scores, and summarizes
experiments.
