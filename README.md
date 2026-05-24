# dreamworld

A tiny codebase for building world models from pixels, and for studying how to automate the researcher running the experiments.

The first environment is Gymnasium `CarRacing-v3`. The loop is:

```text
collect rollouts -> learn a visual tokenizer -> learn dynamics -> imagine futures -> plan actions -> inspect failures
```

The goal is to build an agent-friendly research loop: propose an experiment, run it on remote GPUs, track the artifacts, compare the result, and decide what to try next.

## Why world models?

World models have a little bit of everything:

1. Pixels need compression
2. Actions need dynamics
3. Rewards are sparse and easy to exploit
4. Planning exposes model errors quickly
5. Bad predictions are visible as visuals, not just numbers

CarRacing is small enough to iterate on, but still annoying enough to be useful.

Inspired by [AutoGo](https://github.com/ericjang/autogo) and [autoresearch](https://github.com/karpathy/autoresearch).

## Setup

```bash
uv sync
```

Smoke test:

```bash
uv run python -m dreamworld.data.collect_rollouts \
  --episodes 1 \
  --policy random \
  --image-size 64 \
  --out data/car_racing_debug \
  --max-steps 40
```

## Local dashboard

```bash
uv run python -m dreamworld.experiments.dashboard \
  --runs-root runs \
  --port 8787
```

Open `http://127.0.0.1:8787`.

## Modal

Authenticate:

```bash
modal token set
```

Run a full debug pipeline:

```bash
modal run infra/modal_app.py::run_pipeline_remote \
  --spec-name baseline_debug
```

Deploy the dashboard:

```bash
modal deploy infra/modal_app.py
```

The Modal jobs write to the `dreamworld-runs` Volume:

```text
/dreamworld/runs/<run-id>/
  run.json
  metrics.json
  logs.txt
  checkpoints/
  plots and videos
```

## One-off jobs

```bash
modal run infra/modal_app.py::collect_rollouts_remote \
  --name collect-debug \
  --episodes 1 \
  --max-steps 40
```

```bash
modal run infra/modal_app.py::train_vqvae_remote \
  --data-path /dreamworld/datasets/<dataset-run>/rollouts.npz
```

```bash
modal run infra/modal_app.py::train_dynamics_remote \
  --data-path /dreamworld/datasets/<dataset-run>/rollouts.npz \
  --tokenizer-path /dreamworld/runs/<vqvae-run>/vqvae/checkpoints/best_codebook.pt
```

