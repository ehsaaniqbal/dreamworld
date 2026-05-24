# Architecture

`dreamworld` is a small autoresearch loop for pixel-based world models.

## Runtime

Normal runs use one deployed Modal app:

```text
local CLI or dashboard
  -> Modal Function.from_name("dreamworld", ...)
  -> deployed Modal function
  -> Modal Volume artifacts
```

Use `modal run` only for development. It creates ephemeral Modal apps.

Launch deployed jobs with:

```bash
uv run python -m dreamworld.research.launch loop --max-iterations 3
```

## Modal App

The deployed app is `dreamworld`.

It owns:

- `dashboard`
- `run_autoresearch_loop_remote`
- `run_autoresearch_once_remote`
- `propose_next_experiment_remote`
- `evaluate_completed_run_remote`
- `run_pipeline_remote`
- one-off collect, train, eval, and failure-analysis jobs

## Storage

The durable source of truth is the `dreamworld-runs` Modal Volume:

```text
/dreamworld
  /runs
    /<run-id>
      run.json
      metrics.json
      logs.txt
      note.md
      checkpoints/
      plots and videos

  /research
    goal.md
    state.json
    /queue
      pending/*.yaml
      done/*.yaml        # reserved for completed spec archival
    /notes
      <proposal>.md
```

The dashboard is read-only and renders this Volume state.

## Autoresearch Loop

Current loop:

```text
read state and prior runs
propose a YAML spec
run the full Modal pipeline
score the completed run
write a note
keep or discard the result
update state
stop on budget or goal
```

The loop is config-only. It mutates experiment specs, not Python model code.

## Model Stack

Current model path:

```text
CarRacing pixels
  -> VQ-VAE tokenizer
  -> token transformer dynamics
       predicts next tokens
       predicts reward
       predicts done probability
  -> CEM planner in imagined token rollouts
  -> real environment evaluation
```

This is the baseline architecture. The next major model variant should be an RSSM-style latent dynamics model behind a config flag such as `model.type: rssm`.
