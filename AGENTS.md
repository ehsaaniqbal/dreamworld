# Agent Rules

Read `ARCHITECTURE.md` before changing the loop.

Use the deployed Modal app for normal runs:

```bash
uv run python -m dreamworld.research.launch loop --max-iterations 3
```

Do not use `modal run` except for debugging Modal code.

Do not train on the laptop. The Mac is the control plane. Modal is the compute plane.

Default research permission is config-only:

- edit YAML specs
- adjust rollout, tokenizer, dynamics, and planner config
- read metrics, videos, notes, and state
- do not edit Python model code unless the human explicitly allows it

Before launching compute:

- check `git status`
- check `/dreamworld/research/state.json`
- respect `requested_status`
- avoid launching if a live `active_call.json` exists unless the human says to continue

After every experiment:

- write or verify `note.md`
- keep/discard/candidate-best the run
- update `state.json`
- move the spec out of `pending`

