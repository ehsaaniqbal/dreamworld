# Goal

Improve the CarRacing dream-planner score using config-only experiments.

Use the deployed Modal app. Do not train locally.

Stop when either:

- the experiment budget is exhausted
- `requested_status` is `paused` or `stopped`
- a confirmed best clears the target improvement

Treat a one-episode improvement as `candidate_best`, not confirmed progress.

Write short notes. Prefer small, interpretable changes over broad sweeps.

