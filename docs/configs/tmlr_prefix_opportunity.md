# tmlr_prefix_opportunity.yaml

Original experiment configuration. Materialize with `python scripts/materialize_configs.py`.

Output path: `experiments/deadline_comm/configs/tmlr_prefix_opportunity.yaml`.

SHA256: `a1121eeef6c71d76dcbd6fa538cb933dc245a13837c2a145891825f535d5ace3`

```yaml
output: experiments/deadline_comm/runs/tmlr_prefix_opportunity_20261119_001
source: experiments/deadline_comm/runs/tmlr_prefix_20261119_001/episodes.jsonl
max_episodes: 800
max_oracle_steps: 11200
scope: Replay saved prefixes exactly; schedule unresolved groups at earliest distinct completion steps using global state. This privileged remaining-return reference tests reward opportunity, not learned-policy competence or a deployable baseline. All prefixes retained, no selection on outcome.
```
