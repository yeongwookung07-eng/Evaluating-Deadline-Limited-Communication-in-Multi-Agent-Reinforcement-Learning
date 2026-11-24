# reviewer_endpoints.yaml

Original experiment configuration. Materialize with `python scripts/materialize_configs.py`.

Output path: `experiments/deadline_comm/configs/reviewer_endpoints.yaml`.

SHA256: `17e604a7b290a78cf75d1eb4aa9b0729b517a64081edfd5de7827aada8b1ad51`

```yaml
parent_run: experiments/deadline_comm/runs/training_controls_20261118_001
output: experiments/deadline_comm/runs/reviewer_endpoints_20261118_001
seeds: [0, 1, 2]
links: {fast: 1000000, medium: 100000, slow: 20000}
deadlines_ms: [5.0, 2.5, 2.0, 1.5]
episodes: 30
warmups: 2
reset_seed_start: 7200000
bootstrap_resamples: 10000
wall_hours: 6
```
