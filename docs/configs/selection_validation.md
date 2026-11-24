# selection_validation.yaml

Original experiment configuration. Materialize with `python scripts/materialize_configs.py`.

Output path: `experiments/deadline_comm/configs/selection_validation.yaml`.

SHA256: `8c3c1721098a3ebaef8461f6d09edfb24a61ae977733aeff15e430e4f7e7b267`

```yaml
output: experiments/deadline_comm/runs/selection_validation_20261118_001
seeds: [0, 1, 2]
conditions: [[slow, 5.0], [medium, 2.0], [medium, 5.0]]
links: {slow: 20000, medium: 100000}
methods: [rr_cap, random_cap, age_cap, dac]
episodes: 30
modes: [measured, fixed_ready]
wall_hours: 4
checkpoints:
  full2k: experiments/deadline_comm/runs/day2_gap_closure_20261118_001/seed{seed}/checkpoint_0500.pt
  adapted2k: experiments/deadline_comm/runs/day2_followup_20261118_001/seed{seed}/checkpoint_0500.pt
  new_adapted2k: experiments/deadline_comm/runs/training_controls_20261118_001/continuation_seed{seed}_dac/checkpoint_0500.pt
```
