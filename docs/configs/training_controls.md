# training_controls.yaml

Original experiment configuration. Materialize with `python scripts/materialize_configs.py`.

Output path: `experiments/deadline_comm/configs/training_controls.yaml`.

SHA256: `fd21286eac9967736850500abe16ab0e390337315cfe25ae52256b79695a2a1c`

```yaml
seeds: [0, 1, 2]
devices: [cuda:0, cuda:1]
base_run: experiments/deadline_comm/runs/day2_dac_marl_main_20261118_001
continuation_updates: 500
scratch_updates: 1500
oracle_episodes: 50
batch_size: 16
wall_limit_hours: 8
checkpoint_interval: 250
evaluation_episodes: 30
warmup_episodes: 2
conditions: [[medium, 5.0], [slow, 5.0], [medium, 2.0], [medium, 1.5]]
links: {medium: 100000, slow: 20000}
evaluation_seed_start: 6200000
training_seed_start: 6100000
```
