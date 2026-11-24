# reviewer_hidden_training.yaml

Original experiment configuration. Materialize with `python scripts/materialize_configs.py`.

Output path: `experiments/deadline_comm/configs/reviewer_hidden_training.yaml`.

SHA256: `f9665bf7f4e44fe634ee0cf41d5d247cd0472fdd431ab604aada3f9d77360ecd`

```yaml
output: experiments/deadline_comm/runs/reviewer_hidden_training_20261118_001
seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
tasks: [broadcast, xor]
methods: [no_comm, ideal_full, periodic, dac]
agent_counts: [5, 10, 20, 50]
updates: 1000
batch_size: 16
horizon: 20
evaluation_episodes: 50
devices: [cuda:0, cuda:1]
wall_hours: 48
# Scale a task only after its ideal-link N5 policies learn robustly.
min_ideal_accuracy: 0.75
min_ideal_successful_seeds: 8
```
