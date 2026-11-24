# hidden_mechanism.yaml

Original experiment configuration. Materialize with `python scripts/materialize_configs.py`.

Output path: `experiments/deadline_comm/configs/hidden_mechanism.yaml`.

SHA256: `4011fd03444603fcca8134a6231fce0f5f22a771fca1e15f34c1b5bf38009689`

```yaml
output: experiments/deadline_comm/runs/hidden_mechanism_20261119_001
seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
tasks: [broadcast, xor]
methods: [no_comm, ideal_full, periodic, rr_cap, dac]
agent_counts: [5]
updates: 1000
batch_size: 16
horizon: 20
evaluation_episodes: 50
devices: [cuda:0, cuda:1]
wall_hours: 24
```
