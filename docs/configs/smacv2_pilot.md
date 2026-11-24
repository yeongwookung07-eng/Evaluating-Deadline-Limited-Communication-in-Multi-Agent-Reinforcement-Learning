# smacv2_pilot.yaml

Original experiment configuration. Materialize with `python scripts/materialize_configs.py`.

Output path: `experiments/deadline_comm/configs/smacv2_pilot.yaml`.

SHA256: `f3cf941fba9a3da9f25e02d8ef79c71161acd5277608f25df21d210de6fa9c6c`

```yaml
output: experiments/deadline_comm/runs/smacv2_pilot_20261119_001
seeds: [0, 1, 2]
scenarios: [terran, protoss]
arms: [ideal_full, no_comm]
stage_order: [ideal_learning_gate, no_comm_only_for_passing_scenarios]
devices: [0, 1]
environment_steps: 100000
wall_hours: 12
evaluation_interval: 25000
evaluation_episodes: 32
train_seed_base: 15000000
eval_seed_base: 16000000
batch_size: 16
replay_capacity: 1000
epsilon_anneal_steps: 50000
epsilon_final: 0.05
updates_per_episode: 1
hidden_dim: 64
message_dim: 8
learning_gate_win_rate: 0.20
learner:
  optimizer: adam
  learning_rate: 0.0005
  gamma: 0.99
  grad_clip: 10.0
  target_update_interval: 200
  bootstrap_timeouts: true
```
