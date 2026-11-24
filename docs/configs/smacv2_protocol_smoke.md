# smacv2_protocol_smoke.yaml

Original experiment configuration. Materialize with `python scripts/materialize_configs.py`.

Output path: `experiments/deadline_comm/configs/smacv2_protocol_smoke.yaml`.

SHA256: `592eea9dbef06d805dda3fcb66e809cba333c037ad21f7486ce3764664ff28b0`

```yaml
output: experiments/deadline_comm/runs/smacv2_protocol_20261121_001_smoke
pilot: experiments/deadline_comm/runs/smacv2_pilot_20261119_001
runtime: experiments/deadline_comm/runs/smacv2_runtime_20261119_001
scenario: terran
seeds:
- 0
devices:
- 0
- 1
training_milestones:
- 102000
wall_hours: 12
batch_size: 4
replay_capacity: 1000
hidden_dim: 64
message_dim: 8
train_seed_base: 50000000
gate_seed_base: 52000000
heldout_seed_base: 53000000
gate_episodes: 2
evaluation_episodes: 1
learning_gate_win_rate: 0.2
continuation_epsilon_start: 0.3
continuation_epsilon_final: 0.05
continuation_epsilon_steps: 200000
learner:
  optimizer: adam
  learning_rate: 0.0005
  gamma: 0.99
  grad_clip: 10.0
  target_update_interval: 200
  bootstrap_timeouts: true
evaluation:
  conditions:
    slow_5ms:
      bandwidth: 20000
      deadline_ms: 5.0
    medium_2ms:
      bandwidth: 100000
      deadline_ms: 2.0
  methods:
  - ideal_full
  - no_comm
  - full
  - rr_cap
  - dac
  fixed_ready_ns: 500000
  r_post_ns: 450000
  decay_lambda: 1.0
  budget_bytes: 120.0
  warmup_episodes: 0
  bootstrap_samples: 10000
```
