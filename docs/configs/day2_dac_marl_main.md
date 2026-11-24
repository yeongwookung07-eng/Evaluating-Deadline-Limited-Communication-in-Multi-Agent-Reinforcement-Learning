# day2_dac_marl_main.yaml

Original experiment configuration. Materialize with `python scripts/materialize_configs.py`.

Output path: `experiments/deadline_comm/configs/day2_dac_marl_main.yaml`.

SHA256: `9193ebbf3d5121d92fe34ec12d8128f24e932214200554e97bd0a05fb1eb1d97`

```yaml
run_name: day2_dac_marl_main_20261118_001
seeds: [0, 1, 2]
training:
  episodes: 1500
  oracle_episodes: 50
  batch_size: 16
  replay_capacity: 2000
  epsilon_start: 1.0
  epsilon_end: 0.05
  evaluation_interval: 200
  evaluation_episodes: 30
  target_update_interval: 100
  gamma: 0.99
  learning_rate: 0.0005
  grad_clip: 10.0
  bootstrap_timeouts: false
  hidden_dim: 64
  bc_weight: 0.5
calibration:
  seed_start: 2100000
  episodes: 20
  iterations: 14
  target_bytes: 120.0
  tolerance: 0.05
evaluation:
  seed_start: 2200000
  episodes: 20
  repeats: 3
  warmup_episodes: 2
  deadlines_ms: [5.0, 2.5, 2.0, 1.5]
  links: {fast: 1000000, medium: 100000, slow: 20000}
  ablations: [[slow, 5.0], [medium, 1.5], [medium, 2.0]]
harness:
  deadline_ns: 5000000
  r_post_ns: 450000
  training_ready_ns: 500000
  link: {bandwidth_bytes_s: 100000, latency_ns: 50000, header_bytes: 16}
  gate: {period: 2, p: 0.5, q_threshold: 0.1, message_threshold: 0.1}
  dac: {alpha: 0.1, threshold: 0.0, budget_bytes: 120.0, packet_bytes: 48, burst_packets: 3}
  decay_lambda: 0.9
```
