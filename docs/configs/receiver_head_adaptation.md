# receiver_head_adaptation.yaml

Original experiment configuration. Materialize with `python scripts/materialize_configs.py`.

Output path: `experiments/deadline_comm/configs/receiver_head_adaptation.yaml`.

SHA256: `70b222c51262941632269e605463c2e0bc13733f05bb3fcff18e061b9146fe89`

```yaml
output: experiments/deadline_comm/runs/receiver_head_adaptation_20261123_001
seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
condition: [medium, 2.0]
method: rr_cap
schedulers: [normal, source_aware]
train_episodes: 10
test_episodes: 20
test_reset_base: 17300000
updates: 100
learning_rate: 0.001
wall_seconds: 600
ridge: 0.001
checkpoint_template: experiments/deadline_comm/runs/hidden_mechanism_20261119_001/broadcast_n5_seed{seed}_rr_cap/final.pt
feature_template: experiments/deadline_comm/runs/tmlr_receiver_probe_20261119_001/seed{seed}_rr_cap_medium_2ms_{scheduler}.npz
scope: CPU-only supervised follower-head adaptation of existing Oracle-free learned broadcast models; not new MARL training or standard-benchmark evidence.
```
