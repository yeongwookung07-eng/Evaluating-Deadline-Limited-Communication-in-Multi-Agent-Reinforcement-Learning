# tmlr_prefix.yaml

Original experiment configuration. Materialize with `python scripts/materialize_configs.py`.

Output path: `experiments/deadline_comm/configs/tmlr_prefix.yaml`.

SHA256: `ccf7e8b36454a83eeb0a8812ee8fa953176c397bf2cd7397e1a24d96f9ca7b0b`

```yaml
output: experiments/deadline_comm/runs/tmlr_prefix_20261119_001
seeds: [0,1,2,3,4,5,6,7,8,9]
checkpoints: [rr_cap,dac]
conditions: [[slow,5.0],[medium,2.0]]
episodes: 20
reset_base: 14100000
wall_hours: 2
```
