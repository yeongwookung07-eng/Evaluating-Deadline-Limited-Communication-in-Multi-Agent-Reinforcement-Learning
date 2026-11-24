# tmlr_fresh.yaml

Original experiment configuration. Materialize with `python scripts/materialize_configs.py`.

Output path: `experiments/deadline_comm/configs/tmlr_fresh.yaml`.

SHA256: `3354a56b07b2af4ccfd049f271d41d6c1d91843b9ae4fa498f476e20d6f678cd`

```yaml
output: experiments/deadline_comm/runs/tmlr_fresh_20261119_001
seeds: [0,1,2,3,4,5,6,7,8,9]
checkpoints: [rr_cap,dac]
conditions: [[slow,5.0],[medium,2.0]]
episodes: 10
reset_base: 14000000
sample_steps: [0,2,4,6,8,10,12]
wall_hours: 6
```
