# tmlr_readiness.yaml

Original experiment configuration. Materialize with `python scripts/materialize_configs.py`.

Output path: `experiments/deadline_comm/configs/tmlr_readiness.yaml`.

SHA256: `2d78d7c91d98272d0ad28f3f6f44732384c0f81fd6067e3c8d802cda32d03f25`

```yaml
output: experiments/deadline_comm/runs/tmlr_readiness_20261119_001
seeds: [0,1,2,3,4,5,6,7,8,9]
checkpoints: [rr_cap,dac]
deadlines_ms: [1.5,1.75,2.0,2.25,2.5]
episodes: 10
reset_base: 14300000
wall_hours: 2
```
