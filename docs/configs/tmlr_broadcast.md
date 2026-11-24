# tmlr_broadcast.yaml

Original experiment configuration. Materialize with `python scripts/materialize_configs.py`.

Output path: `experiments/deadline_comm/configs/tmlr_broadcast.yaml`.

SHA256: `6a37c4f049106954a061860a06d06a6df8405e775f2cc1fa214dd75dcedd9c02`

```yaml
output: experiments/deadline_comm/runs/tmlr_broadcast_20261119_001
seeds: [0,1,2,3,4,5,6,7,8,9]
methods: [no_comm,ideal_full,periodic,rr_cap,dac]
conditions: [[slow,5.0],[medium,2.0],[medium,5.0]]
episodes: 20
reset_base: 14200000
wall_hours: 3
```
