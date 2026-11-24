# review_identity.yaml

Original experiment configuration. Materialize with `python scripts/materialize_configs.py`.

Output path: `experiments/deadline_comm/configs/review_identity.yaml`.

SHA256: `1d4659a0db322ac57849b37f5b9552a21599ca04874a884e621fe851e71cae7d`

```yaml
output: experiments/deadline_comm/runs/review_identity_20261119_001
checkpoint: experiments/deadline_comm/runs/reviewer_ten_seeds_20261118_001/scratch_slow_5ms_seed{seed}_full/checkpoint_1500.pt
seeds: [0,1,2,3,4,5,6,7,8,9]
conditions: [[slow,5.0],[medium,2.0],[medium,5.0]]
methods: [full,periodic,rr_cap,dac]
decays: [1.0,0.9]
episodes: 20
reset_start: 11200000
wall_hours: 4
```
