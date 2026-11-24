# tmlr_hallway_sensitivity.yaml

Original experiment configuration. Materialize with `python scripts/materialize_configs.py`.

Output path: `experiments/deadline_comm/configs/tmlr_hallway_sensitivity.yaml`.

SHA256: `6af3b07aa1949d50880e6d423ebba5aca8868c2d852aeb90ec47cff6f6cc4358`

```yaml
output: experiments/deadline_comm/runs/tmlr_hallway_sensitivity_20261119_001
source: experiments/deadline_comm/runs/tmlr_fresh_20261119_001
bootstrap_resamples: 5000
bootstrap_seed: 20261119
wall_seconds: 600
scope: Reconstruct original 400 factual episodes; verify all2654 sampled-state returns and delivered subsets. Stratify before intervention by unresolved groups, time and previous simultaneous-arrival penalty. Compute paired factual-minus-uniform and state/episode weighting without filtering training seeds.
```
