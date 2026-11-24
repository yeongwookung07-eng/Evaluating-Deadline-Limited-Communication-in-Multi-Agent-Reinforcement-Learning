# tmlr_cost_check.yaml

Original experiment configuration. Materialize with `python scripts/materialize_configs.py`.

Output path: `experiments/deadline_comm/configs/tmlr_cost_check.yaml`.

SHA256: `6b494dcbfe189b17c31f9fd86e9ca1ce6d597b3a4cad6a6c91b391cfabdb681b`

```yaml
output: experiments/deadline_comm/runs/tmlr_cost_check_20261119_001
episode: 0
max_steps: 6000
wall_seconds: 300
scope: Replay first saved episode of each of 300 broadcast cells; verify observed accuracy, delivered sets and attempted wire accounting. No new training or claims from additional resets.
```
