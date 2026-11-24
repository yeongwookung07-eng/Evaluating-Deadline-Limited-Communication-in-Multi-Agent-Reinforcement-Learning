# tmlr_editorial_controls.yaml

Original experiment configuration. Materialize with `python scripts/materialize_configs.py`.

Output path: `experiments/deadline_comm/configs/tmlr_editorial_controls.yaml`.

SHA256: `c375e26acae88003991eea4ad6e84928ae0beb695378876c8ed017af86b6c090`

```yaml
output: experiments/deadline_comm/runs/tmlr_editorial_controls_20261119_001
seeds: [0,1,2,3,4,5,6,7,8,9]
episodes: 50
reset_base: 8200000
max_steps: 7000
scope: Uniform independent random actions; no training or communication. Same reset set as primary evaluation. RNG seeds are not training seeds.
```
