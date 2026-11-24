# tmlr_cpu_repeats.yaml

Original experiment configuration. Materialize with `python scripts/materialize_configs.py`.

Output path: `experiments/deadline_comm/configs/tmlr_cpu_repeats.yaml`.

SHA256: `26c375846d5ee4c453787e14d80973eab7333846d210d92b3ac2822a60e0873f`

```yaml
output: experiments/deadline_comm/runs/tmlr_cpu_repeats_20261119_001
repeats: 3
seeds: [0,1,2,3,4,5,6,7,8,9]
checkpoints: [rr_cap,dac]
deadlines_ms: [2.0,2.25,2.5]
episodes: 3
reset_base: 14300000
wall_hours: 1
snapshot_steps: [0,2,4]
scope: Sequential independent processes on one common allowed CPU. Paired rollouts plus same-state selector timing snapshots. No simultaneous own diagnostic workloads; no claim of exclusive-host or second-machine timing. Repeats are not additional training seeds.
```
