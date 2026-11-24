# q2_priority.yaml

Original experiment configuration. Materialize with `python scripts/materialize_configs.py`.

Output path: `experiments/deadline_comm/configs/q2_priority.yaml`.

SHA256: `093c2eeb7e53f6ce6e781a70b82d4bc3d47f0719194c3bc2e1656196f21272f7`

```yaml
checkpoints:
  adapted2k: experiments/deadline_comm/runs/day2_followup_20261118_001/seed{seed}/checkpoint_0500.pt
conditions:
- - slow
  - 5.0
- - medium
  - 2.0
- - medium
  - 5.0
episodes: 30
links:
  medium: 100000
  slow: 20000
methods:
- rr_cap
- dac
modes:
- measured
- fixed_ready
output: experiments/deadline_comm/runs/q2_priority_20261118_001
seeds:
- 0
- 1
- 2
wall_hours: 4
```
