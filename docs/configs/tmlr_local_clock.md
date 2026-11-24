# tmlr_local_clock.yaml

Original experiment configuration. Materialize with `python scripts/materialize_configs.py`.

Output path: `experiments/deadline_comm/configs/tmlr_local_clock.yaml`.

SHA256: `f3291d3f25b189f14d7b8b59ce9685377b9748d565c7f678fd487d2527a5623d`

```yaml
output: experiments/deadline_comm/runs/tmlr_local_clock_20261119_001
checkpoint_metadata: experiments/deadline_comm/runs/reviewer_ten_seeds_20261118_001/scratch_slow_5ms_seed0_full/checkpoint_1500.pt
max_initial_states: 1024
max_steps: 14336
contract: Each agent uses only own distance/active flag, agent ID, reset-synchronized timestep and public fixed maximum initial distance/group roster. No other-agent observations, reward feedback or messages.
falsification: Any reset failing return20 refutes the proposed constructive all-reset no-communication solution.
```
