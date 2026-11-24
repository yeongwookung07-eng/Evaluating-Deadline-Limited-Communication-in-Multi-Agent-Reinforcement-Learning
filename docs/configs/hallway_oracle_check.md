# hallway_oracle_check.yaml

Original experiment configuration. Materialize with `python scripts/materialize_configs.py`.

Output path: `experiments/deadline_comm/configs/hallway_oracle_check.yaml`.

SHA256: `c0c5e7bb6059ab04f5bc239a512ab4f5993695c96ab1e40dde8dbe9993bb7427`

```yaml
question: Does the unchanged Hallway wrapper return positive rewards for a valid oracle?
hypothesis: A centralized oracle synchronizing within groups and staggering groups obtains Return 20.
strongest_control: Same initial positions; direct upstream-class step versus wrapper step parity.
competing_explanation: Straight-line motion fails due to unequal initial distances or simultaneous group completion, not reward suppression.
falsification: Any valid oracle failure, unexpected reward, or wrapper/upstream transition mismatch fails the check.
environment:
  n_agents: 5
  state_numbers: [4, 4, 4, 4, 4]
  group_ids: [0, 0, 1, 1, 1]
seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 41000, 41001, 41002, 41003, 41004, 41005, 41006, 41007, 41008, 41009, 41010, 41011, 41012, 41013, 41014, 41015, 41016, 41017, 41018, 41019]
exhaustive_initial_positions: true
policies: [straight, synchronized_staggered, wait]
budget:
  max_episodes: 3162
  max_wrapper_steps: 44268
  device: cpu
output_name: hallway_oracle_check_001
```
