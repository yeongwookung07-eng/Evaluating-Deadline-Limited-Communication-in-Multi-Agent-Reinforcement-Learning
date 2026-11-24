# attribution_structural_history_v2.yaml

Original experiment configuration. Materialize with `python scripts/materialize_configs.py`.

Output path: `experiments/deadline_comm/configs/attribution_structural_history_v2.yaml`.

SHA256: `bf3e6b5387601331d1c6c2d5ed56c979e084253925c703e31c9383ddcf258378`

```yaml
question: Identify structural delivery-identity boundaries and joint-history effects in retained traces.
identity_run: experiments/deadline_comm/runs/review_identity_20261119_001
prefix_run: experiments/deadline_comm/runs/tmlr_prefix_20261119_001
readiness_run: experiments/deadline_comm/runs/tmlr_readiness_20261119_001
repeat_run: experiments/deadline_comm/runs/tmlr_cpu_repeats_20261119_001
output: experiments/deadline_comm/runs/attribution_structural_history_20261123_002
seeds: [0,1,2,3,4,5,6,7,8,9]
budget_seconds: 120
primary_estimand: average episodes within each seed, then equally weight seeds; repeat processes kept separate
secondary_estimand: average steps within each seed, then equally weight seeds
hypothesis: Exact Full-RR identity is conditional on order-aligned delivery, not a generally equivalent selection rule.
strongest_control: frozen matched Full checkpoint, reset, cache rule and readiness; common-prefix branch starts from shared full state
competing_explanation: output identity follows queue order and one-packet capacity by construction; shared-prefix attenuation reflects joint history, not cache alone
falsification: mismatched order or counts invalidates structural identity prediction; any recorded trace disagreement defeats exact identity
adoption: qualify identity as sufficient-condition/implementation result; report boundary failures without treating branch equality as universal equivalence
status: registered_before_execution
refinement: First run preserved; add per-step offered savings to distinguish episode-length effects and first sender mismatch timing.
```
