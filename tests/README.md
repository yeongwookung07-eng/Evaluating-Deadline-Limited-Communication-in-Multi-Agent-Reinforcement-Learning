# Implementation tests

Run from the repository root:

```bash
python -m pytest -q -p no:cacheprovider tests
```

Tests cover packet deadlines, cache updates, recurrent replay, batched learning, Oracle reference behavior, sender interventions, receiver resets, receiver heads, external-game objectives and SMACv2 protocol logic. CPU fixtures do not start a full training campaign or simulator.

Some checks inspect saved artifacts when present. Run `python experiments/prepare_inputs.py` to restore included results. The separate `experiments/verify_results.py --recompute` command checks recorded scientific comparisons. A unit-test pass alone does not reproduce an experiment's measured effect.

JAAMAS deployment extension: `deployment_robustness/test_contracts.py` verifies exact readiness/no-cost limits, noise/cost boundaries, a cost-induced failure case, hard fallback selection, executed-action recurrent feedback and no-miss full-rollout identity. New saved-outcome audits are also run by `experiments/verify_results.py --recompute`.
