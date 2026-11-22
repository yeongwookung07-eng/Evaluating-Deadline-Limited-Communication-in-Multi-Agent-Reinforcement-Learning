# Paper evidence map

All paths below start at `experiments/deadline_comm/`. Small per-seed and aggregate JSON/CSV are in Git at their original paths; detailed traces and model checkpoints are omitted and remain in the original research workspace. Selection retains every seed within the declared cohorts, including unsuccessful controls.

| Claim/control | Code | Evidence under `runs/` |
| --- | --- | --- |
| Full–RR delivery identity | `review_revision/identity.py` | `review_identity_20261119_001` |
| Same-count sender substitutions | `tmlr_revision/experiments.py` | `tmlr_fresh_20261119_001` |
| Common RR initialization | `tmlr_revision/experiments.py` | `tmlr_prefix_20261119_001`, `tmlr_prefix_opportunity_20261119_001` |
| CPU readiness/cost | `tmlr_revision/readiness.py`, `robustness/cpu_repeat.py` | `tmlr_readiness_20261119_001`, `tmlr_cpu_repeats_20261119_001`, `tmlr_cost_check_20261119_001` |
| Broadcast input probes | `tmlr_revision/robustness/aggregate_probe.py` | `tmlr_broadcast_20261119_001`, `tmlr_receiver_probe_20261119_001` |
| Local-clock zero communication | `tmlr_revision/local_clock.py` | `tmlr_local_clock_20261119_001` |
| XOR/broadcast learning gates | `reviewer_validation/hidden_source.py` | `hidden_mechanism_20261119_001`, `review_oracle_free_20261119_001` |
| Failed SMACv2 learning and technical smoke | `standard_benchmarks/`, `smac_protocol/` | `smacv2_pilot_20261119_001`, `smacv2_protocol_20261121_001`, `smacv2_protocol_20261121_001_smoke` |

Aggregate tables, per-seed estimates and intervals are in `tmlr_revision/results/{summary,editorial_audit,robustness_summary}.json`. State-weighted and episode-weighted results must not be conflated. Fixed readiness and measured CPU timing answer different questions. Equal payload counts do not imply equal sender information. Historical reports may describe earlier manuscript versions; the KBS English manuscript is canonical.
