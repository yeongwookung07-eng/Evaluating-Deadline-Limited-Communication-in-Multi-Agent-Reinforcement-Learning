# Experiment map

All entry points below are relative to `experiments.deadline_comm`. Settings are preserved as Markdown in [the configuration catalogue](configs/README.md); materialize them before calling the original modules. Entry points read or produce local checkpoints and traces. Those binary/raw inputs are not distributed in this source edition.

| Question | Python module | Configuration | Control | Outputs / prerequisites |
| --- | --- | --- | --- | --- |
| Learned Hallway reference | `reviewer_validation.train_ten_seeds` | [reviewer_ten_seeds](configs/reviewer_ten_seeds.md) | Fresh weights, Oracle demonstrations, equal update budgets; compare channel access and access rules. | Primary Hallway checkpoints and evaluation seed summaries; checkpoint files are generated locally. |
| Oracle-free broadcast and failed XOR reference | `reviewer_validation.train_hidden` | [hidden_mechanism](configs/hidden_mechanism.md) | Train the hidden-source reference and retain learning-gate outcomes. | Broadcast/XOR checkpoints and learning-gate summaries. |
| Full/RR delivery identity | `review_revision.identity` | [review_identity](configs/review_identity.md) | Match Full checkpoints, reset seeds, cache decay and fixed readiness. | Matched sender/cache/action/reward histories. |
| Same-count sender choices and fresh-update withholding | `tmlr_revision.experiments` | [tmlr_fresh](configs/tmlr_fresh.md) | Enumerate feasible one-decision substitutions; separately remove each factual update, then resume the original scheduler. | Sampled-state JSONL with suffix return and action comparisons; use `--kind fresh`. |
| Common history | `tmlr_revision.experiments` | [tmlr_prefix](configs/tmlr_prefix.md) | Run the same RR prefix before branching into RR and DAC; retain early termination. | Cold/shared-prefix episode comparisons; use `--kind prefix`. |
| State/episode weighting | `tmlr_revision.robustness.strata` | [tmlr_hallway_sensitivity](configs/tmlr_hallway_sensitivity.md) | Pair alternatives at the same factual state; report equal-state and equal-episode estimates. | Stratified, seed-level estimates and uncertainty. |
| Preparation and deadline boundary | `tmlr_revision.readiness` | [tmlr_readiness](configs/tmlr_readiness.md) | Compare measured and fixed readiness on the same link and policies. | CPU time, usable packet counts, misses and control outcomes. |
| Independent CPU repeats | `tmlr_revision.robustness.cpu_repeat` | [tmlr_cpu_repeats](configs/tmlr_cpu_repeats.md) | Repeat on one pinned CPU and compare selectors on identical input snapshots. | Process-specific readiness and selector-overhead summaries. |
| Received information and probes | `tmlr_revision.robustness.aggregate_probe` | [tmlr_receiver_probe](configs/tmlr_receiver_probe.md) | Fit decoders on disjoint receiver-input episodes. | Aggregate/input decoding and unchanged-policy accuracy. |
| Structural identity and shared-history summary | `attribution_boundaries.structural_history_audit` | [attribution_structural_history_v2](configs/attribution_structural_history_v2.md) | Reaggregate existing delivery, prefix and timing traces. | Counts, paired effects, remaining reward and timing-boundary summaries. |
| Receiver-memory removal | `attribution_boundaries.receiver_cache_reset_interaction` | [receiver_cache_reset_interaction](configs/receiver_cache_reset_interaction.md) | Cross retained/cleared cache with RR/DAC after the same prefix. | Action-history changes and suffix returns for four branches. |
| Acting receiver adaptation | `receiver_adaptation.evaluate` | [receiver_head_adaptation](configs/receiver_head_adaptation.md) | Fit the follower head with true versus shuffled labels; evaluate fresh paired rollouts. | Actual accuracy/return, access histories and deadline misses. |
| Learned MNIST-sum communication | `external_biases.mnist_sum_learning` | [external_biases_mnist](configs/external_biases_mnist.md) | Reconstruct published signaling/listening objectives; retain validation milestones and all seeds. | Speaker/listener checkpoints and held-out accuracy; separate parallel continuation settings are retained. |
| Equal-cost MNIST admission | `external_deadline.mnist_deadline_attribution` | [external_deadline_mnist_v2](configs/external_deadline_mnist_v2.md) | Freeze passing policies and compare informed/random admission at equal offered cost over five cutoffs. | Expected accuracy, message-use measures and delivered fractions. |
| SMACv2 feasibility and replay | `smac_protocol.train / smac_protocol.evaluate` | [smacv2_protocol](configs/smacv2_protocol.md) | Preserve the learning gate and independent-engine replay requirement. | Training milestones and replay diagnostics; external SC2/runtime/maps are required. |

## Commands with separately supplied inputs

From the repository root, first materialize the documented settings:

```bash
python scripts/materialize_configs.py
```

The following example enumerates one-decision sender alternatives. It requires the exact primary checkpoint bank at the paths in the configuration. Use a new output directory for a new run.

```bash
python -m experiments.deadline_comm.tmlr_revision.experiments --kind fresh --config experiments/deadline_comm/configs/tmlr_fresh.yaml
```

The receiver intervention similarly requires the retained broadcast checkpoints and their learning-gate summary:

```bash
python -m experiments.deadline_comm.receiver_adaptation.evaluate --config experiments/deadline_comm/configs/receiver_head_adaptation.yaml
```

Read each module's command-line parser before running it: some analysis tools retain a fixed configuration path. The paths, seeds and bounded budgets in the configuration catalogue document the completed studies. Training and SMAC campaigns are separate work and are not launched by source checks.

## Dependencies retained from earlier modules

`day2/` supplies the decayed policy, selector, model loader, accounting and log utilities imported by the final studies. `selection_validation/` supplies the fixed-state selection harness. `framework/` and `models/qmix/` supply the recurrent agent, episode representation, TD learner and monotonic mixer. These are execution dependencies of the current analysis, not additional claimed contributions.

## Export changes

Scientific experiment implementations are copied from the research checkout. Required shared dependencies use the already prepared public subset. The `table()` function in `tmlr_revision/editorial.py` is extracted unchanged for result formatting; manuscript rewriting routines are omitted. The continuation helper's docstring now identifies its checkpoints as separately supplied; its execution logic is unchanged. Settings are stored verbatim in Markdown and restored byte-for-byte by the materializer. This changes packaging, not experiment semantics.
