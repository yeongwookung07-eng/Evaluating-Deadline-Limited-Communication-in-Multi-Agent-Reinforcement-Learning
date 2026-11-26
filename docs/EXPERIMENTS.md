# Experiment map

All entry points below are relative to `experiments.deadline_comm`. Settings are in [the configuration catalogue](CONFIGURATION.md). The checkpoint and result folders contain the recorded inputs. Run `python experiments/prepare_inputs.py` once to restore the paths expected by the modules.

| Question | Python module | Configuration | Control | Outputs / prerequisites |
| --- | --- | --- | --- | --- |
| Learned Hallway reference | `reviewer_validation.train_ten_seeds` | [reviewer_ten_seeds](../experiments/deadline_comm/configs/reviewer_ten_seeds.yaml) | Fresh weights, Oracle demonstrations, equal update budgets; compare channel access and access rules. | Primary Hallway checkpoints and evaluation seed summaries; recorded checkpoint files are included. |
| Oracle-free broadcast and failed XOR reference | `reviewer_validation.train_hidden` | [hidden_mechanism](../experiments/deadline_comm/configs/hidden_mechanism.yaml) | Train the hidden-source reference and retain learning-gate outcomes. | Broadcast/XOR checkpoints and learning-gate summaries. |
| Full/RR delivery identity | `review_revision.identity` | [review_identity](../experiments/deadline_comm/configs/review_identity.yaml) | Match Full checkpoints, reset seeds, cache decay and fixed readiness. | Matched sender/cache/action/reward histories. |
| Same-count sender choices and fresh-update withholding | `tmlr_revision.experiments` | [tmlr_fresh](../experiments/deadline_comm/configs/tmlr_fresh.yaml) | Enumerate feasible one-decision substitutions; separately remove each factual update, then resume the original scheduler. | Sampled-state JSONL with suffix return and action comparisons; use `--kind fresh`. |
| Common history | `tmlr_revision.experiments` | [tmlr_prefix](../experiments/deadline_comm/configs/tmlr_prefix.yaml) | Run the same RR prefix before branching into RR and DAC; retain early termination. | Cold/shared-prefix episode comparisons; use `--kind prefix`. |
| State/episode weighting | `tmlr_revision.robustness.strata` | [tmlr_hallway_sensitivity](../experiments/deadline_comm/configs/tmlr_hallway_sensitivity.yaml) | Pair alternatives at the same factual state; report equal-state and equal-episode estimates. | Stratified, seed-level estimates and uncertainty. |
| Preparation and deadline boundary | `tmlr_revision.readiness` | [tmlr_readiness](../experiments/deadline_comm/configs/tmlr_readiness.yaml) | Compare measured and fixed readiness on the same link and policies. | CPU time, usable packet counts, misses and control outcomes. |
| Independent CPU repeats | `tmlr_revision.robustness.cpu_repeat` | [tmlr_cpu_repeats](../experiments/deadline_comm/configs/tmlr_cpu_repeats.yaml) | Repeat on one pinned CPU and compare selectors on identical input snapshots. | Process-specific readiness and selector-overhead summaries. |
| Received information and probes | `tmlr_revision.robustness.aggregate_probe` | [tmlr_receiver_probe](../experiments/deadline_comm/configs/tmlr_receiver_probe.yaml) | Fit decoders on disjoint receiver-input episodes. | Aggregate/input decoding and unchanged-policy accuracy. |
| Structural identity and shared-history summary | `attribution_boundaries.structural_history_audit` | [attribution_structural_history_v2](../experiments/deadline_comm/configs/attribution_structural_history_v2.yaml) | Reaggregate existing delivery, prefix and timing traces. | Counts, paired effects, remaining reward and timing-boundary summaries. |
| Receiver-memory removal | `attribution_boundaries.receiver_cache_reset_interaction` | [receiver_cache_reset_interaction](../experiments/deadline_comm/configs/receiver_cache_reset_interaction.yaml) | Cross retained/cleared cache with RR/DAC after the same prefix. | Action-history changes and suffix returns for four branches. |
| Acting receiver adaptation | `receiver_adaptation.evaluate` | [receiver_head_adaptation](../experiments/deadline_comm/configs/receiver_head_adaptation.yaml) | Fit the follower head with true versus shuffled labels; evaluate fresh paired rollouts. | Actual accuracy/return, access histories and deadline misses. |
| Learned MNIST-sum communication | `external_biases.mnist_sum_learning` | [external_biases_mnist](../experiments/deadline_comm/configs/external_biases_mnist.yaml) | Reconstruct published signaling/listening objectives; retain validation milestones and all seeds. | Speaker/listener checkpoints and held-out accuracy; separate parallel continuation settings are retained. |
| Equal-cost MNIST admission | `external_deadline.mnist_deadline_attribution` | [external_deadline_mnist_v2](../experiments/deadline_comm/configs/external_deadline_mnist_v2.yaml) | Freeze passing policies and compare informed/random admission at equal offered cost over five cutoffs. | Expected accuracy, message-use measures and delivered fractions. |
| SMACv2 feasibility and replay | `smac_protocol.train / smac_protocol.evaluate` | [smacv2_protocol](../experiments/deadline_comm/configs/smacv2_protocol.yaml) | Preserve the learning gate and independent-engine replay requirement. | Training milestones and replay diagnostics; external SC2/runtime/maps are required. |

## Recompute the recorded comparisons

```bash
python experiments/prepare_inputs.py
python experiments/verify_results.py --recompute --tests
```

The verification command checks the packaged files, loads checkpoints, recomputes sender/history and receiver statistics, compares saved external-game probabilities and runs a small frozen-policy replay in temporary directories. It does not train policies. [Detailed commands and dependencies](REPRODUCE.md).

For a new experiment, copy its YAML and choose a new output directory. The recorded output paths already contain the completed observations after preparation. The original experiment implementations remain under their existing module names; shared GRU, replay and TD infrastructure is under `models/shared/`.

## JAAMAS extension — 2026-11-26

[Uncertain readiness and causal late-action rejection](../experiments/deadline_comm/deployment_robustness/README.md): frozen MNIST-sum3seeds and broadcast10seeds, existing adapted/shuffled heads. Entry points `deployment_robustness.mnist_uncertain_readiness` / `broadcast_hard_deadline`; settings `mnist_uncertain_readiness.yaml` / `broadcast_hard_deadline.yaml`.450modeled timing cells and18,000fresh broadcast episodes. [Complete results](../experiments/deadline_comm/deployment_robustness/results/RESULTS.md).
