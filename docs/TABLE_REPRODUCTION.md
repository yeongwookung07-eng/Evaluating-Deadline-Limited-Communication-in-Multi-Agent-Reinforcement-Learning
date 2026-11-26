# From a clean install to the paper tables

Use the isolated CPU installation commands in [REPRODUCE.md](REPRODUCE.md). Then run, from the repository root:

```bash
python experiments/verify_results.py --recompute --tests
python experiments/reproduce_paper_tables.py
```

Both commands must exit with status 0. The first verifies the packaged scientific inputs and retained analyses. The second recomputes the selected table rows below, writes CSVs and checks the formatted rows against the actual editable LaTeX manuscript. It also verifies the primary hard-hold deadline miss rates in the main text. Values are calculated from arrays and actions, not copied from the paper or result-summary JSON. Decimal half-way counts use half-up display rounding; CSV numeric columns retain unrounded values.

| Paper table (stable LaTeX label) | Exact export coverage | Raw inputs after preparation | CSV under `results/local_recompute/paper_tables/` |
|---|---|---|---|
| Main: `tab:external_deadlines` | All five reception-deadline rows; offered bytes, expected accuracy, paired gain and CI | `experiments/deadline_comm/external_deadline/runs/mnist_deadline_20261123_002/seed*_episodes.npz`; speaker/listener probabilities, labels, readiness, saved random masks | `external_deadlines.csv` |
| Main: `tab:receiver_controls` | Six acting-policy rows; accuracy and deadline miss fraction. The two probe rows are not exported | `experiments/deadline_comm/runs/receiver_head_adaptation_20261123_001/episodes.jsonl`; 1,200 episodes, actual actions, bits and measured timing flags | `receiver_controls_acting_rows.csv` |
| Supplement S7: `tab:uncertain_readiness` | All four selected error/cost rows; expected offered bytes and paired gain/CI | `experiments/deadline_comm/runs/mnist_uncertain_readiness_20261126_001/seed*_cells.npz`; independently integrate Gaussian errors using `erfc` | `uncertain_readiness.csv` |
| Supplement S7: `tab:hard_action` | All seven printed rows; original/adapted accuracy and gain/CI. Hold and zero at 500 µs are separately recomputed, giving eight CSV rows | `experiments/deadline_comm/runs/broadcast_hard_deadline_20261126_001/repeat*/episodes.jsonl.gz`; actual actions, bits and CPU/wait/post costs | `hard_action.csv` |

Expected checkpoints for an immediate comparison:

| Condition | Reproduced result |
|---|---|
| Original MNIST deadline 750 µs | Full/informed 29.37%; random 18.78%; +10.59 pp [7.25, 13.92] |
| Noisy MNIST: SD100 µs, cost100 µs, deadline750 µs | +9.25 pp [6.64, 11.85]; 8.276 offered B |
| Normal broadcast: hold, no extra cost | 57.69% → 63.31%; +5.63 pp [1.80, 9.46] |
| Same primary condition: candidate deadline misses | 0.100% original, 0.058% adapted |
| Normal broadcast: hold, extra250 µs | +5.45 pp [1.99, 8.92] |
| Normal broadcast: hold/zero, extra500 µs | 47.75% for both heads; gain 0 |

`verification.json` records the source-aware failure control and manuscript hashes. Processes are averaged within each training seed; three CPU repetitions do not become 30 independent seeds. Intervals use the paper's seed-level paired Student t calculation. A successful export covers 22 distinct printed rows across four tables, not every table in the manuscript. Other sender/history/probe comparisons are checked by the first command; its detailed scope is in [REPRODUCE.md](REPRODUCE.md).

This reproduces numerical analysis of the saved observations. It does not retrain policies, rerun all environment trajectories or reproduce hardware timings on a new computer. The earlier small frozen replay is part of `verify_results.py`; full MNIST inference requires separately downloaded images, as documented in REPRODUCE.md. Rerunning the measured hard-fallback experiment can change deadline miss rates with host load and therefore need not reproduce its exact recorded percentages. Its separate evaluator, YAML, fixed checkpoints and bounded CPU protocol are in the [component README](../experiments/deadline_comm/deployment_robustness/README.md).

Clean-install evidence and logs: [validation record](validation_clean_cpu_20261126.json), [verification log](validation_clean_cpu_20261126/verification.log), [table export log](validation_clean_cpu_20261126/paper_tables.log). The initial public-source audit and the revised table exporter are distinguished in that record.
