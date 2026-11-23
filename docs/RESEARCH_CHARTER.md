# Research charter — Deadline-limited communication evaluation

Active scope, explicitly aligned to the written KBS manuscript by the user on 2026-11-22 KST: **Evaluating Deadline-Limited Communication in Multi-Agent Reinforcement Learning: Delivery, Computation, and Control**.

Canonical source: [English KBS manuscript](../experiments/deadline_comm/publication/kbs/kbs_main.tex). [Current claims](../claim.md) define the supported scope. This replaces the previous minimum-pairwise topic mandate; [the prior charter and ledger](research_history/before_kbs_alignment_20261122/INDEX.md) remain preserved.

## Research question

Under action deadlines and cached reception, which mechanisms explain observed communication-efficiency outcomes: reduced attempted traffic, timely delivery, sender identity, initialization, selector computation or policy use of received information?

## Evaluation contract

1. Validate learned references with declared supervision, budgets, seeds and held-out learning gates. Separate learned communication utility from task-level necessity.
2. Account for attempted wire bytes, delivered payload, sender identities/timestamps, cache histories, CPU readiness, deadlines and return. Compare fixed weights/cache rules/resets with measured and fixed readiness.
3. Isolate sender choice using full-state matched-checkpoint replay, same-count feasible substitutions, factual/uniform controls and common initialization. Verify no-op/forced-original identity. Distinguish seed, state and episode weighting.
4. Check downstream use through held-out probes of the actual receiver input versus the unchanged acting policy. When a decoder suggests a remedy, intervene on the acting head and evaluate actual performance on fresh episodes at matched access.

## Evidence and limits

The paper combines Oracle-assisted five-agent Hallway, Oracle-free hidden-source broadcast and a reconstruction of the published Eccles et al. one-step MNIST-sum game. The default deployment estimand is an equal-episode contrast within each independently trained seed. Preserve the order-dependent 400-pair Hallway identity, weighting-sensitive sender results, public-clock zero-message solution, normal-access receiver intervention and external timing-informed versus random admission comparison. The external case tests modeled reception with known readiness. Preserve the failed XOR and early SMACv2 gates, the later Terran learning pass and failed independent state replay, and the limited SMACv2 technical smoke. Large sequential-benchmark generalization remains open.

## Execution discipline

Register new questions, controls, falsification criteria, seeds, budgets and unique output directories before execution; retain failures and whole-cost accounting. New experiments need separate entry points/configs, nearest README updates and root ledger entries. Preserve frozen foundations and all existing artifacts. This documentation request launches no training or experiment and does not renew historical GPU budgets. Existing experiment-specific authorizations remain bounded; GPU2 is excluded.

Manuscript build/status: [KBS README](../experiments/deadline_comm/publication/kbs/README.md). Experiment source and historical protocols: [deadline_comm](../experiments/deadline_comm/README.md). Submission and author declarations remain distinct from scientific evidence.
