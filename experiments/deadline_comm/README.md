# Deadline-limited communication experiments

This component implements the controlled communication-evaluation study described in the root README. Its research question is whether traffic savings and team return identify message-selection benefit when delivery deadlines, cached reception and receiver use can intervene.

The shared GRU/message/utility policy and QMIX learner provide the learned reference. The transport and cache expose offered attempts, received payloads, sender IDs, timestamps and readiness. Frozen-policy branches hold checkpoints and full pre-decision state fixed while changing a sender set, withholding an update, switching access after a common prefix or clearing receiver memory.

- [Experiment questions, controls, entry points and required inputs](../../docs/EXPERIMENTS.md)
- [Settings, seed ranges and bounded budgets](../../docs/configs/README.md)
- [Completed findings, negative results and scope](../../docs/RESULTS.md)
- [Execution and source checks](../../docs/REPRODUCE.md)

Learning, evaluation and analysis entry points produce their own local model, JSON/CSV/JSONL, plot and report outputs. This source edition does not contain those generated artifacts. Hallway is Oracle-assisted; hidden-source broadcast and the external MNIST-sum reference are distinguished in the research description. Timing is one-host measured CPU plus modeled serial waiting, not a physical-network benchmark.

The original scientific implementations and historical module names are retained. The only extracted utility is the unchanged result-table formatter in `tmlr_revision/editorial.py`; manuscript rewrite functions are omitted. Current status: completed experimental implementations, with source-release checks recorded in the setup document. No experiment campaign is launched by preparing this repository.
