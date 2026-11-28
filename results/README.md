# Recorded results and analysis inputs

This directory contains existing experiment summaries, paired traces, training logs and validation outcomes. [manifest.json](manifest.json) maps packaged files to their original paths and records sizes and SHA-256 hashes. Generated NumPy inputs are stored in [data](data/README.md). Model tensors are stored in [checkpoints](../checkpoints/README.md).

| Directory | Contents |
| --- | --- |
| `runs/` | Hallway, broadcast/XOR, timing, receiver interventions and SMACv2 observations |
| `tmlr_revision/` | Frozen-policy sender/history comparisons and supporting inputs |
| `external_biases/` | MNIST-sum learning results |
| `external_deadline/` | MNIST-sum deadline comparisons and saved probabilities |
| `evidence/` | Compressed paired observations and input manifests |

Run `python experiments/prepare_inputs.py` to verify and copy files to their original input locations. Packaged observations stay here; recomputation uses separate working copies and temporary outputs. Use a new directory for new observations.

[The findings](../experiments/docs/RESULTS.md) explain estimates, uncertainty and failed checks. [Setup](../experiments/docs/REPRODUCE.md) describes which statistics can be regenerated and which evaluations require separate datasets or simulators.

The table verifier reads the printed values from the two archived LaTeX sources in [table_sources](table_sources).

## JAAMAS deployment robustness

`runs/mnist_uncertain_readiness_20261126_001/` stores the exact uncertainty/cost summaries; `runs/broadcast_hard_deadline_20261126_001/` contains three process runs,18,000episode traces,360,000steps, summaries and independent validation. The manifest now includes5,741assets; original observations are preserved. New NPZ inputs reside under `results/data/runs/`. [Interpretation](../experiments/deadline_comm/deployment_robustness/results/RESULTS.md).
