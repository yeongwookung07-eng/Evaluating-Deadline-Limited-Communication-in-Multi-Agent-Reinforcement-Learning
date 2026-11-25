# Recorded results and analysis inputs

This directory contains existing experiment summaries, paired traces, training logs and validation outcomes. [manifest.json](manifest.json) maps packaged files to their original paths and records sizes and SHA-256 hashes. Generated NumPy inputs are stored in [data](../data/README.md). Model tensors are stored in [checkpoints](../checkpoints/README.md).

| Directory | Contents |
| --- | --- |
| `runs/` | Hallway, broadcast/XOR, timing, receiver interventions and SMACv2 observations |
| `tmlr_revision/` | Frozen-policy sender/history comparisons and supporting inputs |
| `external_biases/` | MNIST-sum learning results |
| `external_deadline/` | MNIST-sum deadline comparisons and saved probabilities |
| `evidence/` | Compressed paired observations and input manifests |

Run `python experiments/prepare_inputs.py` to verify and copy files to their original input locations. Packaged observations stay here; recomputation uses separate working copies and temporary outputs. Use a new directory for new observations.

[The findings](../docs/RESULTS.md) explain estimates, uncertainty and failed checks. [Setup](../docs/REPRODUCE.md) describes which statistics can be regenerated and which evaluations require separate datasets or simulators.
