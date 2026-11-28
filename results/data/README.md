# Generated analysis data

This directory contains 103 original NumPy archives (57,167,165 bytes): receiver inputs and labels, paired observations and MNIST-sum evaluation probabilities used by the published analyses. All files are copied from the retained experiments without changing their bytes. They are generated analysis inputs, not the raw MNIST image dataset.

The [asset manifest](../manifest.json) records sizes, SHA-256 hashes and original experiment paths. Run `python experiments/prepare_inputs.py` from the repository root to verify and restore inputs; `--check-only` verifies without copying. [Results](../README.md) contains summaries and traces, while [checkpoints](../../checkpoints/README.md) contains model states. See [reproduction instructions](../../experiments/docs/REPRODUCE.md) for analysis commands and the checksum-verifying MNIST downloader.

Scope and limitations: these are the studied checkpoints and reset distributions; they do not establish standard-benchmark generalization. No new experiment or training accompanies this organization.

## November26 extension

Three additional `seed*_cells.npz` files under `runs/mnist_uncertain_readiness_20261126_001/` retain exact offer probabilities, per-example rewards and timing for450settings. They extend the existing103NPZ assets without changing them. See the manifest and deployment-robustness protocol.
