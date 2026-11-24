# Experiment and analysis code

`deadline_comm/` contains policy, message cache, transport, learning and analysis implementations. Original settings are in `deadline_comm/configs/`; the [experiment map](../docs/EXPERIMENTS.md) connects each question to its entry point, controls and outputs.

| Helper | Purpose |
| --- | --- |
| `prepare_inputs.py` | Verify file hashes and restore original input paths |
| `verify_results.py` | Recompute recorded analyses, optionally run tests and external inference |
| `verify_statistics.py` | Reaggregate sender, history and weighting comparisons |
| `verify_models.py` | Load policies, recompute probes and run a small frozen replay |
| `inspect_checkpoints.py` | Load all checkpoint tensors on CPU |
| `fetch_mnist.py` | Fetch and checksum the separate MNIST inputs |
| `reproduce_external_continuation.py` | Preview continuation commands; training requires `--execute` |

Start with [setup](../docs/REPRODUCE.md). Inputs are packaged checkpoints, traces and YAML settings. Outputs are verification messages and temporary regenerated comparisons. [The findings](../docs/RESULTS.md) describe scientific results and limits.
