# Saved checkpoints

The 375 `.pt` files retain trained weights and, where recorded, optimizer and random-generator state. [The manifest](../results/manifest.json) records hashes and original paths.

| Study | Files | Use |
| --- | ---: | --- |
| Hallway | 180 | Independently trained references and matched-policy comparisons |
| Hidden-source broadcast/XOR | 100 | Learning and receiver-input analyses, including failed XOR checks |
| Receiver-head adaptation | 46 | True-label and shuffled-label interventions, including smoke runs |
| MNIST-sum | 12 | Six common 1,500-update states and six selected 12,000-update states |
| SMACv2 | 37 | Earlier pilot, later learning/replay protocol and technical smoke |

Run `python experiments/inspect_checkpoints.py` to load every file on CPU. Run `python experiments/prepare_inputs.py` before modules that expect original paths. Architecture-specific checks and commands are in [setup](../experiments/docs/REPRODUCE.md).

These models are inputs to recorded comparisons. Failed XOR and SMACv2 checks remain failures; checkpoint availability does not establish successful learning or independent replay. See [results and limits](../experiments/docs/RESULTS.md).

JAAMAS deployment follow-up reuses these saved policies and already fitted receiver heads unchanged. No new checkpoint or training is introduced. See [the frozen-policy protocol](../experiments/deadline_comm/deployment_robustness/README.md).
