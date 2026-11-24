# Source-release helpers

`materialize_configs.py` verifies the original configuration bytes stored in the Markdown catalogue and restores their YAML files locally. `--check` only reads and verifies. Existing differing configuration files are never overwritten.

`check_source.py` compiles the retained Python sources, checks local Markdown links and verifies the configuration blocks. With `--tests`, it creates a temporary source copy, restores configurations there and runs the existing CPU tests with GPU visibility disabled. It does not require or claim to provide the original trained models and evaluation logs.

`fetch_mnist.py` downloads the separately distributed MNIST inputs and checks their published checksums when explicitly run. `reproduce_external_continuation.py` previews the original continuation commands; executing them requires separately supplied common checkpoints and an explicit `--run` argument. These tools do not run during the source check.
