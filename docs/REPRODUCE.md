# Setup and execution

## Install and prepare

The code uses Python, PyTorch, NumPy, PyYAML, SciPy and Matplotlib; pytest runs the tests. The recorded work used Python 3.13.12, PyTorch 2.11.0+cu128 and NumPy 2.4.4. Verification runs on CPU.

```bash
python -m pip install -r requirements.txt
python experiments/prepare_inputs.py
```

The repository includes 375 checkpoints and 103 NumPy input archives in `data/`, together with recorded summaries and traces. [The manifest](../results/manifest.json) lists each packaged file's original path, size and SHA-256. Preparation verifies every entry and creates independent copies at the paths used by the original modules. It refuses to overwrite a different existing file. Allow roughly another 0.8 GB for these working copies. To check files without copying, use `python experiments/prepare_inputs.py --check-only`.

Original YAML configurations are in [the configuration catalogue](CONFIGURATION.md). They preserve seeds, budgets, checkpoint paths and controls.

## Verify the recorded analysis

```bash
python experiments/verify_results.py --recompute --tests
```

This verifies input hashes, loads all checkpoint files on CPU, recomputes sender/history statistics and receiver probes, checks receiver intervention logs, compares saved MNIST-sum probabilities and runs a small frozen-policy replay. It also runs the tests in `tests/`. Regenerated comparisons use temporary directories. No training campaign is launched.

Individual checks:

```bash
python experiments/inspect_checkpoints.py
python experiments/verify_statistics.py
python experiments/verify_models.py --probes --smoke
python -m pytest -q -p no:cacheprovider tests
```

Run preparation before the statistics and model checks. `inspect_checkpoints.py` checks tensor deserialization for every checkpoint; the other checks load the relevant policy architectures and recompute the stated comparisons. Loading a checkpoint is not itself a new evaluation of its performance.

## MNIST inputs and full external-game inference

Raw MNIST images are obtained separately. The downloader validates the four expected gzip checksums:

```bash
python experiments/fetch_mnist.py
python experiments/verify_results.py --external-data experiments/deadline_comm/external_biases/data
```

See `python experiments/fetch_mnist.py --help` for a custom destination. The inference command uses saved models and original test inputs to regenerate external-game results without training. Default verification already checks saved probabilities and labels included in `data/`.

The six common 1,500-update continuation states and six selected 12,000-update states are included. `experiments/reproduce_external_continuation.py` prints continuation commands by default; its `--execute` option starts training and is unnecessary for verification.

## Other experiments

See [the experiment map](EXPERIMENTS.md) for entry points and controls. Copy a YAML and select a new output directory before a new experiment; preparation populates original paths with completed observations. CPU timings depend on hardware and process conditions.

SMACv2 requires a separate StarCraft II installation, maps and simulator dependencies. Files include the earlier failed learning checks, later learning milestone and failed independent replay, and technical-smoke observations. CPU verification does not establish a successful SMACv2 deployment comparison.

## Validation of this edition

On 2026-11-24, a separate copy containing only the public repository files passed:

- SHA-256 and size checks for all 5,703 packaged assets; a second preparation preserved the existing identical working copies.
- CPU deserialization of all 375 checkpoints; architecture-specific loading of the 280 Hallway/broadcast models and 12 MNIST-sum states.
- Recalculation of the 2,654 sampled-state comparisons, 800 prefix episodes, 100 receiver-probe cells and structural summaries covering 2,400 paired episodes.
- Receiver intervention accounting for 1,200 episodes, external-game checks from saved arrays, and the frozen-policy replay smoke.
- **56 tests passed, 1 skipped**. The skip requires the separate early `smoke_eval_003` artifact set, which is not bundled.
- Full external-game inference regenerated from saved models and the separately supplied, checksum-verified MNIST inputs.
- LaTeX build produced the 13-page main paper and 9-page supplement.

No new training, GPU computation or StarCraft II run was used. These checks validate the included comparisons; they do not rerun every training history or reproduce the original hardware timings.

## November 25 layout and paper update

The 103 generated NumPy archives now live in `data/`; `prepare_inputs.py` resolves the updated manifest without changing scientific bytes. Validation completed: 5,703 asset hashes, 375 CPU checkpoint loads, 100 probe cells, 2,654 sender states, 800 shared prefixes, 2,400 structural pairs and 1,200 receiver episodes; saved external-game probability checks match. CPU tests: 56 passed, 1 skip for the unbundled early integration-smoke artifacts. The paper rebuilds to 14 main and 10 supplementary pages. No new training or GPU/simulator experiment ran.
