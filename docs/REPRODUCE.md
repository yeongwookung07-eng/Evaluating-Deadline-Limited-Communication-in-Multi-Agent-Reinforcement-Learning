# Setup and execution

## Dependencies

The source uses Python, PyTorch, NumPy, PyYAML, SciPy and Matplotlib; pytest runs the supplied tests. The original study used Python 3.13.12, PyTorch 2.11.0+cu128 and NumPy 2.4.4. CPU checks do not require CUDA. SMACv2 experiments additionally require their separate simulator, maps and Python dependencies.

```bash
python -m pip install torch numpy PyYAML scipy matplotlib pytest
python scripts/check_source.py
python scripts/check_source.py --tests
```

`check_source.py` checks Python syntax and Markdown configuration hashes. With `--tests`, it copies only this source edition to a temporary directory, materializes settings there and runs the retained CPU tests. It disables GPU visibility and writes no generated data into the upload folder. The saved-smoke-artifact test is skipped when its original result directory is absent. A test pass checks implementation contracts, not recorded scientific effect sizes.

## Configuration files

All original YAML text is stored inside `docs/configs/*.md`, including seeds, update budgets, checkpoints, output paths and controls. Restore it locally with:

```bash
python scripts/materialize_configs.py --check
python scripts/materialize_configs.py
```

The materializer verifies the embedded hashes, refuses to overwrite differing existing files and does not launch an experiment. Edit a copied configuration to use new output paths before a new scientific run. Locally generated YAML files, checkpoints, traces, plots and caches are execution outputs, not source-release upload files.

## Inputs for experiment and analysis commands

The release contains code and documentation only. Frozen-policy comparisons require the recorded training checkpoints. Aggregators require the corresponding state/episode logs. Receiver interventions require the original learning-gate summary as well as the policies. Supplying arbitrary replacement policies changes the scientific comparison and cannot reproduce the reported numbers.

MNIST images can be obtained separately with `python scripts/fetch_mnist.py`; the downloader records the mirror and validates the four expected gzip checksums. The learned communication model must still be supplied or trained under the recorded protocol. `scripts/reproduce_external_continuation.py` previews commands for continuation from the six common 1,500-update states; it does not provide those states. The retained historical continuation module also contains the original owned-process handoff path; the preview helper avoids that manual handoff.

The exact original output paths are retained to describe the completed campaign. Scripts refuse existing output directories where their contract requires isolation. Source checks do not start training, rerun CPU timing or launch SC2.

## Validation of this edition

On 2026-11-24, all 106 Python files passed syntax checks and all 47 Markdown documents passed local-link checks. The 34 configuration blocks matched the original YAML bytes. In a separate temporary source-only copy, the retained tests reported **56 passed, 1 skipped**. The skip is the saved-smoke-artifact check, whose recorded outputs are intentionally absent from this edition. No experiment campaign, GPU job or original-result recomputation was performed.
