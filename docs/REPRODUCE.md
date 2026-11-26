# Setup and execution

## Install and prepare

The verified clean installation uses Linux x86-64, Python 3.13.12 and CPU-only PyTorch 2.15.0. It starts from a fresh GitHub source download with no inherited Python packages. The original research environment (PyTorch 2.11.0+cu128, NumPy 2.4.4) is a separate provenance record; it is not required for the saved-result audit.

Run from the root of a fresh clone or extracted GitHub source ZIP:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt -c constraints-cpu-verified.txt
python experiments/verify_results.py --recompute --tests
python experiments/reproduce_paper_tables.py
```

The constraints file records the package versions actually tested. Other Python/platform combinations are not certified. The main verification command prepares its inputs automatically. The table command then writes `results/local_recompute/paper_tables/{external_deadlines,receiver_controls_acting_rows,uncertain_readiness,hard_action}.csv` and `verification.json`, checks printed manuscript values and exits nonzero on a difference. [Table mapping and scope](TABLE_REPRODUCTION.md) distinguish reaggregation, model inference and a new experiment.

The repository includes 375 checkpoints and 106 generated NumPy archives, together with summaries and traces. [The manifest](../results/manifest.json) lists 5,741 scientific assets with original paths, sizes and SHA-256. Preparation verifies every entry and creates independent working copies at original paths, refusing to overwrite different files. Allow about 0.8 GB for the extra working copies, in addition to the download and Python environment. For a hash-only check: `python experiments/prepare_inputs.py --check-only`.

Original YAML configurations are in [the configuration catalogue](CONFIGURATION.md). They preserve seeds, budgets, checkpoint paths and controls. No GPU, raw MNIST download, StarCraft II installation or training is required for the default commands above. Document compilation separately requires LaTeX and Poppler; see [paper build](../paper/README.md).

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

Typography-only update (2026-11-25): `paper/build.py` now produces a 15-page main paper and 9-page supplement with common 10pt text, including figure labels. Scientific text, all 18 tables and plotted numeric primitives match the preceding edition. See `paper/typography_validation_20261125.json`. The earlier dated build results above remain historical.

Author-requested 8pt edition (2026-11-25): the current build yields an 11-page main paper and 7-page supplement (18 total). Only typesetting and plotted text sizes changed; body source and scientific plot values are preserved. See `paper/typography_8pt_validation_20261125.json`.

## Current JAAMAS edition — November26

Active documents: [JAAMAS submission files](../submission/jaamas/README.md). Earlier dated KBS builds above remain historical. The new manifest has5,741assets and106NPZ files, with375unchanged checkpoints. The usual `prepare_inputs.py` and `verify_results.py --recompute --tests` commands include the new exact timing model and18,000hard-fallback rollout audits. These audits use saved observations, not a new measured CPU campaign. Raw MNIST is unnecessary for saved-result verification.

To regenerate the two new evaluations, first copy their YAML files and select unique output paths; the recorded directories are never overwritten. MNIST requires the separately fetched raw images, and measured broadcast timings depend on host load. Full commands, budgets and limitations: [component README](../experiments/deadline_comm/deployment_robustness/README.md). To compile and package documents: `python scripts/package_jaamas.py`. To create portable reviewer materials too: `python scripts/package_jaamas.py --skip-build --reviewer`.

Validation completed for this JAAMAS edition: all5,741asset hashes,375checkpoint loads, retained paired/probe/replay checks and the new full raw audits pass. CPU tests: **63passed,1skipped** (unbundled early integration-smoke inputs). A separately extracted source ZIP rebuilds all five document texts identically. No training/GPU work. See `docs/validation_jaamas_20261126.json` and `paper/submission_validation.json`.

## Clean isolated CPU installation and revised package — November 26

An actual GitHub source ZIP at commit `d44c371` was extracted independently of the research workspace. A new venv with `include-system-site-packages = false` installed CPU PyTorch and the requirements successfully. Complete verification passed with 5,741 asset hashes, 375 CPU checkpoint files and 63 passing tests (one unbundled-fixture skip). The revised reviewer package was then separately extracted and passed both the complete verifier and the table-export command in that isolated environment. [Machine-readable evidence and logs](validation_clean_cpu_20261126.json) distinguish these two checks. The source ZIP alone also rebuilds all five PDFs with identical extracted text. Saved numerical results are reproducible within this declared scope; historical training trajectories, two receiver-probe export rows and hardware latency invariance are not claimed by the table helper.

## Problem-and-rationale manuscript rewrite — November 26

The current main paper is 31 pages with a 232-word abstract. The supplement has 17 pages and the Information Sheet has 2 pages. The four-stage protocol table has moved ahead of the experiments and now states questions/controls; scientific table values and all supplementary source remain unchanged. Table reproduction uses stable LaTeX labels, so the same commands apply after renumbering. No training/evaluator code or scientific assets changed. See [rewrite validation](../paper/problem_rationale_rewrite_validation.json) for current text/build/table checks; earlier dated CPU-environment receipts remain valid records of their editions.
