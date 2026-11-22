# Reproduction and verification

Run commands at repository root. Core requirements are in `requirements.txt`; this is a compatibility range, not an exact historical environment lock. CPU-only verification is supported. SMACv2 additionally needs its pinned upstream and separately installed StarCraft II runtime/maps; those binaries are not redistributed. Inspect `experiments/deadline_comm/configs/smacv2_setup.yaml` before running the optional setup module (large downloads).

## Code checks

```bash
python -m pip install -r requirements.txt
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
```

The default selection covers transport, cache, replay, training arithmetic, sender interventions, and the SMAC protocol using test doubles. It excludes the historical `test_day2_gap_closure.py`: two tests require an earlier day2 checkpoint/harness outside the paper release. That original test remains unchanged. `test_pipeline.py` skips an integration-artifact check when the earlier smoke directory is absent. Neither exclusion is evidence of a full historical test-suite pass.

## Frozen evidence and checkpoint evaluation

This folder contains source, configurations and small recorded results, plus **four compressed raw editorial logs**, but no trained checkpoints or receiver NPZ inputs. There is no packaged or published Release dependency. The supported editorial verification below works from this checkout alone. It checks compressed and uncompressed hashes, expands logs into a temporary directory, runs the unchanged scientific audit, and compares every field with the frozen result. It leaves tracked results untouched. No YAML config or checkpoint is required. Consult `EVIDENCE.md` for the corresponding run directories.

```bash
python scripts/verify_editorial.py
```

This reconstructs recorded denominators, confidence intervals and controls; it does not retrain policies. The following frozen-policy command still requires omitted original checkpoints at their recorded paths and is **not runnable from this checkout alone**:

```bash
python -m experiments.deadline_comm.tmlr_revision.experiments --kind fresh --config experiments/deadline_comm/configs/tmlr_fresh.yaml --smoke
```

It uses fixed seed-0 RR/DAC checkpoints and writes a new `_smoke` output. Use a copy of the config with a fresh `output` path if that directory exists. The full configs retain seeds 0–9, reset rules, conditions and checkpoint selection; no new training was performed for this export. `settings.py`, `report.py`, `editorial_audit.py` and `robustness/report.py` retain analysis/plotting code. Some historical packaging/report helpers refer to omitted old manuscript bundles; they are not the supported public build path.

## Paper

```bash
cd experiments/deadline_comm/publication/kbs
pdflatex -interaction=nonstopmode -halt-on-error kbs_main.tex
bibtex kbs_main
pdflatex -interaction=nonstopmode -halt-on-error kbs_main.tex
pdflatex -interaction=nonstopmode -halt-on-error kbs_main.tex
pdflatex -interaction=nonstopmode -halt-on-error kbs_main.tex
```

TeX must provide the CAS dependencies, Latin Modern and microtype. The old `build.py` references historical manuscripts and is intentionally omitted. Scientific code is unchanged except project-path portability replacements; SMAC setup's machine-specific Python setting is `python3`. `FILES.sha256` records the files in this snapshot. Results come from the original retained runs, not new measurement on this export machine.
