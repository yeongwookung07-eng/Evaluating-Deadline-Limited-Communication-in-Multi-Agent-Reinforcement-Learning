# Reproduction and verification

Run commands at repository root. Core requirements are in `requirements.txt`; this is a compatibility range, not an exact historical environment lock. CPU-only verification is supported. SMACv2 additionally needs its pinned upstream and separately installed StarCraft II runtime/maps; those binaries are not redistributed. Inspect `experiments/deadline_comm/configs/smacv2_setup.yaml` before running the optional setup module (large downloads).

## Code checks

```bash
python -m pip install -r requirements.txt
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
```

The default selection covers transport, cache, replay, training arithmetic, sender interventions, and the SMAC protocol using test doubles. It excludes the historical `test_day2_gap_closure.py`: two tests require an earlier day2 checkpoint/harness outside the paper release. That original test remains unchanged. `test_pipeline.py` skips an integration-artifact check when the earlier smoke directory is absent. Neither exclusion is evidence of a full historical test-suite pass.

## Frozen evidence and checkpoint evaluation

This folder contains source, configurations and small recorded results, but **no trained checkpoints, NPZ inputs or detailed message traces**. There is no packaged or published Release dependency. The following audit and frozen-policy commands require those original inputs to be copied from the research workspace into matching relative paths first; they are not runnable from this folder alone. Consult `EVIDENCE.md` for the corresponding run directories.

```bash
python -m experiments.deadline_comm.tmlr_revision.editorial_audit
```

This reconstructs recorded denominators, confidence intervals and controls; it does not retrain policies. A bounded frozen-policy fresh-message smoke uses:

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
