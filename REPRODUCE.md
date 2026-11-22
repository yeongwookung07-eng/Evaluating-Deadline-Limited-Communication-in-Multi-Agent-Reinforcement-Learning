# Reproduction and verification

Run commands at repository root. Core requirements are in `requirements.txt`; this is a compatibility range, not an exact historical environment lock. CPU-only verification is supported. SMACv2 additionally needs its pinned upstream and separately installed StarCraft II runtime/maps; those binaries are not redistributed. Inspect `experiments/deadline_comm/configs/smacv2_setup.yaml` before running the optional setup module (large downloads).

## Code checks

```bash
python -m pip install -r requirements.txt
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
```

The default selection covers transport, cache, replay, training arithmetic, sender interventions, and the SMAC protocol using test doubles. It excludes the historical `test_day2_gap_closure.py`: two tests require an earlier day2 checkpoint/harness outside the paper release. That original test remains unchanged. `test_pipeline.py` skips an integration-artifact check when the earlier smoke directory is absent. Neither exclusion is evidence of a full historical test-suite pass.

## Frozen evidence and checkpoint evaluation

Exact original final models and receiver inputs are now included: 180 Hallway checkpoints, 100 broadcast/XOR checkpoints and 100 NPZ cells. [Reviewer guide](REVIEWER_GUIDE.md) maps experiments to settings and inputs. [Manifest](evidence/reviewer/manifest.json) records hashes. CPU verification preserves recorded outputs:

```bash
python scripts/verify_reviewer.py --probes --smoke
python scripts/verify_editorial.py
```

The first checks all artifact hashes and model loads, recomputes four probe types from saved features in every cell, and runs seed-0 RR/DAC fresh-subset replay in a temporary directory. The second reconstructs the frozen editorial audit from four compressed original logs. Neither trains an acting policy. The exact available test scope/results are in [validation](evidence/reviewer/VALIDATION.md).

The original fresh command is now supplied with its required models:

```bash
python -m experiments.deadline_comm.tmlr_revision.experiments --kind fresh --config experiments/deadline_comm/configs/tmlr_fresh.yaml --smoke
```

It writes the configured `_smoke` directory and refuses an existing directory. Prefer the temporary wrapper above or a copied config with a unique `output`. Full configs preserve ten seeds, matched checkpoints and reset protocols. Some historical reports/bundle helpers and secondary runs still need omitted inputs; see the guide's remaining gaps. No new full campaign or historical GPU allocation is authorized by these instructions.

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
