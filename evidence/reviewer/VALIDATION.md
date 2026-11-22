# Reviewer release validation — 2026-11-22 KST

Executed in the public checkout with CPU-only settings. [Verification environment](verification_environment.json) records current versions, not a historical training lock.

| Command | Observed result |
| --- | --- |
| `python scripts/verify_reviewer.py --probes --smoke` | 900 artifact hashes passed, including decompressed hashes; 280 strict model loads; all four probe types and policy accounting matched in 100 cells; temporary seed-0 RR/DAC fresh replay completed at Slow/5 ms and Medium/2 ms |
| `python scripts/verify_editorial.py` | Every frozen editorial field matched; 2,654 fresh states and 800 prefix episodes retained |
| `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q` | 45 passed, 1 skipped in 2.93 seconds |

Raw supplied bytes: 239,326,479 (checkpoint/NPZ files plus compressed trace files). Manifest covers 280 checkpoints, 100 receiver inputs, 240 identity histories and 280 training logs. Checkpoints and NPZ bytes are unchanged from original workspace artifacts; gzip trace payloads have raw hashes. All methods/seeds within the declared final cohorts are retained.

No full training, full evaluation campaign, timing measurement, SMAC rollout or manuscript rebuild was run in this update. Existing test exclusions remain documented in `REPRODUCE.md`. Probe agreement validates recomputation from frozen inputs, not independent regeneration of those inputs. The short replay verifies only its declared subset. No scientific implementation was changed.
