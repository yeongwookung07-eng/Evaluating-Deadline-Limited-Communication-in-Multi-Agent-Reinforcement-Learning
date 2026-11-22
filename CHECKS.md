# Snapshot checks — 2026-11-22 KST

Executed from this `github/` directory in the existing installed Python environment:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
```

Result: **45 passed, 1 skipped in 3.53 seconds**. No training or GPU job was launched. This checks the configured transport/cache/replay/intervention and protocol test selection, not full paper reproduction or a newly provisioned environment.

`pytest.ini` explicitly excludes `test_day2_gap_closure.py`, whose two historical tests require omitted day2 fixtures. An integration-artifact check in `test_pipeline.py` skips because its earlier smoke directory is absent. Original test code is retained. Checkpoints, NPZ inputs and detailed traces are not distributed here. Raw evidence audits, frozen-policy evaluation and manuscript compilation were not rerun for this folder reorganization.

`FILES.sha256` records all shipped files except itself; verify from this directory with `sha256sum -c FILES.sha256`. Python sources, configs, small results and the paper were retained from the previous selected export; documentation was updated for this simpler distribution. Original research files and results were not deleted.

## Public checkout audit — 2026-11-22 KST

From `github_publish/` (the checkout of the public target repository), the CPU command above returned **45 passed, 1 skipped in 3.45 seconds**. `python scripts/verify_editorial.py` passed: 2,654 fresh states, 800 prefix episodes, readiness and broadcast accounting and hidden-source CSV checks; all generated editorial fields match the preserved result with float tolerances `rel_tol=1e-10`, `abs_tol=1e-12`. No training or GPU evaluation. This supersedes the earlier statement that no raw evidence audit was rerun. Full policy/probe reproduction remains unavailable.

## Expanded reviewer inputs (2026-11-22)

The new manifest/hash/model/probe/replay checks passed, and the configured CPU suite again returned 45 passed, 1 skipped. See [exact commands, scope and current environment](evidence/reviewer/VALIDATION.md). The previous source-only omissions are superseded only for artifacts explicitly listed in the reviewer manifest.
