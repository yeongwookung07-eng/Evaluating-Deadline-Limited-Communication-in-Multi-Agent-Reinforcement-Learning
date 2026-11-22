# Frozen editorial raw evidence

These four gzip files are exact, lossless copies of existing research logs, added on 2026-11-22 KST to make the manuscript's editorial accounting independently recomputable. No scientific run was launched. The source paths, uncompressed sizes, raw SHA-256 and compressed SHA-256 are recorded in `manifest.json`; gzip timestamps are zero for deterministic packaging.

Inputs cover fresh same-count sender substitutions, common RR prefixes, measured/fixed readiness, and Oracle-free broadcast steps. The retained logs include every original row and seed. Existing `runs/hidden_mechanism_20261119_001` CSVs supply the audit's learning-evaluation consistency checks.

Run from repository root:

```bash
python scripts/verify_editorial.py
```

The utility checks both hashes, expands the logs in a temporary directory, invokes the unchanged `tmlr_revision/editorial_audit.py` analysis, compares every field against its recorded JSON and removes temporary data. It writes no tracked result. This verifies denominators, descriptive seed-level intervals, prefix replay, timing accounting, reconstructed broadcast offered counts and hidden-source CSV consistency. Broadcast attempted traffic is reconstructed under the recorded deterministic gate, not newly measured CPU timing.

Verified: 2,654 fresh states and 800 prefix episodes; all editorial result fields match. This is not checkpoint execution, full Full–RR delivery-trace verification, receiver-probe fitting, or full paper reproduction. Uncompressed inputs need about 70 MB of temporary disk space. See the root `COMPLETENESS.md` for outstanding dependencies.
