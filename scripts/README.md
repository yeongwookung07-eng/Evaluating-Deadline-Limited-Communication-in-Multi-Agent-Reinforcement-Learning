# Public evidence verification tools

`verify_editorial.py` makes the existing editorial audit runnable from this checkout without training or model checkpoints. It reads `evidence/editorial/manifest.json`, compressed raw logs, the shipped hidden-source CSVs and the frozen editorial result. Install root `requirements.txt`, then run `python scripts/verify_editorial.py` from any directory.

The script checks hashes, reconstructs input paths in a temporary directory, calls the existing analysis and recursively compares outputs (exact discrete/text values; floating tolerances `1e-10` relative and `1e-12` absolute). It prints the analysis counts and PASS or fails with the mismatching field. Temporary outputs are deleted and frozen results remain unchanged. CPU-only operation was verified against the shipped data. It does not launch policy evaluation, training, SMAC or probe fitting; see `COMPLETENESS.md`.
