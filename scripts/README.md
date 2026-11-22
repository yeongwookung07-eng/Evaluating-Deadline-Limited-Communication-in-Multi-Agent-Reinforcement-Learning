# Public evidence verification tools

`verify_editorial.py` makes the existing editorial audit runnable from this checkout without training or model checkpoints. It reads `evidence/editorial/manifest.json`, compressed raw logs, the shipped hidden-source CSVs and the frozen editorial result. Install root `requirements.txt`, then run `python scripts/verify_editorial.py` from any directory.

The script checks hashes, reconstructs input paths in a temporary directory, calls the existing analysis and recursively compares outputs (exact discrete/text values; floating tolerances `1e-10` relative and `1e-12` absolute). It prints the analysis counts and PASS or fails with the mismatching field. Temporary outputs are deleted and frozen results remain unchanged. CPU-only operation was verified against the shipped data. It does not launch policy evaluation, training, SMAC or probe fitting; see `COMPLETENESS.md`.

## Reviewer input verification

`verify_reviewer.py` reads the reviewer artifact manifest, checks compressed/raw hashes and strictly loads all 280 original models on CPU. `--probes` recomputes all four receiver-specific ridge probes and policy accounting from 100 saved episode-disjoint datasets against the frozen summary. `--smoke` runs existing fresh-subset replay with seed-0 RR/DAC models in two conditions in a temporary directory. No policy training, GPU or tracked-result modification occurs. Requires root requirements; run from any directory. A mismatch raises an error; these checks do not certify omitted studies or reproduce hardware timing.
