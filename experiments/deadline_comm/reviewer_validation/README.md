# reviewer_validation

Supporting implementation for the deadline-limited communication evaluation.

Inputs: repository-relative YAML settings, fixed checkpoints and matched reset seeds where required. Outputs: code-specific JSON/CSV estimates, traces or test assertions. Preserve checkpoint, cache, timing and replay controls when interpreting results. No new learning objective or universal scheduler advantage is claimed.

This is a selected source snapshot. See the root `REPRODUCE.md` for supported commands/dependencies and `EVIDENCE.md` for evidence, seeds and limitations. Existing study artifacts are retained separately from new runs; selected original final models and receiver inputs are now included; see the root `REVIEWER_GUIDE.md` and `evidence/reviewer/manifest.json` for exact availability and remaining gaps.

Reviewer artifact update (2026-11-22): original final Hallway and hidden-task cohorts, receiver NPZ data and compressed training/identity logs are available. `python scripts/verify_reviewer.py --probes --smoke` verifies hashes/model loads, recomputes saved probes and runs isolated bounded replay. Scientific implementation and reported results are unchanged. See the root reviewer guide for architecture, objectives, schema and experiment mapping.
