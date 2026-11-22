# Research completeness audit — 2026-11-22 KST

Scope: the English KBS manuscript and the fetched public repository at `78ad1b5`. The scientific source, configurations, selected tests, English manuscript/figures/bibliography, small recorded results, dependencies and third-party attribution are present. This is sufficient for inspecting the implementation and its reported findings, **not a complete experimental reproduction release**.

## Four-stage coverage

| Stage | Included implementation and evidence | Remaining reproduction dependency |
| --- | --- | --- |
| Validate learned references | Training/evaluation code, YAML settings, per-seed results, failed XOR/SMACv2 gates | Original trained checkpoints and training traces; historical exact environment is not locked |
| Account for delivery and computation | Transport/cache, Full–RR identity code/results, readiness raw logs and analysis | Full identity message histories/checkpoints; original hardware for timing replication |
| Isolate sender choice | Full-state replay and intervention code, fresh-state and common-prefix raw logs, frozen aggregates | Checkpoints for new policy rollouts; raw-log recomputation is now available |
| Check downstream use | Broadcast and receiver-input probe code/results; broadcast step logs | Receiver NPZ inputs/checkpoints for fitting and rerunning probes |

The local-clock counterexample, ten-seed cohorts, seed-level intervals, weighting distinctions, and failed learning gates remain part of the evidence. SMACv2 needs separately installed StarCraft II/maps and the declared upstream; its technical smoke does not establish learned benchmark generalization. DAC superiority and task-level necessity are not established.

## Additions and actual verification

- Four original logs, losslessly gzip-compressed (about 10 MB total), preserve all records rather than selecting successful seeds. Their source paths, raw byte counts and SHA-256 hashes are in `evidence/editorial/manifest.json`.
- `python scripts/verify_editorial.py` reconstructs the existing editorial analysis in a temporary directory and compares every field to the frozen JSON. It verified 2,654 fresh states, 800 prefix episodes, timing/broadcast accounting and the original hidden-source CSV checks. Float tolerance: relative `1e-10`, absolute `1e-12`.
- The configured CPU test suite passed: 45 passed, 1 skipped. Existing exclusions are documented in `REPRODUCE.md`; this is not a full training or paper rerun.
- Current bounded claims are available in `claim.md`; this audit adds no scientific result or training authorization.

## What still needs a separate artifact release

A full release needs the exact matched checkpoints, remaining message/replay traces, receiver NPZ inputs, environment provenance and a complete mapping of those inputs to each rerun command. They remain in the original workspace; no public download location or complete release is claimed. Publishing unrelated historical runs or substituting newly trained policies would not close the matched-evidence gap. No new training or GPU job was launched during this audit.

The manuscript still has pending author declarations. Publication format is not submission or acceptance. This audit did not rebuild the PDF, rerun probe fitting, or validate learned standard-benchmark transfer.
