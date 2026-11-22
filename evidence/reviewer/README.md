# Original models and generated datasets

This release supplies existing evidence for reviewer inspection; no new scientific experiment or policy training was performed. `manifest.json` lists original repository-relative paths, file sizes and SHA-256 hashes. Checkpoints and NPZ files are byte-identical copies, stored at the paths expected by the original evaluation code. Compressed JSONL files additionally record the uncompressed hash and size.

## Included cohorts

- Hallway: all 180 final `checkpoint_1500.pt` files, ten seeds × three training conditions (Slow/5 ms, Medium/2 ms, Medium/5 ms) × six methods. Intermediate training checkpoints are omitted. Keep independently trained Full, periodic, RR, DAC, ideal and no-communication policies distinct from same-policy interventions.
- Hidden-source tasks: all 100 `final.pt` checkpoints, both broadcast and failed XOR cohorts, ten seeds × five methods. No successful-seed selection.
- Receiver inputs: all 100 original NPZ cells, ten seeds, three methods, two link/deadline conditions, normal and source-aware schedulers where applicable. These are generated frozen-policy features, not an external supervised-learning benchmark.
- Full original identity JSONL histories and available `training.jsonl` logs from the two cohorts, losslessly compressed. Counts and availability are explicit in the manifest; these logs are not a historical dependency lock.

## Dataset schema and use

Each receiver NPZ contains `utility_input` (step × four follower receivers × 74 features), `source_message` (step × 8), binary `bit`, Boolean `source_received`, integer `episode`, and `policy_correct` (step × four followers). Features comprise 64 recurrent hidden coordinates, eight received aggregate coordinates, log age and valid fraction. The first ten reset episodes train each receiver-specific ridge probe; the last ten test it. Never randomly split individual steps. Ridge is 0.001. Source-aware inputs and normal scheduling are separate cells. Probe accuracy is not acting-policy return.

Identity histories retain sender and timestamp details for delivery attribution. To inspect a compressed trace use Python's `gzip.open(path, 'rt')`; `source` in the manifest identifies its original location. Original artifacts remain unmodified. `python scripts/verify_reviewer.py --probes --smoke` checks hashes, strict model state loading, all probe fields and bounded isolated replay. It never trains an acting policy and uses CPU only.

See [reviewer guide](../../REVIEWER_GUIDE.md) for experiment/config mapping, model architecture, remaining omissions and limitations. These artifacts inherit the scientific scope of the manuscript; redistribution of StarCraft II binaries is excluded.
