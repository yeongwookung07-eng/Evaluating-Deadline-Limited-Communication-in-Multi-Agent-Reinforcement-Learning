# Reviewer guide: experiments, models and generated data

The English [KBS manuscript](experiments/deadline_comm/publication/kbs/kbs_main.tex) defines the claims. Start with [current claims](claim.md), then the checks below. This release adds exact original inputs to existing experiments; it does not add a new method, new benchmark success or newly trained evidence.

## Quick verification

```bash
python -m pip install -r requirements.txt
python scripts/verify_reviewer.py --probes --smoke
python scripts/verify_editorial.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
```

The reviewer verifier checks every artifact hash, loads all 280 checkpoints on CPU with `weights_only=True` and strict parameter matching, recomputes all four probe types in 100 cells, and runs the existing seed-0 RR/DAC fresh-subset smoke in a temporary output directory. It removes temporary replay outputs and preserves recorded results. Probe fitting uses saved features; it is not policy training. Unit-test exclusions are in [REPRODUCE.md](REPRODUCE.md). See [verification results](evidence/reviewer/VALIDATION.md).

## Experiment map

Paths in the next table are relative to `experiments/deadline_comm/`. All listed entry points and YAML configurations are included. Ten-seed studies use seeds 0–9 unless noted. Existing evidence is completed, including failed gates; no new full rerun is claimed. Historical configs name existing output directories: copy the config and change `output` before a new evaluation. Some modules hard-code their config path; use an isolated checkout for those reruns.

| Question / control | Entry point; config | Original evidence / inputs | Status and limits |
| --- | --- | --- | --- |
| Does the declared training procedure learn useful references? | `reviewer_validation/train_ten_seeds.py`; `configs/reviewer_ten_seeds.yaml` | `runs/reviewer_ten_seeds_20261118_001/`; all 180 final models and compressed training logs | Oracle-assisted Hallway; six methods, three training conditions. This is not task-level message necessity. |
| Does equal received information preserve control despite different traffic? | `review_revision/identity.py`; `configs/review_identity.yaml` | `runs/review_identity_20261119_001/`; ten Full checkpoints, two cache decays, 20 resets; 240 original trace files in `evidence/reviewer/` | Fixed-readiness 400-pair Slow result; other cells retained. Count wire bytes separately from payload and sender identity. |
| Can feasible same-count sender substitutions alter a frozen policy? | `tmlr_revision/experiments.py --kind fresh`; `configs/tmlr_fresh.yaml` | `runs/tmlr_fresh_20261119_001/`; 40 RR/DAC checkpoints; compressed states in `evidence/editorial/` | 2,654 recorded states; no-op and forced-original replay controls. State and episode weighting differ. |
| Do effects persist after common initialization? | `tmlr_revision/experiments.py --kind prefix`; `configs/tmlr_prefix.yaml` | `runs/tmlr_prefix_20261119_001/`; same 40 checkpoints, 20 resets, compressed episode traces | 800 recorded episodes; preserve early terminations and survivor distinction. |
| How does selector CPU readiness change delivery? | `tmlr_revision/readiness.py`; `configs/tmlr_readiness.yaml` | `runs/tmlr_readiness_20261119_001/`; Medium-trained RR/DAC models and editorial logs | Measured versus fixed readiness; exact timing depends on hardware/software and load. |
| Is broadcast information delivered and used? | `tmlr_revision/broadcast.py`; `configs/tmlr_broadcast.yaml` | `runs/tmlr_broadcast_20261119_001/`; original broadcast models, compressed step traces | Oracle-free custom task; normal and source-aware scheduling are distinct interventions. |
| Can actual receiver input decode information the acting policy misses? | `tmlr_revision/robustness/aggregate_probe.py`; `configs/tmlr_receiver_probe.yaml` | `runs/tmlr_receiver_probe_20261119_001/`; 100 NPZ cells with episode IDs and frozen model inputs | First ten episodes train probes, last ten test; compare actual unchanged policy accuracy. No learned remedy established. |
| Is communication necessary for Hallway itself? | `tmlr_revision/local_clock.py`; `configs/tmlr_local_clock.yaml` | `runs/tmlr_local_clock_20261119_001/`; Full seed-0 checkpoint metadata | Exhaustive 1,024-start constructive local-clock reference. No messages. |
| Do hidden-source learning gates pass? | `reviewer_validation/train_hidden.py`; `configs/hidden_mechanism.yaml` | `runs/hidden_mechanism_20261119_001/`; all 100 broadcast/XOR final models | Broadcast learning and failed XOR gate reported separately; all seeds retained. |
| Does the instrumentation run on SMACv2? | `smac_protocol/`; `configs/smacv2_protocol_smoke.yaml`, `configs/smacv2_protocol.yaml` | `runs/smacv2_protocol_20261121_001_smoke/` and failed run results | Technical smoke only, seed 0; SC2 runtime/maps and exact checkpoint are not supplied here. Failed learning gate remains. |

See [EVIDENCE.md](EVIDENCE.md) for secondary robustness and older adapted-policy studies. Their complete checkpoint and runtime dependencies are not implied by this table.

## Models and objectives

[CommPolicy](experiments/deadline_comm/policy.py) shares a recurrent agent encoder and maps hidden state to an eight-dimensional message. The action utility MLP consumes local recurrent state, the mean of valid other-agent cached messages, log age and valid fraction. [DecayedPolicy](experiments/deadline_comm/day2/policy.py) applies the configured decay while retaining the valid-count denominator. A QMIX monotonic mixer combines selected individual utilities during centralized training; decentralized action selection uses local utilities with received cached messages. The model has no learned coordination-factor graph; the communication model is broadcast with each receiver excluding its own message.

Hallway has two observation features per agent and three actions; checkpoint metadata supplies hidden width, message width and environment configuration. Broadcast/XOR models use four observation features, two actions, hidden width 64 and message width eight. Hallway minimizes masked squared TD error plus 0.5 times Oracle-action cross-entropy, using 50 demonstration episodes and one-quarter Oracle rows per minibatch. Hidden-source training uses TD error with BC weight zero and no Oracle episodes. These procedures have different supervision; inspect the corresponding training code/config and retained logs rather than treating them as equivalent objectives. Full, periodic, RR, DAC, ideal and no-communication arms control admission/reception or training conditions, not six new backbone architectures. DAC is a heuristic, not a generally superior learned scheduler. Existing QMIX/upstream attribution remains in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Environments and datasets

Hallway is the existing five-agent coordination environment adapted in `adapters/hallway.py`; synchronized group arrival determines task reward. The manuscript's local-clock reference reaches return 20 over all declared starts. The hidden-source generator in `reviewer_validation/hidden_source.py` supports at least three agents, while released cohorts use five. Broadcast has one source; XOR has two. Fresh independent source bits must be predicted by followers; reward is mean follower correctness, success means all followers correct, and the horizon is 20. These controlled diagnostic tasks are not standard-benchmark generalization.

No external static dataset is required for these generators. The released datasets are generated training logs, delivery histories and frozen-policy receiver features. Their schema, cohorts, hashes and original paths are documented in [evidence/reviewer/README.md](evidence/reviewer/README.md). Keep reset and training seed roles separate. The manifest records bytes of original artifacts, not a regenerated approximation.

## Remaining gaps

This is a materially expanded reproduction release, not a claim that every paper table has been independently rerun. Older day2/day3 adapted checkpoint banks, intermediate checkpoints, some secondary detailed traces, SMACv2 runtime/maps and its exact smoke model, and an exact historical environment lock remain outside this addition. Native VBC/TMC/SchedNet benchmark reproduction is not claimed. Compatibility ranges in `requirements.txt` are not historical provenance. The English manuscript still has pending author declarations; publication and acceptance are not asserted.
