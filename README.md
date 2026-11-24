# Evaluating Deadline-Limited Communication in Multi-Agent Reinforcement Learning

**Delivery, Computation, and Control** — Yeong-woo Kung and Young-guk Ha.

This repository contains the experiment and analysis code for a study of MARL communication evaluation. The central question is whether traffic and episodic return identify the control contribution of message selection under deadlines and persistent reception. The experiments separate admission, timely delivery, sender choice, prior history, CPU preparation and receiver information use.

The current source edition is **2026-11-24**. It contains Python code and Markdown documentation, including the experiment settings. Trained models, datasets, recorded traces and manuscript files are not bundled. The quantitative findings below describe completed experiments; checking this source release does not regenerate their results.

## Start here

- [Research question, methods and scope](docs/RESEARCH.md)
- [Experiment map and commands](docs/EXPERIMENTS.md)
- [Recorded findings](docs/RESULTS.md)
- [Setup and source checks](docs/REPRODUCE.md)
- [Configuration catalogue](docs/configs/README.md)
- [References and attribution](docs/REFERENCES.md)

The evaluation follows four stages: validate a learned communication reference; match delivery opportunities and costs; isolate sender choice and prior history; test whether the acting receiver uses its information. Oracle-assisted Hallway, Oracle-free broadcast and the published MNIST-sum game answer different parts of this question. DAC is an evaluated heuristic, not a proposed universally superior scheduler.

## Code layout

| Directory | Role |
| --- | --- |
| `experiments/deadline_comm/` | Policy, cache, serial transport, learning and evaluation |
| `reviewer_validation/`, `review_revision/` within that directory | Primary training controls and matched delivery histories |
| `tmlr_revision/` and its `robustness/` directory | Sender substitution, withholding, shared prefixes, weighting, timing and probes |
| `attribution_boundaries/` | Delivery-identity boundary and direct receiver-memory reset |
| `receiver_adaptation/` | Supervised follower-head intervention and shuffled-label control |
| `external_biases/`, `external_deadline/` | Learned MNIST-sum reference and modeled-deadline admission |
| `smac_protocol/` | Technical smoke, learning and independent replay checks |
| `framework/`, `models/qmix/` | Required recurrent-agent, replay, TD and QMIX components |
| `external/upstreams/CASEC-MACO-benchmark/` | Attributed upstream Hallway source |

Historical directory names are retained so imports and saved experiment paths stay recognizable. The export omits manuscript generation and unrelated coordination-graph research.

## Quick source check

```bash
python -m pip install torch numpy PyYAML scipy matplotlib pytest
python scripts/check_source.py
python scripts/check_source.py --tests
```

The checks use CPU only and do not launch an experiment campaign. Evaluation entry points require their documented checkpoints and traces. Configuration YAML files can be generated locally from the Markdown catalogue; see [setup](docs/REPRODUCE.md).

## Edition record

| Date | Work | Status |
| --- | --- | --- |
| 2026-11-24 | Curate this study's experiment code, dependencies, settings and research documents from the completed work | Prepared source edition; validation recorded in `docs/REPRODUCE.md`. No new scientific experiment or training campaign. |
