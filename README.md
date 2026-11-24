# Evaluating Deadline-Limited Communication in Multi-Agent Reinforcement Learning

**Delivery, Computation, and Control**

Experiment code, recorded results, trained checkpoints and analysis of communication under deadlines and cached reception. The study separates offered traffic, timely delivery, sender choice, initialization, CPU preparation and the acting policy's use of received information.

## Download and run

Clone this repository, or choose **Code → Download ZIP** and extract it. Run from the repository directory:

```bash
python -m pip install -r requirements.txt
python experiments/prepare_inputs.py
python experiments/verify_results.py --recompute --tests
```

Preparation verifies the included files and copies them to the original experiment paths. Verification loads checkpoints, recalculates recorded comparisons and runs CPU tests and a small frozen-policy replay. It does not train policies. The repository includes 375 checkpoint files; raw MNIST images and the StarCraft II simulator are separate dependencies. See [setup and execution](docs/REPRODUCE.md).

## Contents

| Directory | Contents |
| --- | --- |
| [experiments](experiments/README.md) | Experiment implementations, YAML settings and reproduction commands |
| [docs](docs/RESEARCH.md) | Research question, experiment map, findings and references |
| [models](models/README.md) | Recurrent-agent, replay, TD and QMIX components |
| [results](results/README.md) | Recorded summaries, paired traces, receiver inputs and file hashes |
| [tests](tests/README.md) | Tests of delivery, cache, learning and replay contracts |
| [paper](paper/README.md) | Main paper, supplement, figures and LaTeX source |
| [checkpoints](checkpoints/README.md) | Saved Hallway, broadcast/XOR, receiver, MNIST-sum and SMACv2 models |

## Analysis

The experiments ask when lower traffic represents different received information, whether sender choices change control, how shared history changes comparisons, whether CPU preparation changes delivery, and whether the acting policy uses information present in its input. Oracle-assisted Hallway, Oracle-free broadcast and the MNIST-sum game are reported separately. Failed learning and replay checks are retained alongside the completed comparisons.

- [Research question and evaluation protocol](docs/RESEARCH.md)
- [Experiment map and configurations](docs/EXPERIMENTS.md)
- [Recorded findings and limitations](docs/RESULTS.md)
- [References and attribution](docs/REFERENCES.md)
