# Deadline-limited communication experiments

These experiments test what explains communication-efficiency comparisons under action deadlines and persistent receiver caches. They hold policy weights and reset seeds fixed where the question concerns delivery or selection, and distinguish those interventions from independently trained references.

Implementations cover Oracle-assisted Hallway, Oracle-free hidden-source broadcast/XOR, the MNIST-sum communication game, and SMACv2 feasibility checks. Manipulated variables include channel access, sender choice, cache history, preparation time and the acting receiver head. Measurements include offered bytes, delivered payload and identities, cache state, deadline misses, actions, returns and accuracy.

The [experiment map](../../docs/EXPERIMENTS.md) lists each question, entry point, configuration and control. YAML files in [configs](configs/) retain original seeds and budgets. [Results](../../results/README.md) and [checkpoints](../../checkpoints/README.md) contain completed observations, including failed learning and replay checks. [Research scope](../../docs/RESEARCH.md) and [findings](../../docs/RESULTS.md) explain the supported interpretation.

Run `python experiments/prepare_inputs.py` from the repository root before using saved inputs. Shared recurrent-agent, replay and TD components are in `models/shared/`; QMIX is in `models/qmix/`. [Tests](../../tests/README.md) cover implementation contracts. See [setup](../../docs/REPRODUCE.md) for verification and external dependencies.
