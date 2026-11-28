# Shared learning infrastructure

This component supplies a recurrent agent, episode/replay representation and TD-learner base. The communication learner extends them with recorded delivery histories and separate online/target recurrent unrolls. Inputs are full episodes and model settings; outputs are learned parameters, Q values and losses.

These components support the existing baselines and analysis. The public directory layout changes their import location without changing learning rules. [Tests](../../tests/README.md) cover replay, learning and cache behavior; [setup](../../experiments/docs/REPRODUCE.md) records verification. Message encoding and receiver heads remain in `experiments/deadline_comm/`.
