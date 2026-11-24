# Required learning infrastructure

This is the subset used by the deadline-communication experiments: a shared recurrent agent, episode/replay representation and TD-learner base. The communication learner extends these components with recorded delivery histories and separate online/target recurrent unrolls. Inputs are full episodes and model/settings; outputs are learned parameters, Q values and losses.

The files retain the prepared public implementation. They are supporting infrastructure, not a new method or a complete export of the broader research repository. Focused replay, learning and cache tests are included under `experiments/deadline_comm/tests`; see the root setup guide for validation.
