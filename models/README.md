# Model and learning components

`shared/` contains the recurrent agent, episode/replay structures and TD-learner base. `qmix/interface.py` implements the monotonic QMIX mixer for centralized training: agent utilities and global state are its inputs. Decentralized actions use recurrent agent utilities at inference. The experiment policy adds message encoding and a receiver-input action head.

The models use TD training and monotonic value mixing. They are baseline components, with no new coordination-factor objective. Architecture settings are in [the YAML catalogue](../experiments/docs/CONFIGURATION.md), saved weights in [checkpoints](../checkpoints/README.md), and tests in [tests](../tests/README.md). Reference: Rashid et al., QMIX, ICML 2018; see [references](../experiments/docs/REFERENCES.md).
