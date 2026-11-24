# QMIX dependency

`qmix/interface.py` implements the monotonic value mixer used for centralized training; agent utilities and global state are its inputs. Shared recurrent agent utilities select decentralized actions during inference. The communication policy adds the sender message encoder and receiver-input action head in the experiment component.

This is an existing QMIX baseline component, with TD training and a monotonic mixing structure. It is not the proposed contribution and does not introduce learned coordination factors or a minimum graph. Architecture settings are recorded in the Markdown configuration catalogue. Reference: Rashid et al., QMIX, ICML 2018.
