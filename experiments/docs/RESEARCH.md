# Research question and design

## Claim

Traffic and episodic return do not identify the control contribution of message selection under deadlines and persistent reception. Lower traffic can remove packets that would never arrive. Equal received counts can hide different senders or ages. Equal reward can hide changed actions. A scheduler comparison can also change the complete history inherited by the policy.

## Policy and channel

Hallway and broadcast use a shared GRU, an eight-float message encoder and an action head receiving local hidden state, the masked mean of cached messages, log mean age and valid-cache fraction. QMIX provides centralized value mixing during training. An offered packet contains 32 payload bytes and a modeled 16-byte header. The serial link charges every offered packet, including deadline drops, and clears its queue each step.

Receiver caches start invalid. A timely update replaces the sender-keyed payload and timestamp. Late updates do not enter the cache. Aggregation divides by the number of valid senders, not the sum of decay weights. Evaluation records offered bytes, delivered payload, sender identities and timestamps separately.

Measured CPU work includes encoding, selection, serialization, transport arithmetic, cache operations and action computation. It is added to modeled waiting without overlap. Fixed readiness supplies a separate controlled reception opportunity. Ordinary task rewards do not impose an extra penalty on late actions; deadline misses must be read separately from return.

## Four comparisons

1. **Validate learning.** Compare channel access at matched training budgets and intervene on messages for frozen policies. A learned channel gap does not prove that the task requires communication.
2. **Account for delivery and cost.** Pair policy checkpoints, resets, cache rules and exogenous randomness. Check full received histories as well as traffic. Charge selector preparation before applying the reception deadline.
3. **Isolate sender choice.** Enumerate feasible sender sets at a fixed full state and received count. Compare the factual set with the uniform mean, then resume the original scheduler. Separately withhold an actual update and branch after a common history. Clearing receiver memory is another intervention.
4. **Test receiver use.** Fit a held-out decoder on the actual receiver input, then evaluate an adapted acting head in fresh closed-loop episodes with true-label and shuffled-label fitting controls.

## Estimands and controls

Deployment effects average paired episodes within each training seed and then weight seeds equally. Fresh-subset analyses report both state-first and episode-first means. Withholding retains its original averaging over removed senders within state, states within seed and then seeds. Branches are paired alternatives, not independent training replications.

A shared RR prefix matches environment state, recurrent state, cache, scheduler history and remaining horizon. It does not by itself identify a cache-mediated effect. Receiver adaptation retains the access schedule but may change later message contents and source actions through feedback.

## Tasks and limits

- **Hallway:** five agents, two groups, finite horizon, Oracle-assisted learned policies. A constructive local-distance/public-clock controller attains the full reward without communication.
- **Hidden-source broadcast:** five agents; one source sees a fresh independent bit, four followers predict it. The normal RR and privileged source-aware regimes are analyzed separately. This task is Oracle-free for the learned MARL reference; receiver-head adaptation uses additional bit labels.
- **MNIST-sum:** reconstruction of the published speaker/listener game of Eccles et al. A speaker sees one digit and sends one symbol; a listener sees the other digit and predicts their sum. Admission knows the modeled exogenous readiness; the resulting comparison is one-step and not a standard sequential MARL transfer result.
- **SMACv2/XOR:** negative learning and replay outcomes remain part of the results. A technical smoke does not establish a learned multi-seed deployment effect.

The study does not establish a new learning objective, native VBC/TMC/SchedNet superiority, universal communication necessity or a generally superior DAC scheduler. The experiments probe interpretations of evaluation practice on the specified tasks and policies. [References](REFERENCES.md).

## JAAMAS scope update

The evaluation-framework contribution is unchanged. Frozen-policy checks now assess modeled readiness uncertainty/decision overhead and actually rejected late actions. Benefits are conditional and can disappear or reverse; no measured predictor, physical actuator guarantee or standard-benchmark generalization is established. [Protocol](../deadline_comm/deployment_robustness/README.md).
