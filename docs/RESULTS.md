# Findings from completed experiments

These findings describe the completed experiments. The recorded checkpoints, arrays and traces are included in `checkpoints/` and `results/`; [the setup guide](REPRODUCE.md) explains how to check and recompute the supported comparisons. Confidence intervals describe the specified seed-level estimands.

| Comparison | Recorded result | Interpretation |
| --- | --- | --- |
| Hallway, one-packet Full versus RR | 400 paired episodes preserve delivery, cache, action and reward histories while RR removes 80% of offered bytes | Cost savings can remove only offers that could not affect the receiver |
| Hallway, two-packet Full versus RR | Received counts agree on common steps; sender histories differ in all 400 pairs; 222 of 259 equal-return pairs have different actions | Equal counts or return do not imply equal information or behavior |
| Feasible sender substitutions | 2,654 sampled states; return sensitivity in 13.94–18.86% across four cells | A sender can matter without the factual priority outperforming uniform feasible choices |
| Factual versus uniform sender choice | All four episode-weighted intervals include zero; two state-weighted intervals exclude zero | Weighting changes the target population and the apparent advantage |
| Withholding one factual update | Changed suffix return in 4.6–10.9% of branches under the original state-then-seed weighting; all four signed mean intervals include zero | Deletion dependence, replacement sensitivity and selection quality are different questions |
| Common RR history | Changed suffix return in 0/400 Slow and 15/400 Medium pairs | The episode-level access contrast includes history; equal suffix reward can hide changed behavior |
| Direct cache reset after a shared prefix | Changed action histories in 154/200 RR and 155/200 DAC pairs; all four branch types have zero suffix return | Memory changes behavior at an uninformative reward endpoint; cache mediation of the earlier return difference remains unresolved |
| Selector computation | Identical-input DAC gate cost exceeds RR by about 98–100 microseconds in the repeated analysis | Extra preparation consumes a reception opportunity only when a packet boundary and credit permit it |
| Normal-access receiver adaptation | 56.575% to 63.006% accuracy; paired gain 6.43 percentage points [3.32, 9.54] across ten seeds | Supervised labels improve the acting head on fresh closed-loop episodes |
| Shuffled-label receiver control | Normal RR accuracy 48.10%, versus 56.58% unchanged and 63.01% with true labels | The fitting targets matter beyond applying extra optimizer updates |
| Receiver deadline accounting | Normal original/adapted miss rates 1.63%/3.53%; every source-aware step misses the charged deadline | Additional reward does not establish improved hard-deadline operation |
| MNIST-sum learned reference | All three both-bias seeds pass at 12,000 updates; sampled held-out accuracy 50.21% versus 9.67% without biases | Intermediate learned policies supply a useful communication reference, not a reproduction of the original paper's final score |
| MNIST-sum, modeled 750 microsecond cutoff | Full and informed admission both 29.37% expected accuracy at 17/8.5 offered bytes per episode; random admission 18.78% at 8.5 bytes | Timing information preserves useful delivery at the same offered cost; paired gain 10.59 points [7.25, 13.92] across three seeds |
| Hallway local-clock control | Return 20 on all 1,024 initial states without messages | Learned channel benefit is not task-level necessity |

## Negative learning and transfer outcomes

XOR failed its learning criterion in all ten seeds. An early SMACv2 pilot also failed. The later three-seed Terran campaign passed the 300,000-step learning milestone, then failed independent-engine replay at the second seed after 320 evaluation episodes. A separate one-checkpoint technical smoke covers 20 episodes and 1,046 steps, including a 47-step Full–RR delivery identity. It does not establish multi-seed learned-policy deployment effects.

## Scope of interpretation

Hallway uses Oracle-assisted five-agent policies. Broadcast normal access and privileged source-aware access remain separate. Receiver adaptation adds supervised labels and permits feedback to later messages and source actions. MNIST-sum admission knows modeled readiness and is one-step. CPU measurements cover one host. The study does not claim a new learning method or generally superior scheduler.
