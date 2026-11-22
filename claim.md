# Current claims — KBS manuscript, 2026-11-22 KST

**Evaluating Deadline-Limited Communication in Multi-Agent Reinforcement Learning: Delivery, Computation, and Control**

Canonical source: [English manuscript](experiments/deadline_comm/publication/kbs/kbs_main.tex), [PDF](experiments/deadline_comm/publication/kbs/kbs_main.pdf), [version and validation status](experiments/deadline_comm/publication/kbs/README.md). Claims below summarize the existing manuscript; this alignment adds no experimental evidence.

## Contribution

A controlled evaluation framework attributes communication-efficiency outcomes to admission, timely delivery, sender selection, initialization, computation and downstream policy use. Matched-checkpoint delivery-identity tests, feasible same-count sender substitutions, common-prefix interventions, measured/fixed readiness and receiver-input probes form a four-stage evaluation protocol. The contribution is their operational combination for deadline-and-cache evaluation, not a new communication-learning objective or a generally superior scheduler.

## Supported findings and boundaries

- **Delivery identity (Section 5.3):** Oracle-assisted five-agent Hallway, ten slow-trained Full checkpoints, 20 paired resets and two matched cache-decay settings: all 400 Full–RR paired episodes preserve delivered histories, caches, actions, rewards and return at fixed readiness, while RR reduces attempted traffic by 80%.
- **Sender effects (Section 6.1):** same-count substitutions change suffix return in approximately 13.9–18.9% of sampled states under the primary aggregation. Priority-over-uniform advantages depend on state versus episode weighting; all four equal-episode intervals include zero. Nonzero sender effects do not establish robust DAC superiority. After a common RR prefix, suffix return changes in 0/400 Slow and 15/400 Medium episodes; equal return can conceal different information/actions.
- **Computation and information use (Sections 6.2–6.3):** selector readiness changes the reception opportunity. In Oracle-free custom broadcast with source-aware selection, held-out aggregate probes reach 99.91–100% accuracy versus 64.17–78.44% for unchanged acting policies. Accessible information is not fully exploited in those cells; this does not establish a training remedy. Broadcast passes the stated learning check, XOR fails (0/10).
- **Task necessity (Section 6.4):** a constructive local-distance/public-clock controller achieves return 20 without messages on all 1,024 Hallway starts. Learned Ideal–No-Comm gaps concern the specified Oracle-assisted training procedure, not task-level communication necessity.
- **SMACv2 feasibility (Section 6.5):** 20 technical-smoke cells/episodes, 1,046 steps, one seed-0 checkpoint. Slow/5 ms fixed-readiness identity holds over one 47-step episode with 80% offered-wire-byte reduction. Medium/2 ms has a sender mismatch despite equal payload goodput. Shadow traces after mismatch are not independent realized return trajectories. The earlier learning pilot fails its gate; learned standard-benchmark statistical generalization remains open.

## Claims not established

No generally superior DAC scheduler, native VBC/TMC/SchedNet superiority, universal communication necessity, new MARL learning objective, minimum coordination graph, universal guarantee or successful standard-benchmark transfer is established. The manuscript is prepared in KBS CAS format; submission, acceptance and final author declarations are not asserted.

Evidence navigation: [frozen-policy campaign report](experiments/deadline_comm/tmlr_revision/results/REPORT.md), [SMACv2 protocol/status](experiments/deadline_comm/smac_protocol/README.md). These reports retain historical naming; the English KBS source defines current manuscript claims.
