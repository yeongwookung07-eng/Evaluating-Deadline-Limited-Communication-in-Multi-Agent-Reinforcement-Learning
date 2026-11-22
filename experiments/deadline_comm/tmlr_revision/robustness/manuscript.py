"""Insert completed follow-up evidence on every manuscript rebuild."""
from pathlib import Path
import json

def integrate(src):
    root=Path('experiments/deadline_comm/tmlr_revision/results')
    result=json.loads((root/'robustness_summary.json').read_text())
    assert result['verified_receiver_steps']==40000 and result['artifact']['stage']=='completed'
    src=src.replace('Measured readiness changes discrete packet opportunities.', 'Measured readiness changes discrete packet opportunities, but the size of the packet-count difference varies across fresh processes.')
    src=src.replace('CPU readiness changes discrete reception opportunities, and source delivery alone does not ensure successful use of decodable broadcast messages.', 'CPU readiness changes discrete reception opportunities, with process-sensitive effect sizes. Held-out probes of the actual receiver aggregate distinguish accessible broadcast information from its use by the trained action head.')
    anchor=r'\paragraph{Initialization versus later refresh.}'
    addition=r'''\paragraph{Weighting and remaining-task sensitivity.}
Replaying all 400 factual trajectories verifies the 2,654 sampled states and finds no simultaneous-arrival penalty in those factual trajectories. Outcome-resolved waiting occupies 476 of 5,290 factual steps. Excluding resolved sampled states raises effect rates to 16.48--25.92\%; equal-episode instead of equal-state weighting gives 14.76--21.79\% (Table~\ref{tab:robust_strata}). These rates remain conditional on the sampled checkpoints and histories. With equal-state weighting, the directly paired factual-minus-uniform contrast is positive with an unadjusted t interval excluding zero for Slow/DAC and Medium/RR, but not the other two cells. With equal-episode weighting, all four intervals include zero. Thus a positive selection advantage over uniform is not robust to this change of estimand, even though nonzero subset effects remain. This pattern does not support a general DAC selection advantage. Appendix~\ref{app:robustness} gives the aggregation and counterfactual sensitivity boundary.

'''
    assert anchor in src;src=src.replace(anchor,addition+anchor,1)
    src=src.replace('group counts and remaining time do not prove that every survivor still has a reward-changing decision.',r'group counts alone do not establish a reward opportunity. A separate executed global-state suffix reference verifies unused attainable reward in 335/400 slow-link pairs, including every pair with an unresolved group. Thus universal reward saturation also fails to explain the zero contrast, although the reference changes actions and does not prove that sender changes alone can realize that reward (Appendix~\ref{app:robustness}).')
    anchor=r'\paragraph{Locating the broadcast bottleneck.}'
    add=r'''\paragraph{Independent-process timing check.}
Three new sequential processes on the same pinned CPU repeat all ten primary seeds at 2.0, 2.25 and 2.5 ms, using three evaluation resets per cell rather than the original ten. At 2.25 ms, RR delivers 2.213, 2.005 and 2.003 packets/step, while DAC delivers 2.000 in all three processes (Table~\ref{tab:robust_cpu}). The original packet-count advantage is therefore not a stable effect magnitude. On identical frozen-policy/input snapshots, DAC's additional gate cost averages 98.5, 99.4 and 100.7 microseconds across these processes, respectively. This isolates an implementation cost difference, while its conversion into packet opportunities remains sensitive to timing tails and credit. The repetitions are not additional independent training seeds or a second-machine replication.

'''
    assert anchor in src;src=src.replace(anchor,add+anchor,1)
    old='Source-message probe success does not identify whether the receiver aggregate preserves the bit or its action head uses it; no aggregate probe was performed, and diagnostic selection CPU overhead was not measured.'
    new=r'''Source-message decoding alone cannot localize the downstream bottleneck. We therefore probe the actual receiver aggregate and full utility-head input, with 40,000 replayed steps checked against saved outcomes and an independent reconstruction of the input. On the held-out half of the reset cohort, source-aware aggregate decoding is 99.91--100\% while the unchanged acting policies achieve 64.17--78.44\% across the two tested conditions and RR/DAC weights (Table~\ref{tab:receiver_probe}). This rules out absent linearly accessible aggregate information as the sole explanation in those source-aware cells. It does not identify why the original policy failed to learn to use it. Diagnostic selection CPU overhead remains unmeasured.'''
    assert old in src;src=src.replace(old,new)
    src=src.replace('Measured preparation time changes packet opportunities, and broadcast diagnostics distinguish source access from downstream use of decodable messages.', 'Measured preparation time changes packet opportunities, although independent-process repeats show that the packet-count advantage is not stable in magnitude. Receiver-aggregate probes distinguish source access and linearly accessible information from the original action head\'s use of it.')
    appendix=r'''\section{Weighting, Receiver-Input and Process Robustness}\label{app:robustness}
\paragraph{Sampling and weighting.}
All comparisons reuse frozen checkpoints. The factual-minus-uniform estimand at sampled state $x$ is
\[
\Delta_x=V_x(A_{\rm factual})-\frac{1}{|\mathcal A_{k(x)}|}\sum_{A\in\mathcal A_{k(x)}}V_x(A).
\]
Equal-state aggregation averages sampled states within a seed; equal-episode aggregation first averages sampled states in each episode, then episodes within a seed. Both finally weight ten training seeds equally. All new intervals are unadjusted descriptive 95\% Student-t intervals of seed means; 5,000 seed bootstrap resamples and seed-specific denominators are supplied in the companion JSON. Unresolved-group strata use the environment's pre-intervention group flags; empty strata are not zero-imputed. The 224 sampled states with no unresolved group all have zero range.

The factual replay contains no negative penalty episode. Among 19,875 saved candidate suffix totals, one is negative, at one Medium/DAC state. Excluding that state is an outcome-selected sensitivity check, not a replacement primary analysis; the paired contrast and its interval remain supplied in the companion table data. A nonnegative suffix total can conceal an intermediate penalty offset by a reward, so this scan is not a full audit of intermediate counterfactual rewards. It does not establish independence of the diagnostic from all Hallway transition rules.

\paragraph{Remaining reward opportunities.}
At each common-prefix endpoint, a privileged reference schedules active groups at distinct completion times no earlier than their maximum remaining distance and within the remaining horizon. The two-group setting permits an exact attainable remaining-reward reference. All 800 prefixes are replayed and the reference's predicted reward is checked against 3,406 executed suffix steps. Slow/5 ms has 65, 296 and 39 endpoints with remaining attainable reward 0, 10 and 20; 335 endpoints admit more reward than either learned suffix. Medium/2 ms has 1, 260 and 139 such endpoints, and 361 with unused opportunity. This action-changing reference verifies opportunity, not a message-only causal explanation or a deployable learned controller.

\paragraph{Probe protocol.}
For each broadcast checkpoint, condition and selection rule, episodes 14200000--14200009 fit per-receiver ridge probes and 14200010--14200019 evaluate them. All normalization statistics are fitted on the training episodes, with fixed ridge coefficient $10^{-3}$ and an unpenalized intercept. The probes access labels unavailable to the acting policy. A pre-hook captures the actual 74-dimensional utility-head input; its aggregate component is independently reconstructed, including cache validity, decay, normalization and receiver self-exclusion. All 40,000 steps agree with the saved source bit, source delivery, delivered count and reward. Probe fits and normal/source-aware policy scores in Table~\ref{tab:receiver_probe} use the same held-out episodes; these values must not be substituted for the full 20-episode decomposition means. Local recurrent state plus cache metadata is a separate diagnostic, not No-Comm, since availability itself can signal information. Linear probe failure would not prove information destruction, and successful probe decoding does not identify a learning mechanism.

\paragraph{Process and artifact controls.}
The three CPU processes execute sequentially and use the same allowed core, one inference thread, two discarded warmups, automatic GC disabled during rollouts, and explicit collection outside them. Seed and selector order are counterbalanced. Identical-input snapshots clone the same policy, recurrent/cache state and scheduler state before swapping only the selector; cloning is outside each measured harness call. Repeated processes are summarized separately rather than counted as 30 training seeds. The original host can still experience external load.

The exported code and evidence archives were also extracted into a clean temporary directory. All five intervention tests passed, the recomputed raw-audit JSON matched exactly, and a fresh smoke reproduced 28 states with 28 no-op checks. Archive hashes and command logs identify the tested exports. This checks execution outside the working tree on the same installed software environment, not an independently provisioned OS or public artifact availability.

'''+(root/'robustness_tables.tex').read_text()+'\n'
    anchor=r'\section{Reproducibility Contract and Implementation Details}'
    assert anchor in src;src=src.replace(anchor,appendix+'\n'+anchor,1)
    return src
