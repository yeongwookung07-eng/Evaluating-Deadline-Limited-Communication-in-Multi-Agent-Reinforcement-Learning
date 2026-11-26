"""Render completed frozen-policy results; do not run during CPU timing campaigns."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def fmt(m,key='ci95',scale=1):
    return f"{m['mean']*scale:.2f} [{m[key][0]*scale:.2f}, {m[key][1]*scale:.2f}]"


def main(mnist,broadcast,out):
    mn=json.loads((mnist/'results.json').read_text());br=json.loads((broadcast/'summary.json').read_text())
    validation=json.loads((broadcast/'independent_validation.json').read_text())
    assert validation['mnist']['passed'] and validation['broadcast']['passed']
    out.mkdir(parents=True,exist_ok=True)
    p=mn['primary'];m=p['metrics'];b=br['primary']
    lines=['# Frozen-policy readiness and hard-action evaluation — 2026-11-26 KST','',
        'Completed user-requested CPU-only evaluations. No parameters were trained, adapted again or selected using these outcomes. Existing manuscript/PDFs are unchanged. Results below extend evidence only within the stated models.','',
        '## 1. MNIST-sum: readiness uncertainty and decision overhead','',
        'Three frozen 12,000-update both-bias checkpoints. Reused the earlier 1,024 test image pairs as a sensitivity cohort; reloaded weights and independently verified full/zero-message distributions and input hashes. Gaussian estimate errors are integrated exactly. All 450 registered settings are retained (1,350 seed cells; 1,382,400 example cells).','',
        'Primary: reception deadline 750 µs, unbiased estimate error SD 100 µs, sequential estimate/admission overhead 100 µs. Random admission has the same expected offered bytes and no incremental prediction overhead.','',
        '| Primary quantity | Result |','|---|---|',
        f"| Uncertain timing admission accuracy | {fmt(m['informed'],'t95',100)}% |",
        f"| Equally costly random admission accuracy | {fmt(m['random_no_extra_cost'],'t95',100)}% |",
        f"| Paired accuracy gain | {fmt(m['gain_practical_pp'],'t95')} percentage points |",
        f"| Expected offered bytes, both | {m['offered_B']['mean']:.4f} B/example |",
        f"| Expected timely payload, informed / random | {m['delivered_payload_B']['mean']:.4f} / {m['random_cheap_payload_B']['mean']:.4f} B/example |",
        f"| Uncertain admission minus Full without overhead | {fmt(m['informed_minus_full_pp'],'t95')} percentage points |",'',
        f"Predeclared practical primary gate: **{'PASS' if mn['practical_primary_pass'] else 'FAIL'}**. Positive means benefit over random at equal expected offered traffic, not preservation of the original Full policy. Exact Full–admit identity does not generally survive incorrect timing estimates.", '',
        'Unbiased error sensitivity at 750 µs (paired gain over zero-extra-overhead random, percentage points; descriptive 95% t intervals across three seeds):','',
        '| Error SD (µs) | Added decision time (µs) | Offered B/example | Paired gain |','|---:|---:|---:|---|']
    for sd in [0,100,400,800]:
        for cost in [0,100,200,400]:
            x=next(x for x in mn['summary'] if (x['deadline_us'],x['noise_sd_us'],x['bias_us'],x['decision_cost_us'])==(750,sd,0,cost))
            lines.append(f"| {sd} | {cost} | {x['metrics']['offered_B']['mean']:.3f} | {fmt(x['metrics']['gain_practical_pp'],'t95')} |")
    lines+=['','All five deadlines and all biases −200/0/+200 µs are in the JSON/NPZ artifacts. These are model sensitivities, not a trained predictor or measured radio. Decision overhead is charged on every decision; equal traffic does not equal equal CPU cost. Negative estimate values are allowed by the error model. Intervals across the grid are descriptive and unadjusted; no universal monotonic boundary is inferred.','',
        'The zero-error/zero-overhead control gives +9.88 pp against exact expected random admission. The earlier paper’s +10.59 pp used one finite random offer mask. Both use the same trained policies and image pairs; changing the random-control estimand explains the difference, not learning or checkpoint selection.','',
        '## 2. Broadcast: reject late actions during fresh rollouts','',
        'Ten frozen RR checkpoints and the previously saved true-label/shuffled-label receiver heads; 20 new resets per seed. Normal access uses measured readiness for actual simulated packet admission. Three serial independent CPU processes use the same reset pairs, two excluded warmup episodes per cell, and randomized cell order. Three heads × three fallback rules × three added postprocessing costs. Process repeats are averaged within training seed before n=10 intervals; they are not 30 independent training seeds.','',
        'Hold executes the previous actual joint action (initially zero); zero executes action 0. The actually executed action is fed to the next recurrent step. Ordinary executes the computed candidate even when late. Cache and recurrent state are retained, not rolled back. Every reward is generated by a new rollout.','',
        '| Primary normal hold, no added cost | Result |','|---|---|',
        f"| Original actual accuracy | {fmt(b['metrics']['original']['accuracy'],scale=100)}% |",
        f"| Adapted actual accuracy | {fmt(b['metrics']['adapted']['accuracy'],scale=100)}% |",
        f"| Shuffled-head actual accuracy | {fmt(b['metrics']['shuffled']['accuracy'],scale=100)}% |",
        f"| Adapted − original | {fmt(b['contrasts']['adapted_minus_original_pp'])} pp |",
        f"| Adapted − shuffled | {fmt(b['contrasts']['adapted_minus_shuffled_pp'])} pp |",
        f"| Original / adapted late candidate fraction | {b['metrics']['original']['miss_fraction']['mean']*100:.2f}% / {b['metrics']['adapted']['miss_fraction']['mean']*100:.2f}% |",'',
        f"Predeclared robust primary gate (positive paired lower bounds versus both controls in **each** process): **{'PASS' if br['primary_robust_gain_pass'] else 'FAIL'}**.",'',
        '| Process | Adapted − original (pp) | Adapted − shuffled (pp) |','|---|---|---|']
    for x in b['per_repeat']:lines.append(f"| {x['repeat']} | {fmt(x['adapted_minus_original_pp'])} | {fmt(x['adapted_minus_shuffled_pp'])} |")
    lines+=['','Normal access sensitivity (processes averaged within seed):','',
        '| Fallback | Added post cost µs | Original accuracy % | Adapted accuracy % | Adapted − original pp | Original / adapted misses % |',
        '|---|---:|---:|---:|---|---|']
    for x in br['cells']:
        if x['scheduler']!='normal':continue
        o=x['metrics']['original'];a=x['metrics']['adapted']
        lines.append(f"| {x['fallback']} | {x['post_cost_us']} | {o['accuracy']['mean']*100:.2f} | {a['accuracy']['mean']*100:.2f} | {fmt(x['contrasts']['adapted_minus_original_pp'])} | {o['miss_fraction']['mean']*100:.2f} / {a['miss_fraction']['mean']*100:.2f} |")
    lines+=['','Source-aware selection is privileged and remains a fixed-ready diagnostic; its clone cost is included in the action deadline. It is not pooled with measured normal access:','',
        '| Fallback | Original accuracy % | Adapted accuracy % | Adapted − original pp | Original / adapted misses % |','|---|---:|---:|---|---|']
    for x in br['cells']:
        if x['scheduler']!='source_aware':continue
        o=x['metrics']['original'];a=x['metrics']['adapted']
        lines.append(f"| {x['fallback']} | {o['accuracy']['mean']*100:.2f} | {a['accuracy']['mean']*100:.2f} | {fmt(x['contrasts']['adapted_minus_original_pp'])} | {o['miss_fraction']['mean']*100:.2f} / {a['miss_fraction']['mean']*100:.2f} |")
    lines+=['','Rejecting a late candidate does not make its computation timely. A preinstalled fallback is assumed ready at the deadline; this study does not benchmark actuator switching, continuous plant dynamics, backlog or overlapping control steps. Added post costs are modeled, not sleeps. Timings are instrumented on one host and can vary between policies and processes. The custom broadcast source bits are independent of actions, while recurrent policy/message histories depend on executed actions. Original adaptation used labels; no label-free improvement is established.','',
        '## Correctness, costs and evidence','',
        '- 120 historical fixed-readiness episodes exactly reproduced for actions/rewards/senders/timestamps/bytes; no-miss identity and forced-all-miss fallback checks passed.',
        '- All new hard-arm late candidates were rejected and actual-action feedback was asserted. Independent saved-trace audit recomputed reset bits, rewards, fallback rules, arrival cutoffs, traffic and seed-level intervals.',
        f"- Independent audit: {validation['broadcast']['episodes']:,} actual evaluation episodes / {validation['broadcast']['steps']:,} steps, plus excluded warmups and smoke. MNIST audit: {validation['mnist']['example_cells']:,} episode/setting/seed calculations.",
        '- Input checkpoint/head hashes and parameter tensors remained unchanged. Scientific tests: see validation record; no shared framework/training pipeline changes.','',
        f"MNIST artifact root: `{mnist}`. Broadcast artifact root: `{broadcast}`. Full JSON summaries, NPZ arrays, compressed step traces, host metadata, input hashes, wall times, smoke and independent validation remain available. Code/config/commands are in the component README.",'',
        'Next hypothesis: evaluate a measured readiness predictor or lower-cost action head under a separately registered budget. Current outcomes do not authorize additional training or claim standard-benchmark/hardware generalization.']
    lines += ['', 'Figure: [PNG](robustness.png) / [PDF](robustness.pdf). Points are evaluated settings; connecting lines guide the eye only. Bands are descriptive seed-level 95% intervals, not simultaneous confidence bands or evaluated interpolation. Raw t intervals on rates are not truncated to [0,1]. The all-zero fallback accuracy47.75% is conditional on these20paired resets; it is not a universal chance estimate.', '']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for sd in [0,100,400,800]:
        xs=[];ys=[];lo=[];hi=[]
        for cost in [0,50,100,200,400]:
            r=next(x for x in mn['summary'] if (x['deadline_us'],x['noise_sd_us'],x['bias_us'],x['decision_cost_us'])==(750,sd,0,cost))['metrics']['gain_practical_pp']
            xs.append(cost);ys.append(r['mean']);lo.append(r['t95'][0]);hi.append(r['t95'][1])
        axes[0].plot(xs,ys,marker='o',label=f'Error SD {sd} µs');axes[0].fill_between(xs,lo,hi,alpha=.1)
    axes[0].axhline(0,color='black',lw=.7);axes[0].set(xlabel='Added decision time (µs)',ylabel='Gain over equal-traffic random (pp)',title='MNIST-sum, 750 µs reception deadline');axes[0].legend(fontsize=8)
    for fb in ['ordinary','hold','zero']:
        xx=[x for x in br['cells'] if x['scheduler']=='normal' and x['fallback']==fb];xx.sort(key=lambda x:x['post_cost_us'])
        y=[x['contrasts']['adapted_minus_original_pp']['mean'] for x in xx];low=[x['contrasts']['adapted_minus_original_pp']['ci95'][0] for x in xx];high=[x['contrasts']['adapted_minus_original_pp']['ci95'][1] for x in xx]
        axes[1].plot([x['post_cost_us'] for x in xx],y,marker='o',label=fb);axes[1].fill_between([x['post_cost_us'] for x in xx],low,high,alpha=.1)
    axes[1].axhline(0,color='black',lw=.7);axes[1].set(xlabel='Added postprocessing time (µs)',ylabel='Adapted − original accuracy (pp)',title='Normal broadcast, 2 ms action deadline');axes[1].legend()
    fig.tight_layout();fig.savefig(out/'robustness.png',dpi=180);fig.savefig(out/'robustness.pdf');plt.close(fig)
    print(out/'RESULTS.md')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mnist',required=True);p.add_argument('--broadcast',required=True);p.add_argument('--out',required=True);a=p.parse_args();main(Path(a.mnist),Path(a.broadcast),Path(a.out))
