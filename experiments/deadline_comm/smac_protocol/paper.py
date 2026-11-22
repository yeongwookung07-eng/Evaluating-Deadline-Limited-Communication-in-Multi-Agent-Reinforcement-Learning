"""Insert only audited completed SMACv2 evidence into the active KBS version."""
import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from ..common import load_config
from ..day2.io import atomic_json
from .audit import audit


def section(spec,summary,gate,identities):
    labels={'ideal_full':'Ideal Full','no_comm':'No-Comm','full':'Full','rr_cap':'RR','dac':'DAC'}
    rows=[]
    for cell in summary['means']:
        if cell['mode']!='fixed':continue
        cond='Slow/5 ms' if cell['condition']=='slow_5ms' else 'Medium/2 ms'
        rows.append(f"{cond} & {labels[cell['arm']]} & ${cell['return_mean']:.2f}\\pm{cell['return_mean_sd']:.2f}$ & {100*cell['win_rate']:.1f} & {cell['offered_b_per_step']:.2f} & {cell['goodput_b_per_step']:.2f}"+r'\\')
    g=', '.join(f"{100*r['win_rate']:.1f}\\%" for r in gate['records'])
    s=r'''\subsection{Delivery attribution on a learned SMACv2 policy}\label{sec:smac_transfer}
We apply the delivery-attribution protocol to procedural SMACv2 Terran 5-vs-5~\citep{ellis2023smacv2}. Three Ideal Full policies continue the earlier approximately 100,000-step pilot, without Oracle data or behavior cloning. The registered common learning checkpoints are 300,000, 600,000 and 1,000,000 total environment steps. All three seeds first meet the 20\% win-rate gate at MILESTONE steps, with gate win rates GATEWINS on 64 resets. This is a feasibility threshold, not a convergence criterion. Online/target weights and optimizer states are retained, but replay is refilled; exploration restarts once at 0.30 and decays to 0.05 over 200,000 additional steps.

Evaluation uses HELDOUT disjoint held-out resets per seed. All deployments share each seed's weights, cache decay $\lambda=1$ and 48-byte packet size; they are zero-shot access interventions, not independently trained scheduler algorithms. Table~\ref{tab:smac_transfer} reports fixed 0.5 ms readiness with the retained 0.45 ms post-reception reserve. Measured-readiness sensitivity and raw CPU components accompany the results. Game simulation time is excluded from the modeled communication deadline.

\begin{table*}[pos=t]\centering\small
\caption{SMACv2 Terran held-out zero-shot deployments of the same Ideal-trained checkpoints. Return is mean $\pm$ sample SD over all three training seeds; wins are percentages. Traffic is averaged per executed step within each seed, then across seeds. Three seeds provide limited statistical precision.}\label{tab:smac_transfer}
\begin{tabular}{llllrr}\toprule
Condition & Access & Return & Wins (\%) & Wire B/step & Payload B/step\\\midrule
ROWS
\bottomrule\end{tabular}
\end{table*}

'''.replace('MILESTONE',f"{gate['milestone']:,}").replace('GATEWINS',g).replace('ROWS','\n'.join(rows)).replace('HELDOUT',str(spec['evaluation_episodes']))
    for cond in spec['evaluation']['conditions']:
        rr=[r for r in identities if r['condition']==cond];passed=sum(r['passed'] for r in rr)
        s+=f"For {('Slow/5 ms' if cond=='slow_5ms' else 'Medium/2 ms')}, Full--RR shadow replay preserves all delivered sender sets, cache values/timestamps, hidden states and actions in {passed} of {len(rr)} episodes. "
    s+=r'''Each shadow has independent policy and scheduler state and receives the factual native environment input. Only complete action/history identity supports an unchanged-trajectory conclusion; after a mismatch, no alternative suffix return is imputed. Separate full-episode repeats in fresh native engines pass for all three checkpoint checks. This transfers the delivery control to a standard benchmark without assuming that Hallway's identity result must recur.

'''
    for contrast in ['ideal_full-no_comm','dac-rr_cap']:
        rr=[r for r in summary['paired'] if r['mode']=='fixed' and r['contrast']==contrast and r['metric']=='return_mean']
        name='Ideal--No-Comm' if contrast=='ideal_full-no_comm' else 'DAC--RR'
        s+=name+' paired seed-mean return differences are '+', '.join(f"{r['mean']:+.2f} [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}] ({'Slow/5 ms' if r['condition']=='slow_5ms' else 'Medium/2 ms'})" for r in rr)+r'''. Brackets are exploratory, unadjusted percentile-bootstrap 95\% intervals resampling three training seeds; they do not establish equivalence. '''
    s+=r'''No-Comm here tests removal of messages from an Ideal-trained policy, not the best achievable independently trained no-message controller. The additional evidence covers one scenario and delivery attribution, not all components of the protocol or universal scheduler superiority.

'''
    return s


def run(spec,install=True):
    root=Path(spec['output']);audit(spec)
    gate=json.loads((root/'learning_gate.json').read_text());summary=json.loads((root/'evaluation/summary.json').read_text());identities=json.loads((root/'evaluation/identity.json').read_text())
    body=section(spec,summary,gate,identities);(root/'SMACV2_RESULTS.tex').write_text(body)
    if not install:return
    p=Path('experiments/deadline_comm/publication/kbs');f=p/'kbs_main.tex';manifest=json.loads((root/'manifest.json').read_text())
    if hashlib.sha256(f.read_bytes()).hexdigest()!=manifest['paper_source_sha256']:
        atomic_json(root/'paper_status.json',dict(stage='addendum_ready_source_changed',reason='KBS source changed during campaign; no overwrite',addendum=str(root/'SMACV2_RESULTS.tex')));return
    original=f.read_text();oldbib=(p/'references.bib').read_text();oldreadme=(p/'README.md').read_text()
    with zipfile.ZipFile(root/'kbs_before_smacv2.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['kbs_main.tex','kbs_main.pdf','references.bib','README.md']:z.write(p/name,name)
    marker='\\section{Discussion:';assert original.count(marker)==1
    updated=original.replace(marker,body+marker)
    updated=updated.replace('standard-benchmark replication remains open.','an additional three-seed SMACv2 study tests delivery attribution on a standard task.')
    updated=updated.replace('larger standard tasks are needed to measure their prevalence beyond this setting.','a bounded SMACv2 extension tests delivery attribution beyond these custom tasks (Section~\\ref{sec:smac_transfer}).')
    updated=updated.replace('The demonstrated scope is Oracle-assisted five-agent Hallway and an Oracle-free custom broadcast task.','The evidence covers Oracle-assisted five-agent Hallway, an Oracle-free custom broadcast task, and the bounded SMACv2 Terran delivery study in Section~\\ref{sec:smac_transfer}.')
    updated=updated.replace('A separate SMACv2 pilot also fails its preset criterion','The earlier 100,000-step SMACv2 pilot failed its preset criterion')
    updated=updated.replace('so no scheduler comparisons followed.','so that pilot did not trigger scheduler comparisons. The separately registered continuation reported here passes the common learning gate before held-out deployment evaluation.')
    start='The broadcast probe extends the delivery--use diagnosis beyond Hallway, but does not replicate its full delivery-identity and subset-enumeration results on an external benchmark.'
    stop='Failure of that learning gate supplies no evidence about scheduler quality.'
    if start in updated:
        a=updated.index(start);b=updated.index(stop,a)+len(stop)
        updated=updated[:a]+'Broadcast tests downstream use; SMACv2 tests the delivery-identity control. Exhaustive sender-subset and common-prefix effects remain specific to Hallway in this study. The failed earlier learning gate itself supplies no evidence about scheduler quality.'+updated[b:]
    updated=updated.replace('with standard-benchmark replication as the outstanding test of transfer.','with a bounded SMACv2 delivery study and broader task/network transfer still open.')
    reference=r'''
@misc{ellis2023smacv2,
 author={Ellis, Benjamin and Cook, Jonathan and Moalla, Skander and Samvelyan, Mikayel and Sun, Mingfei and Mahajan, Anuj and Foerster, Jakob N. and Whiteson, Shimon},
 title={{SMACv2}: An Improved Benchmark for Cooperative Multi-Agent Reinforcement Learning},
 year={2023},note={arXiv:2212.07489v2},url={https://arxiv.org/abs/2212.07489}}
'''
    f.write_text(updated)
    if 'ellis2023smacv2' not in oldbib:(p/'references.bib').write_text(oldbib+reference)
    (p/'README.md').write_text(oldreadme+'\nAudited SMACv2 delivery-attribution extension added from `'+str(root)+'`; source checksums and gate/evaluation/raw audit retained there. No prior experimental rows replaced.\n')
    try:
        with (root/'paper_build.log').open('w') as log:subprocess.run([sys.executable,str(p/'build.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
        validation=json.loads((p/'validation.json').read_text());assert validation['pages']<=15,validation
        atomic_json(root/'paper_status.json',dict(stage='updated',pages=validation['pages'],pdf=str(p/'kbs_main.pdf')))
    except Exception:
        f.write_text(original);(p/'references.bib').write_text(oldbib);(p/'README.md').write_text(oldreadme)
        subprocess.run([sys.executable,str(p/'build.py')],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT,check=True)
        atomic_json(root/'paper_status.json',dict(stage='addendum_ready_layout_failed',addendum=str(root/'SMACV2_RESULTS.tex')));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();run(load_config(a.config))
