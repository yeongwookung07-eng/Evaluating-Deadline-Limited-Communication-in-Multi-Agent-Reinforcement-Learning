"""Seed-level analysis of registered reviewer interventions; no outcome filtering."""
import json,statistics,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import t as student
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path('experiments/deadline_comm');HERE=BASE/'tmlr_revision';OUT=HERE/'results'

def load(name):
 p=BASE/'runs'/f'tmlr_{name}_20261119_001';assert json.loads((p/'status.json').read_text())['stage']=='completed',name;return p

def ci(x):
 a=np.array(x,float);n=len(a);m=float(a.mean());se=float(a.std(ddof=1)/np.sqrt(n)) if n>1 else 0.;q=student.ppf(.975,n-1) if n>1 else 0.
 return dict(mean=m,sd=float(a.std(ddof=1)) if n>1 else None,n=n,lo=m-q*se,hi=m+q*se)
def lines(p):
 with p.open() as f:
  for line in f:yield json.loads(line)
def main():
 OUT.mkdir(exist_ok=True);fresh=load('fresh');prefix=load('prefix');broadcast=load('broadcast');ready=load('readiness');groups=defaultdict(list);phase_groups=defaultdict(list);fresh_count=0
 for r in lines(fresh/'states.jsonl'):
  vals=[v['return_'] for v in r['variants']];best=max(vals);gap=np.array(r['gap']);age=np.array(r['age_term']);base=r['normal_senders'];alts={str(a['alpha']):a['regret'] for a in r['rankings']}
  row=dict(withhold_delta=float(np.mean([v['delta'] for v in r['withhold']])),withhold_absolute_delta=float(np.mean([abs(v['delta']) for v in r['withhold']])),withhold_changed_fraction=float(np.mean([v['delta']!=0 for v in r['withhold']])),range=r['range_return'],normal_regret=best-r['baseline_return'],any_return_effect=float(max(vals)!=min(vals)),action_flip_rate=float(np.mean([v['action_flip'] for v in r['variants']])),gap_mean=float(gap.mean()),age_term_mean=float(age.mean()),age_dominates=float(np.mean(age>gap)),normalized_regret=r['normalized_regret'],scale_changes=float(r['reward_scale_rankings']['0.1']!=r['reward_scale_rankings']['10.0']),**{'alpha_'+k:v for k,v in alts.items()})
  groups[r['condition'],r['checkpoint_arm'],r['seed']].append(row);phase_groups[r['condition'],r['checkpoint_arm'],r['seed'],'initial' if r['initial_invalid']>0 else 'all_valid'].append(row);fresh_count+=1
 fseed=[dict(condition=k[0],checkpoint_arm=k[1],seed=k[2],states=len(v),**{m:float(np.mean([x[m] for x in v])) for m in v[0]}) for k,v in groups.items()]
 phase_seed=[dict(condition=k[0],checkpoint_arm=k[1],seed=k[2],phase=k[3],states=len(v),range=float(np.mean([x['range'] for x in v])),normal_regret=float(np.mean([x['normal_regret'] for x in v]))) for k,v in phase_groups.items()]
 pg=defaultdict(list)
 for r in lines(prefix/'episodes.jsonl'):
  pg[r['condition'],r['checkpoint_arm'],r['seed']].append(dict(cold_delta=r['originals']['dac']['return_']-r['originals']['rr_cap']['return_'],warm_delta=r['paired']['dac']['suffix_return']-r['paired']['rr_cap']['suffix_return'],survived=float(r['survived']),prefix_steps=len(r['prefix']),prefix_wire=r['prefix_wire'],prefix_model_ns=r['prefix_model_ns']))
 pseed=[]
 for k,v in pg.items():
  survivors=[x['warm_delta'] for x in v if x['survived']]
  pseed.append(dict(condition=k[0],checkpoint_arm=k[1],seed=k[2],episodes=len(v),cold_delta=float(np.mean([x['cold_delta'] for x in v])),warm_unconditional_delta=float(np.mean([x['warm_delta'] for x in v])),warm_survivor_delta=float(np.mean(survivors)) if survivors else None,survivor_count=len(survivors),warm_mean_absolute_delta=float(np.mean([abs(x['warm_delta']) for x in v])),warm_changed_fraction=float(np.mean([x['warm_delta']!=0 for x in v])),prefix_steps=float(np.mean([x['prefix_steps'] for x in v])),prefix_wire=float(np.mean([x['prefix_wire'] for x in v])),prefix_model_ns=float(np.mean([x['prefix_model_ns'] for x in v]))))
 bseed=json.loads((broadcast/'summary.json').read_text())['records'];rseed=json.loads((ready/'summary.json').read_text())['records']
 result=dict(fresh_states=fresh_count,fresh_seed=fseed,phase_seed=phase_seed,prefix_seed=pseed,broadcast_seed=bseed,readiness_seed=rseed,unit='training seed; unadjusted descriptive t intervals; no selection of successful checkpoints')
 (OUT/'summary.json').write_text(json.dumps(result,indent=2));tables=[];report=['# TMLR reviewer experiment results','', 'All intervals use independent training-seed summaries. Branch/episode counts are not the inference sample size.','']
 def table(title,headers,rows):
  report.extend([title,'|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|',*['|'+'|'.join(r)+'|' for r in rows],''])
 fr=[];pr=[]
 for cond in ['slow_5ms','medium_2ms']:
  for arm in ['rr_cap','dac']:
   v=[r for r in fseed if (r['condition'],r['checkpoint_arm'])==(cond,arm)];assert len(v)==10
   z=ci([r['range'] for r in v]);fr.append([cond,arm,f"{z['mean']:.3f} [{z['lo']:.3f},{z['hi']:.3f}]",f"{np.mean([r['normal_regret'] for r in v]):.3f}",f"{100*np.mean([r['any_return_effect'] for r in v]):.1f}"])
   v=[r for r in pseed if (r['condition'],r['checkpoint_arm'])==(cond,arm)];z=ci([r['cold_delta'] for r in v]);w=ci([r['warm_unconditional_delta'] for r in v]);pr.append([cond,arm,f"{z['mean']:+.3f}",f"{w['mean']:+.3f} [{w['lo']:.3f},{w['hi']:.3f}]",str(sum(r['survivor_count'] for r in v))+'/'+str(sum(r['episodes'] for r in v))])
 table('## Fresh feasible sender subsets',['Condition','Weights','Mean best-worst suffix range [95% CI]','Normal selection hindsight regret','States with return effect (%)'],fr)
 table('## Shared-prefix control',['Condition','Weights','Cold DAC-RR','Prefix DAC-RR total [95% CI]','Prefix survivors'],pr)
 ar=[]
 for cond in ['slow_5ms','medium_2ms']:
  for arm in ['rr_cap','dac']:
   v=[r for r in fseed if r['condition']==cond and r['checkpoint_arm']==arm]
   ar.append([cond,arm,*[f"{np.mean([r[k] for r in v]):.3f}" for k in ['gap_mean','age_term_mean','age_dominates','alpha_0.0','alpha_0.03','alpha_0.1','alpha_0.3','alpha_1.0','normalized_regret','scale_changes']]])
 table('## Score scale and hindsight regret',['Condition','Weights','Mean gap','Mean age term','Age dominates fraction','alpha0 regret','alpha.03','alpha.1','alpha.3','alpha1','Normalized regret','Rescaling rank change'],ar)

 wr=[]
 for cond in ['slow_5ms','medium_2ms']:
  for arm in ['rr_cap','dac']:
   v=[r for r in fseed if r['condition']==cond and r['checkpoint_arm']==arm];z=ci([r['withhold_delta'] for r in v])
   wr.append([cond,arm,f"{z['mean']:+.3f} [{z['lo']:.3f},{z['hi']:.3f}]",f"{np.mean([r['withhold_absolute_delta'] for r in v]):.3f}",f"{100*np.mean([r['withhold_changed_fraction'] for r in v]):.1f}"])
 table('## Withholding an actual fresh update',['Condition','Weights','Withhold minus normal Return [95% CI]','Mean absolute effect','Changed branch %'],wr)
 report.append('Phase-stratified observational summaries are retained in JSON; conditioning on all-valid cache is not the common-prefix causal intervention and may select different trajectories.')
 br=[]
 for cond in ['slow_5ms','medium_2ms','medium_5ms']:
  for method in ['no_comm','ideal_full','periodic','rr_cap','dac']:
   for sched in ['normal','source_aware']:
    rr=[r for r in bseed if (r['condition'],r['method'],r['scheduler'])==(cond,method,sched)];assert len(rr)==10
    br.append([cond,method,sched,*[f"{100*np.mean([r[k] for r in rr]):.2f}" for k in ['q','accuracy','reference','linear_probe_accuracy']]])
 table('## Broadcast pipeline',['Condition','Weights','Access','Source delivery %','Accuracy %','Reference %','Probe test %'],br)
 bp=[]
 for cond in ['slow_5ms','medium_2ms','medium_5ms']:
  for method in ['periodic','rr_cap','dac']:
   get=lambda sched:{r['seed']:r for r in bseed if (r['condition'],r['method'],r['scheduler'])==(cond,method,sched)}
   normal,aware=get('normal'),get('source_aware');z=ci([100*(aware[i]['accuracy']-normal[i]['accuracy']) for i in range(10)])
   bp.append([cond,method,f"{z['mean']:+.2f} [{z['lo']:.2f},{z['hi']:.2f}]",f"{100*np.mean([aware[i]['q'] for i in range(10)]):.2f}"])
 table('## Paired source-aware access effects',['Condition','Weights','Accuracy change percentage points [95% CI]','Source-aware current delivery %'],bp)

 rr=[]
 for arm in ['rr_cap','dac']:
  for d in [1.5,1.75,2.,2.25,2.5]:
   for mode in ['measured','fixed']:
    v=[r for r in rseed if (r['checkpoint_arm'],r['deadline_ms'],r['mode'])==(arm,d,mode)];assert len(v)==10
    rr.append([arm,str(d),mode,*[f"{np.mean([r[k] for r in v]):.3f}" for k in ['mean_return','mean_k','ready_us','gate_us']]])
 table('## Readiness boundary',['Weights','Deadline ms','Mode','Return','K delivered','Readiness us','Gate us'],rr)
 paired=[]
 for arm in ['rr_cap','dac']:
  for d in [1.5,1.75,2.,2.25,2.5]:
   get=lambda mode:{r['seed']:r for r in rseed if (r['checkpoint_arm'],r['deadline_ms'],r['mode'])==(arm,d,mode)}
   m,f=get('measured'),get('fixed');v=[m[seed]['mean_return']-f[seed]['mean_return'] for seed in range(10)];z=ci(v)
   paired.append([arm,str(d),f"{z['mean']:+.3f} [{z['lo']:.3f},{z['hi']:.3f}]",f"{np.mean([m[i]['mean_k']-f[i]['mean_k'] for i in range(10)]):+.3f}"])
 table('## Paired readiness effects',['Weights','Deadline ms','Measured minus fixed Return [95% CI]','K difference'],paired)

 report+=['The warm-prefix unconditional effect includes zero suffix effects for episodes terminating before branching. Survivor-only effects are retained separately in JSON. Neither is a pure coverage-mediated effect: payload, aggregation denominator and validity features jointly change.','Source-aware fixes the realized packet count at the visited state; it can change later histories. True-bit reference ignores signaling through silence. The probe tests linear decodability, not optimal decoding.','Alpha/normalized/reward-scale diagnostics use every sampled state; they are explanatory comparisons, not a tuned winning scheduler.','Current CPU is single-host data. Fixed-readiness measurements are not deployable latency.']
 (OUT/'REPORT.md').write_text('\n'.join(report)+'\n')
 # Plot primary checkpoint diagnostics, not historical cohorts.
 fig,axs=plt.subplots(1,3,figsize=(12,3.4))
 for i,arm in enumerate(['rr_cap','dac']):
  for j,c in enumerate(['slow_5ms','medium_2ms']):
   v=[r['range'] for r in fseed if r['condition']==c and r['checkpoint_arm']==arm];axs[0].scatter(np.linspace(i*2+j-.12,i*2+j+.12,10),v)
 axs[0].set_xticks(range(4),['RR slow','RR med','DAC slow','DAC med'],rotation=25);axs[0].set_ylabel('Mean suffix return range per seed')
 for arm in ['rr_cap','dac']:
  v=[r for r in pseed if r['condition']=='slow_5ms' and r['checkpoint_arm']==arm];axs[1].scatter([r['cold_delta'] for r in v],[r['warm_unconditional_delta'] for r in v],label=arm)
 axs[1].axhline(0,color='grey',lw=.5);axs[1].axvline(0,color='grey',lw=.5);axs[1].set_xlabel('Cold DAC minus RR return');axs[1].set_ylabel('Common-prefix DAC minus RR return');axs[1].legend()
 for arm in ['rr_cap','dac']:
  for mode,ls in [('measured','-'),('fixed','--')]:
   xs=sorted({r['deadline_ms'] for r in rseed});ys=[np.mean([r['mean_k'] for r in rseed if r['checkpoint_arm']==arm and r['mode']==mode and r['deadline_ms']==d]) for d in xs];axs[2].plot(xs,ys,ls,label=arm+' '+mode)
 axs[2].set_xlabel('Deadline (ms)');axs[2].set_ylabel('Delivered packets / step');axs[2].legend(fontsize=7)
 axs[0].set_title('Primary checkpoint subsets');axs[1].set_title('Slow / 5 ms');axs[2].set_title('Medium; weights trained at 2 ms')
 fig.tight_layout();fig.savefig(OUT/'diagnostics.pdf');fig.savefig(OUT/'diagnostics.png',dpi=180)
 # compact LaTeX tables kept separate for the manuscript builder
 tex=[]
 for label,title,heads,rs in [('fresh','Feasible fresh-message interventions on primary checkpoints. Range and regret are hindsight diagnostics.', ['Condition','Weights','Range [95\\% CI]','Regret','Effect \\%'],fr),('prefix','Shared-prefix effects, including early termination in total-return contrasts.', ['Condition','Weights','Cold diff.','Prefix diff. [95\\% CI]','Survivors'],pr)]:
  esc=lambda x:x.replace('_',r'\_')
  tex.extend([r'\begin{table}[htbp]\centering\scriptsize',r'\caption{'+title+r'}\label{tab:new_'+label+'}',r'\begin{tabular}{lllll}\toprule',' & '.join(heads)+r'\\\midrule',*[' & '.join(esc(x) for x in r)+r'\\' for r in rs],r'\bottomrule\end{tabular}\end{table}'])
 (OUT/'tables.tex').write_text('\n'.join(tex)+'\n');print(OUT/'REPORT.md')
if __name__=='__main__':main()
