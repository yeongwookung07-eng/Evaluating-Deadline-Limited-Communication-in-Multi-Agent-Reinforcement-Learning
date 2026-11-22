"""Deep read-only analyses of existing paired branches and coverage logs."""
import csv,json,hashlib
from pathlib import Path
from collections import defaultdict
import statistics,math
import numpy as np
from ..day2.io import atomic_json,CSVLog

R=Path('experiments/deadline_comm/runs');OUT=R/'deep_analysis_20261119_001'

def rd(path):
 with Path(path).open() as f:return list(csv.DictReader(f))
def mean(x):return statistics.mean(x) if x else None
def corr(x,y):
 if len(x)<3 or np.std(x)==0 or np.std(y)==0:return None
 return float(np.corrcoef(x,y)[0,1])
def paired_delta_ci(rows,key,seedkey='seed'):
 out=[]
 for seed in sorted({int(r[seedkey]) for r in rows}):
  v=[float(r[key]) for r in rows if int(r[seedkey])==seed]
  if v:out.append((seed,mean(v)))
 vals=[v for _,v in out]
 return dict(seed_means=out,mean_of_seed_means=mean(vals),min_seed_mean=min(vals) if vals else None,max_seed_mean=max(vals) if vals else None)

def main():
 OUT.mkdir(exist_ok=False); branches=[json.loads(x) for x in (R/'mechanism_branches_20261119_001/branches.jsonl').read_text().splitlines()]
 paired=rd(R/'discussion_evidence_20261119_001/paired_interventions.csv')
 assert len(branches)==11370 and len(paired)==5685
 # paired message manipulation by episode/state/sender: age, action, trajectory effect, sign-vs-removal
 pairmap={}
 for r in paired:pairmap[(int(r['seed']),r['condition'],r['method'],int(r['episode']),int(r['t']),int(r['sender']))]=r
 groups=defaultdict(list)
 for r in branches:groups[(r['seed'],r['condition'],r['method'],r['intervention'],r['age'])].append(r)
 age_rows=[]
 for k,rr in sorted(groups.items()):
  seed,cond,method,intervention,age=k;ds=[float(r['return_delta']) for r in rr]
  age_rows.append(dict(seed=seed,condition=cond,method=method,intervention=intervention,age=age,n=len(rr),action_flip=sum(r['any_action_flip'] for r in rr),changed=sum(x!=0 for x in ds),benefit=sum(x>0 for x in ds),harm=sum(x<0 for x in ds),mean_delta=mean(ds),median_delta=statistics.median(ds),mean_abs_delta=mean(map(abs,ds))))
 # action flip by whether outcome changes, and direction alignment
 action_rows=[]
 for method in ['dac','rr_cap']:
  for cond in ['slow_5ms','medium_2ms']:
   for intervention in ['zero','negate']:
    rr=[r for r in branches if (r['method'],r['condition'],r['intervention'])==(method,cond,intervention)]
    d=[float(r['return_delta']) for r in rr]
    flips=[bool(r['any_action_flip']) for r in rr]
    action_rows.append(dict(method=method,condition=cond,intervention=intervention,n=len(rr),flip_rate=mean(flips),return_changed_rate=mean(x!=0 for x in d),return_changed_given_flip=mean([x!=0 for x,f in zip(d,flips) if f]),return_changed_without_flip=mean([x!=0 for x,f in zip(d,flips) if not f]),benefit_rate=mean(x>0 for x in d),harm_rate=mean(x<0 for x in d),delta_mean=mean(d)))
 # Heldout episode split exploratory prediction: simple features age, gap, priority, qgap, norm, flip.
 features=[]
 for r in branches:
  d=pairmap[(int(r['seed']),r['condition'],r['method'],int(r['episode']),int(r['t']),int(r['sender']))]
  features.append(dict(seed=int(r['seed']),episode=int(r['episode']),method=r['method'],condition=r['condition'],kind=r['intervention'],age=int(r['age']),q_gap=float(r['q_gap']),priority=float(r['priority']),norm=float(r['message_norm']),flip=int(r['any_action_flip']),delta=float(r['return_delta']),absdelta=abs(float(r['return_delta'])),harm=int(float(r['return_delta'])<0),benefit=int(float(r['return_delta'])>0)))
 pred=[]
 for method in ['dac','rr_cap']:
  for cond in ['slow_5ms','medium_2ms']:
   for kind in ['zero','negate']:
    x=[r for r in features if (r['method'],r['condition'],r['kind'])==(method,cond,kind)]
    for feature in ['age','q_gap','priority','norm','flip']:
     pred.append(dict(method=method,condition=cond,intervention=kind,feature=feature,n=len(x),spearman=float(__import__('scipy').stats.spearmanr([r[feature] for r in x],[r['absdelta'] for r in x]).statistic) if len(set(r[feature] for r in x))>1 else None,pearson=corr([r[feature] for r in x],[r['absdelta'] for r in x])))
 # Coverage sender identities across reset seed and fixed step window.
 coverage=[]
 for mode in ['fixed_ready']:
  for cond in ['slow_5ms','medium_2ms','medium_5ms']:
   for seed in range(3):
    for method in ['rr_cap','dac']:
     ep=defaultdict(list)
     for p in (R/'q2_priority_20261118_001').glob('*_seed*_'+cond+'_'+mode+'_'+method+'/steps.csv'):
      # derive seed via summary to avoid brittle glob parsing
      meta=json.loads((p.parent/'summary.json').read_text());
      if meta['seed']!=seed:continue
      for r in rd(p):ep[int(r['episode'])].append(r)
     if not ep:continue
     for window in [1,2,3,4,5]:
      elig=[v for v in ep.values() if len(v)>=window]
      per=[]
      for rows in elig:
       r=sorted(rows,key=lambda a:int(a['t']))[window-1];src=json.loads(r['source_steps']);n=len(src);per.append(sum(x<0 for x in src)/n)
      coverage.append(dict(seed=seed,condition=cond,method=method,window=window,n_episodes=len(elig),invalid_fraction=mean(per)))
 # exact cap load distribution by seed
 load=[]
 for r in rd(R/'discussion_evidence_20261119_001/same_cap_actual_load.csv'):load.append(r)
 # Main concise summary.
 summ=dict(branches=len(branches),paired_sender_interventions=len(paired),action_sensitivity=action_rows,age_strata=age_rows,exploratory_predictors=pred,matched_window_coverage=coverage,load=load,limitations=['Counterfactuals branch from factual visited states and are correlated.','Age/policy-feature correlations are exploratory; split is by episode but only3 checkpoint seeds.','Coverage windows condition on episode survival; state/action paths differ after t=0.','This validates no causal mediator between access policy and return.'])
 atomic_json(OUT/'summary.json',summ)
 for name,rows in [('age_strata.csv',age_rows),('action_effects.csv',action_rows),('feature_predictors.csv',pred),('matched_coverage.csv',coverage)]:
  log=CSVLog(OUT/name);log.append(rows);log.close()
 (OUT/'manifest.json').write_text(json.dumps({'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'mechanism_branches_20261119_001/branches.jsonl',R/'discussion_evidence_20261119_001/paired_interventions.csv',R/'discussion_evidence_20261119_001/summary.json']},'status':'completed'},indent=2)+'\n')
 print(json.dumps({'action':action_rows,'pred':pred,'coverage':coverage},indent=2))
if __name__=='__main__':main()
