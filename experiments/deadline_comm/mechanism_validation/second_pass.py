"""Cluster-aware reanalysis of message interventions; no new rollouts."""
import csv,json,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr,pearsonr
from ..day2.io import atomic_json,CSVLog
ROOT=Path('experiments/deadline_comm/runs');SRC=ROOT/'mechanism_branches_20261119_001';OUT=ROOT/'deep_analysis_20261119_003'
def write(name,rows):
 with (OUT/name).open('w') as f:
  cols=list(rows[0]) if rows else ['empty'];w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
def main():
 OUT.mkdir(exist_ok=False);raw=[json.loads(x) for x in (SRC/'branches.jsonl').read_text().splitlines()]
 units=defaultdict(dict)
 for r in raw:
  k=(int(r['seed']),r['condition'],r['method'],int(r['episode']),int(r['t']),int(r['sender']))
  units[k][r['intervention']]=r
 assert len(units)==5685 and all(set(v)=={'zero','negate'} for v in units.values())
 # Paired intervention contrast within identical state/sender/episode.
 paired=[]
 for k,d in units.items():
  z,n=d['zero'],d['negate'];assert z['age']==n['age'] and z['baseline_return']==n['baseline_return']
  paired.append(dict(seed=k[0],condition=k[1],method=k[2],episode=k[3],t=k[4],sender=k[5],age=z['age'],q_gap=z['q_gap'],priority=z['priority'],norm=z['message_norm'],zero_delta=z['return_delta'],negate_delta=n['return_delta'],sign_minus_zero=float(n['return_delta'])-float(z['return_delta']),zero_flip=z['any_action_flip'],negate_flip=n['any_action_flip']))
 # episode is inferential cluster; report within-seed episode means, not branch-level p values.
 paired_by=defaultdict(list)
 for x in paired:paired_by[(x['seed'],x['condition'],x['method'])].append(x)
 cluster=[]
 for key,rr in sorted(paired_by.items()):
  eps=defaultdict(list)
  for x in rr:eps[x['episode']].append(x)
  vals=[np.mean([float(q['sign_minus_zero']) for q in z]) for z in eps.values()]
  cluster.append(dict(seed=key[0],condition=key[1],method=key[2],episodes=len(vals),mean_episode_delta=np.mean(vals),positive_episode_fraction=np.mean(np.array(vals)>0),negative_episode_fraction=np.mean(np.array(vals)<0),zero_episode_fraction=np.mean(np.array(vals)==0)))
 # Episode-heldout predictive check for absolute future return effect; four simple univariate signals.
 predictors=[]
 for method in ['dac','rr_cap']:
  for condition in ['slow_5ms','medium_2ms']:
   for kind in ['zero','negate']:
    for seed in range(3):
     rows=[r for r in raw if r['method']==method and r['condition']==condition and r['intervention']==kind and int(r['seed'])==seed]
     episodes=sorted({int(r['episode']) for r in rows});hold=set(episodes[::3]);train=[r for r in rows if int(r['episode']) not in hold];test=[r for r in rows if int(r['episode']) in hold]
     for feature in ['q_gap','priority','age','message_norm']:
      def score(a,b):
       x=np.array([float(r[feature]) for r in a]);y=np.array([abs(float(r['return_delta'])) for r in a]);u=np.array([float(r[feature]) for r in b]);v=np.array([abs(float(r['return_delta'])) for r in b])
       rho=float(spearmanr(u,v).statistic) if len(set(u))>1 and len(set(v))>1 else None
       # Baseline feature prediction uses one-dimensional OLS fit on training episodes, predicts held-out absolute effect.
       beta=np.polyfit(x,y,1) if len(set(x))>1 else [0,float(np.mean(y))]
       pred=np.polyval(beta,u);base=float(np.mean(y));mae=float(np.mean(abs(v-pred)));base_mae=float(np.mean(abs(v-base)))
       return rho,mae,base_mae
      rho,mae,bmae=score(train,test)
      predictors.append(dict(seed=seed,method=method,condition=condition,intervention=kind,feature=feature,train_episodes=len(set(int(r['episode']) for r in train)),test_episodes=len(hold),heldout_spearman=rho,heldout_mae=mae,heldout_base_mae=bmae,beats_constant=mae<bmae))
 # stratify paired message effects by factual step and age, preserving seed summaries.
 bins=[]
 for seed in range(3):
  for condition in ['slow_5ms','medium_2ms']:
   for method in ['dac','rr_cap']:
    for kind in ['zero','negate']:
     rr=[r for r in raw if r['method']==method and r['condition']==condition and r['intervention']==kind and int(r['seed'])==seed]
     for label,fn in [('fresh','age==0'),('age1','age==1'),('age2plus','age>=2'),('early_t0_2','t<=2'),('later_t3plus','t>=3')]:
      q=[r for r in rr if (int(r['age'])==0 if label=='fresh' else int(r['age'])==1 if label=='age1' else int(r['age'])>=2 if label=='age2plus' else int(r['t'])<=2 if label=='early_t0_2' else int(r['t'])>=3)]
      if q:bins.append(dict(seed=seed,condition=condition,method=method,intervention=kind,stratum=label,n=len(q),flip_rate=np.mean([r['any_action_flip'] for r in q]),benefit_rate=np.mean([float(r['return_delta'])>0 for r in q]),harm_rate=np.mean([float(r['return_delta'])<0 for r in q]),mean_delta=np.mean([float(r['return_delta']) for r in q])))
 summary=dict(source_branches=len(raw),paired_states=len(paired),paired_sign_zero_effects=cluster,episode_holdout_prediction=predictors,stratified_effects=bins,interpretation='All intervention contrasts are conditional on observed cached messages and policy trajectories. Episode-heldout splits use the same three trained checkpoints; they do not test generalization to new learned policies.')
 atomic_json(OUT/'summary.json',summary);write('paired_sign_vs_zero_by_seed.csv',cluster);write('heldout_prediction_by_seed.csv',predictors);write('stratified_effects_by_seed.csv',bins)
 atomic_json(OUT/'manifest.json',dict(status='completed',source_sha256=hashlib.sha256((SRC/'branches.jsonl').read_bytes()).hexdigest(),episodes_split_rule='heldout episode IDs every third episode, separately within each seed/method/condition/intervention',cluster_unit='episode',raw_rows_used=len(raw),paired_states=len(paired)))
 print(json.dumps({'paired':cluster,'holdout':predictors},indent=2))
if __name__=='__main__':main()
