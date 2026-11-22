"""Reviewer checks from raw episodes, messages and training logs; no pooled-episode CI."""
import argparse,csv,json,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import yaml
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read(p):
    with p.open() as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(cfg):
    root=Path(cfg['source']);out=Path(cfg['output']);out.mkdir(exist_ok=False)
    (out/'registration.json').write_text(json.dumps(cfg,indent=2))
    rows=read(root/'by_seed.csv');conditions=['slow_5ms','medium_2ms','medium_5ms'];arms=['full','periodic','rr_cap','dac'];raw=[];sets={};hashes={};transition=[]
    for cond in conditions:
      for arm in arms:
       for seed in range(10):
        job=root/f'scratch_{cond}_seed{seed}_{arm}';cell=job/f'eval_{arm}_{cond}'
        ep=read(cell/'episodes.csv');v=np.array([float(x['episode_return']) for x in ep]);r=next(x for x in rows if x['family']=='scratch_'+cond and x['condition']==cond and x['arm']==arm and int(x['seed'])==seed)
        assert len(v)==50 and abs(v.mean()-float(r['mean_return']))<1e-10
        raw.append(dict(condition=cond,arm=arm,seed=seed,mean=float(v.mean()),partial_success=float(np.mean(v>=10)),complete_success=float(np.mean(v>=20)),**{'return_'+str(int(k)):int(np.sum(v==k)) for k in np.unique(v)}))
        d={(int(x['episode']),t):set() for x in ep for t in range(int(x['length']))}
        with (cell/'messages.jsonl').open() as f:
          for line in f:
            m=json.loads(line)
            if m['accepted']:d[(m['episode'],m['t'])].add(m['sender'])
        sets[cond,arm,seed]=d
        hashes[str(cell/'episodes.csv')]=sha(cell/'episodes.csv')
        train=[json.loads(x) for x in (job/'training.jsonl').read_text().splitlines()]
        transition.append(dict(condition=cond,arm=arm,seed=seed,updates=train[-1]['update'],online_transitions=train[-1]['online_transitions'],checkpoint_sha256=sha(job/'checkpoint_1500.pt')))
    comparisons=[];overlap=[]
    for cond in conditions:
      for left,right in [('dac','rr_cap'),('full','rr_cap'),('periodic','rr_cap')]:
        get=lambda a:np.array([next(r['mean'] for r in raw if (r['condition'],r['arm'],r['seed'])==(cond,a,s)) for s in range(10)])
        d=get(left)-get(right);mean=float(d.mean());se=float(stats.sem(d));df=9;crit=stats.t.ppf(.975,df);margin=cfg['equivalence_margin']
        if se>0:
          bca=stats.bootstrap((d,),np.mean,method='BCa',n_resamples=cfg['bootstrap_resamples'],rng=np.random.default_rng(cfg['seed'])).confidence_interval
          tost=max(stats.t.sf((mean+margin)/se,df),stats.t.cdf((mean-margin)/se,df))
        else:bca=(mean,mean);tost=0. if abs(mean)<margin else 1.
        draws=np.random.default_rng(cfg['seed']).choice(d,(cfg['bootstrap_resamples'],len(d)),replace=True).mean(1)
        comparisons.append(dict(condition=cond,left=left,right=right,mean=mean,seed_differences=d.tolist(),t95=[mean-crit*se,mean+crit*se],percentile95=np.quantile(draws,[.025,.975]).tolist(),bca95=list(bca),t90=[mean-stats.t.ppf(.95,df)*se,mean+stats.t.ppf(.95,df)*se],tost_p=float(tost),exploratory_margin=margin,approx_mde80=(crit+stats.norm.ppf(.8))*se,iqm_left=float(stats.trim_mean(get(left),.25)),iqm_right=float(stats.trim_mean(get(right),.25))))
        for seed in range(10):
          a,b=sets[cond,left,seed],sets[cond,right,seed];common=a.keys()&b.keys()
          overlap.append(dict(condition=cond,left=left,right=right,seed=seed,common_steps=len(common),union_steps=len(a.keys()|b.keys()),same_sets=sum(a[k]==b[k] for k in common),different_sets=sum(a[k]!=b[k] for k in common)))
    fig,axes=plt.subplots(1,3,figsize=(11,3.5),sharey=True)
    for ax,cond in zip(axes,conditions):
      for i,arm in enumerate(arms):
        v=[r['mean'] for r in raw if r['condition']==cond and r['arm']==arm]
        ax.scatter(np.linspace(i-.12,i+.12,10),v,s=25);ax.plot([i-.2,i+.2],[np.mean(v)]*2,color='black')
      ax.set_xticks(range(4),['Full','Periodic','RR','DAC']);ax.set_title(cond);ax.set_ylim(0,20);ax.set_ylabel('Seed mean return')
    fig.tight_layout();fig.savefig(out/'seed_distribution.pdf');fig.savefig(out/'seed_distribution.png',dpi=180)
    result=dict(raw_seed_results=raw,contrasts=comparisons,delivered_set_overlap=overlap,training_transitions=transition,hashes=hashes,notes=['TOST margin is post-hoc exploratory; not confirmatory equivalence.','MDE uses observed paired SD and normal power approximation; sensitivity descriptor, not observed power.','IQM trims floor(n*.25) observations from each tail: with n=10 this is 20% per tail, not exact IQM.','Common-step overlap is censored by episode length and does not imply equal states or policies.','Full/Periodic lambda=1; RR/DAC lambda=.9 in original harness.'])
    # Exact empirical IQM: integrate the central half of the empirical quantile function.
    for c in comparisons:
      for side in ['left','right']:
        v=sorted(r['mean'] for r in raw if r['condition']==c['condition'] and r['arm']==c[side]);n=len(v)
        c['iqm_'+side]=sum(x*max(0,min((i+1)/n,.75)-max(i/n,.25)) for i,x in enumerate(v))/.5
    result['notes'][2]='IQM integrates central 50% of ten seed means with fractional boundary weights; no episode pseudoreplication.'
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    lines=['# Reviewer raw evidence audit','', '120 matched-condition cells / 6,000 episode returns verified. Independent unit: training seed.','', '|Condition|Contrast|Difference|t 95% CI|BCa 95% CI|TOST p (±1, exploratory)|Approx. MDE80|','|---|---|---:|---|---|---:|---:|']
    for c in comparisons:lines.append(f"|{c['condition']}|{c['left']}−{c['right']}|{c['mean']:.3f}|{c['t95']}|{c['bca95']}|{c['tost_p']:.4f}|{c['approx_mde80']:.3f}|")
    lines+=['','## Delivered-set identity (common step support only)','|Condition|Comparison|Equal/common steps|','|---|---|---|']
    for cond in conditions:
      for l,r in [('full','rr_cap'),('periodic','rr_cap')]:
        z=[x for x in overlap if (x['condition'],x['left'],x['right'])==(cond,l,r)];lines.append(f"|{cond}|{l}/{r}|{sum(x['same_sets'] for x in z)}/{sum(x['common_steps'] for x in z)}|")
    lines+=['','## Interpretation','Full–RR is not a training-noise estimate: both weights and cache decay differ. Equal goodput alone does not imply set identity. Periodic/RR matching means are recomputed, not copied; checkpoint hashes and individual returns remain inspectable.','',*result['notes']]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(out/'REPORT.md')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();main(yaml.safe_load(Path(a.config).read_text()))
