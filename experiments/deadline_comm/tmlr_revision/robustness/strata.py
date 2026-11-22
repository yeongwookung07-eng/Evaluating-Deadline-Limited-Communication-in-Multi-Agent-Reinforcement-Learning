"""Same factual trajectories; pre-treatment strata and paired seed inference."""
import json,time,hashlib,itertools
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from scipy.stats import t as student
from ...common import load_config
from ...day2.io import atomic_json
from ...adapters.hallway import Hallway
from ...day2.finetune_comm_aware import load_model
from ...selection_validation.harness import SelectionHarness
from ...timing import configure_cpu
from ..experiments import setup,step,BASE
from ..editorial_audit import lines

def estimate(x,rng,draws):
    x=np.asarray(x,float);n=len(x);m=float(x.mean())
    if n<2:return dict(n=n,mean=m,t95=None,bootstrap95=None)
    d=float(student.ppf(.975,n-1)*x.std(ddof=1)/np.sqrt(n))
    b=x[rng.integers(0,n,size=(draws,n))].mean(1)
    return dict(n=n,mean=m,t95=[m-d,m+d],bootstrap95=np.quantile(b,[.025,.975]).tolist(),seed_values=x.tolist())

def aggregate(rows,s):
    rng=np.random.default_rng(s['bootstrap_seed']);groups=defaultdict(list);output=[]
    for r in rows:groups[r['condition'],r['arm']].append(r)
    filters={'all':lambda r:True,'unresolved':lambda r:r['unresolved']>0,'resolved':lambda r:r['unresolved']==0,'no_prior_penalty':lambda r:not r['prior_penalty'],'episode_without_penalty':lambda r:not r['episode_penalty'],'early_t0_2':lambda r:r['t']<=2,'middle_t4_6':lambda r:4<=r['t']<=6,'late_t8_plus':lambda r:r['t']>=8,'one_unresolved':lambda r:r['unresolved']==1,'two_unresolved':lambda r:r['unresolved']==2}
    for (cond,arm),rr in groups.items():
        for stratum,predicate in filters.items():
            vv=[r for r in rr if predicate(r)];byseed=defaultdict(list)
            for r in vv:byseed[r['seed']].append(r)
            if not vv:
                output.append(dict(condition=cond,arm=arm,stratum=stratum,states=0,contributing_seeds=0));continue
            result=dict(condition=cond,arm=arm,stratum=stratum,states=len(vv),contributing_seeds=len(byseed),seed_ids=sorted(byseed))
            for weighting in ['state','episode']:
                for metric in ['effect','range','factual_minus_uniform']:
                    means=[]
                    for seed,v in sorted(byseed.items()):
                        if weighting=='state':means.append(np.mean([r[metric] for r in v]))
                        else:
                            byep=defaultdict(list)
                            for r in v:byep[r['episode']].append(r[metric])
                            means.append(np.mean([np.mean(w) for w in byep.values()]))
                    result[weighting+'_'+metric]=estimate(means,rng,s['bootstrap_resamples'])
            output.append(result)
    return output

def main():
    configure_cpu();s=load_config(BASE/'configs/tmlr_hallway_sensitivity.yaml');out=Path(s['output']);out.mkdir(exist_ok=False);atomic_json(out/'registration.json',s);start=time.monotonic()
    source=Path(s['source']);factual=defaultdict(list)
    for r in lines(source/'states.jsonl'):factual[r['seed'],r['condition'],r['checkpoint_arm'],r['episode']].append(r)
    contexts=[];episodes=[]
    with torch.inference_mode():
        for (seed,cond,arm),keys in itertools.groupby(sorted(factual),lambda k:k[:3]):
            ck=BASE/f'runs/reviewer_ten_seeds_20261118_001/scratch_{cond}_seed{seed}_{arm}/checkpoint_1500.pt';model,meta=load_model(ck);model.eval();sha=hashlib.sha256(ck.read_bytes()).hexdigest();link,d=cond.split('_');cfg=setup([link,float(d[:-2])])
            for key in keys:
                assert time.monotonic()-start<s['wall_seconds'];ep=key[-1];env=Hallway(**meta['environment']);obs,_=env.reset(seed=14000000+ep);h=SelectionHarness(model,arm,cfg,ep,False);trace=[];state_meta=[];penalty=False
                for t in range(env.episode_length):
                    state_meta.append(dict(unresolved=int(sum(env.env.active_group)),prior_penalty=penalty,remaining_horizon=env.episode_length-t))
                    obs,tr=step(env,h,obs,t);trace.append(tr);penalty|=tr['reward']<0
                    if tr['done']:break
                episodes.append(dict(seed=seed,condition=cond,arm=arm,episode=ep,steps=len(trace),penalty=bool(penalty),resolved_steps=sum(x['unresolved']==0 for x in state_meta),return_=sum(x['reward'] for x in trace)))
                for r in factual[key]:
                    t=r['t'];assert r['source_checkpoint_sha256']==sha
                    assert trace[t]['senders']==r['normal_senders'] and sum(x['reward'] for x in trace[t:])==r['baseline_return']
                    values=[v['return_'] for v in r['variants']]
                    contexts.append(dict(seed=seed,condition=cond,arm=arm,episode=ep,t=t,**state_meta[t],episode_penalty=bool(penalty),effect=float(max(values)>min(values))*100,range=max(values)-min(values),factual_minus_uniform=float(r['baseline_return']-np.mean(values)),candidate_count=len(values)))
                atomic_json(out/'status.json',dict(stage='running',episodes=len(episodes),states=len(contexts)))
    with (out/'states.jsonl').open('w') as f:
        for r in contexts:f.write(json.dumps(r)+'\n')
    atomic_json(out/'episodes.json',episodes)
    atomic_json(out/'summary.json',dict(verified_episodes=len(episodes),verified_states=len(contexts),strata=aggregate(contexts,s),scope='No-prior-penalty and unresolved strata are pre-treatment; episode-without-penalty is a retrospective sensitivity check, not randomized causal conditioning. Empty/missing seed strata are reported, never silently imputed.'))
    atomic_json(out/'status.json',dict(stage='completed'));print(out)
if __name__=='__main__':main()
