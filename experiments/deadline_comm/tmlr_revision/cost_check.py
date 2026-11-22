"""Validate reconstructed attempted traffic by replay of saved broadcast episodes."""
from copy import deepcopy
from pathlib import Path
import json,time
import torch
from ..common import load_config
from ..timing import configure_cpu
from ..day2.policy import DecayedPolicy
from ..reviewer_validation.hidden_source import HiddenSource
from ..reviewer_validation.train_ten_seeds import configure_arm
from ..reviewer_validation.endpoints import EndpointHarness
from .experiments import setup,step,clone,BASE
from .editorial_audit import lines

def main():
    configure_cpu();s=load_config(BASE/'configs/tmlr_cost_check.yaml');out=Path(s['output']);out.mkdir(exist_ok=False)
    (out/'registration.json').write_text(json.dumps(s,indent=2));groups={};start=time.monotonic()
    for r in lines(BASE/'runs/tmlr_broadcast_20261119_001/steps.jsonl'):
        if r['episode']==s['episode']:groups.setdefault((r['seed'],r['method'],r['condition'],r['scheduler']),[]).append(r)
    checks=[]
    with torch.inference_mode():
        for (seed,method,condition,sched),original in groups.items():
            assert time.monotonic()-start<s['wall_seconds']
            link,deadline=condition.split('_');deadline=float(deadline[:-2])
            model=DecayedPolicy(4,10,5,2,hidden_dim=64,message_dim=8,mixer='qmix')
            model.load_state_dict(torch.load(BASE/f'runs/hidden_mechanism_20261119_001/broadcast_n5_seed{seed}_{method}/final.pt',map_location='cpu',weights_only=True)['model']);model.eval()
            _,cfg=configure_arm(setup([link,deadline]),method,{'slow':20000,'medium':100000}[link],deadline);cfg['fixed_ready']=True
            actual='full' if method in ['no_comm','ideal_full'] else 'dac' if method=='rr_cap' else method
            env=HiddenSource();obs,_=env.reset(seed=14200000);h=EndpointHarness(model,actual,cfg,0,False)
            for r in original:
                chosen=None
                if sched=='source_aware':
                    _,p=step(deepcopy(env),clone(h),obs.copy(),r['t']);k=len(p['senders']);chosen=sorted(([0]+[i for i in p['senders'] if i!=0])[:k]) if k else None
                bit=int(env.bits[0]);obs,tr=step(env,h,obs,r['t'],chosen)
                assert tr['reward']==r['followers_accuracy'] and len(tr['senders'])==r['k'] and (0 in tr['senders'])==r['source_delivered'] and bit==r['bit']
                attempt=r['k'] if sched=='source_aware' or method in ['dac','rr_cap'] else 5 if method=='ideal_full' else 0 if method=='no_comm' else sum((r['t']+i)%2==0 for i in range(5))
                assert tr['wire']==48*attempt
                checks.append(dict(seed=seed,method=method,condition=condition,scheduler=sched,t=r['t'],wire=tr['wire'],delivered=r['k']))
    assert len(checks)==s['max_steps']
    (out/'summary.json').write_text(json.dumps(dict(cells=len(groups),steps=len(checks),mismatches=0,scope=s['scope']),indent=2))
    with (out/'steps.jsonl').open('w') as f:
        for r in checks:f.write(json.dumps(r)+'\n')
    (out/'status.json').write_text(json.dumps(dict(stage='completed')))
    print((out/'summary.json').read_text())
if __name__=='__main__':main()
