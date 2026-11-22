"""Freeze gate thresholds and empirical R_post on separate validation seeds.

This is bounded calibration, not selection by evaluation Return. Full-Comm is an
anchor; periodic/random set the nominal sparse budget. Actual matched-budget
eligibility is still checked on held-out evaluation trajectories.
"""
import argparse
from .timing import configure_cpu

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True)
    ap.add_argument('--checkpoint',required=True); ap.add_argument('--run-name',required=True)
    ap.add_argument('--seed',type=int,default=710000); ap.add_argument('--episodes',type=int,default=3)
    ap.add_argument('--target-rate',type=float,default=.5)
    a=ap.parse_args(); configure_cpu()
    import copy
    import math
    import numpy as np
    import torch
    import yaml
    from .common import load_config,output_dir,manifest,save_json,build_policy
    from .adapters.hallway import Hallway
    from .harness import rollout
    if a.episodes<1 or not 0<a.target_rate<=1: raise ValueError('Invalid calibration budget')
    cfg=load_config(a.config); out=output_dir(a.run_name)
    meta=manifest(cfg,'calibrate',a.seed,a.checkpoint); save_json(out/'manifest.json',meta)
    try:
        saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
        if saved['metadata']['policy']!=cfg['policy'] or saved['metadata']['environment']!=cfg['environment']:
            raise ValueError('Checkpoint mismatch')
        eval_start=cfg['evaluation']['seed']; eval_end=eval_start+cfg['evaluation']['episodes']
        if max(a.seed,eval_start)<min(a.seed+a.episodes,eval_end): raise ValueError('Calibration/evaluation seeds overlap')
        model=build_policy(cfg); model.load_state_dict(saved['model']); model.eval()
        env=Hallway(**cfg['environment']); rows=[]
        frozen=copy.deepcopy(cfg)
        frozen['harness']['gate']['p']=a.target_rate
        frozen['harness']['gate']['period']=max(1,round(1/a.target_rate))
        full_bytes=env.n_agents*(4*model.message_dim+cfg['harness']['link']['header_bytes'])
        # Conditions held fixed during gate calibration; return is never a selection input.
        for method,key,grid in [('vbc','q_threshold',[0.,.01,.05,.1,.2,.5,1.,10.]),
                                ('tmc','message_threshold',[0.,.01,.05,.1,.2,.5,1.,10.])]:
            candidates=[]
            for value in grid:
                hc=copy.deepcopy(frozen['harness']); hc['gate'][key]=value
                steps=[]
                for ep in range(a.episodes):
                    _,ss,_,_=rollout(env,model,method,hc,a.seed+ep,True); steps.extend(ss)
                mean_bytes=float(np.mean([r['wire_bytes'] for r in steps]))
                row=dict(method=method,parameter=key,value=value,mean_wire_bytes=mean_bytes,
                    target_wire_bytes=full_bytes*a.target_rate,error=abs(mean_bytes-full_bytes*a.target_rate))
                candidates.append(row); rows.append(row)
            selected=min(candidates,key=lambda r:r['error'])
            frozen['harness']['gate'][key]=selected['value']
        from .gates import METHODS
        post=[]
        for method in METHODS:
            for ep in range(a.episodes):
                _,ss,_,_=rollout(env,model,method,frozen['harness'],a.seed+ep,True)
                post.extend(r['post_cpu_ns'] for r in ss)
        reserve=int(math.ceil(float(np.quantile(post,.99))*1.25))
        if reserve>=frozen['harness']['deadline_ns']:
            raise ValueError('Calibrated post allowance exceeds deadline; configuration is infeasible')
        frozen['harness']['r_post_ns']=reserve
        frozen['calibration']=dict(seed=a.seed,episodes=a.episodes,checkpoint_hash=meta['checkpoint_hash'],
            r_post_rule='1.25 * observed p99 post CPU; empirical, not a bound',target_rate=a.target_rate,
            thresholds_frozen=True)
        (out/'frozen.yaml').write_text(yaml.safe_dump(frozen,sort_keys=False))
        save_json(out/'calibration.json',dict(candidates=rows,r_post_ns=reserve,
            limitation='Threshold search uses base link condition. R_post adjustment changes delivery; recheck held-out budget eligibility.'))
        meta.update(status='completed',r_post_ns=reserve)
    except Exception as exc:
        meta.update(status='failed',error=repr(exc)); raise
    finally: save_json(out/'manifest.json',meta)
    print(out/'frozen.yaml')
if __name__=='__main__': main()
