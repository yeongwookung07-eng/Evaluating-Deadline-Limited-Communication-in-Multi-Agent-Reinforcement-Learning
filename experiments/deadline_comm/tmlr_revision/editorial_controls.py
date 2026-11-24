"""Bounded non-learning reference policies and direct environment semantics checks."""
import json
from pathlib import Path
import numpy as np
from ..common import load_config
from ..adapters.hallway import Hallway

def main():
    cfg=load_config('experiments/deadline_comm/configs/tmlr_editorial_controls.yaml')
    out=Path(cfg['output']);out.mkdir(exist_ok=False)
    (out/'registration.json').write_text(json.dumps(cfg,indent=2))
    env=Hallway(group_ids=[0,0,1,1,1]);rows=[];steps=0
    for seed in cfg['seeds']:
        rng=np.random.default_rng(seed);returns=[]
        for ep in range(cfg['episodes']):
            env.reset(seed=cfg['reset_base']+ep);ret=0
            for t in range(env.episode_length):
                _,reward,term,trunc,_=env.step(rng.integers(0,3,size=5));steps+=1;ret+=reward
                if term or trunc:break
            returns.append(ret)
        rows.append(dict(seed=seed,returns=returns,mean_return=float(np.mean(returns))))
    assert steps<=cfg['max_steps']
    env.reset(seed=0);env.env.state_n[:]=1
    obs,reward,term,trunc,_=env.step(np.ones(5,int))
    assert reward==-10 and not term and not trunc
    assert (env.env.state_n==0).all() and not any(env.env.active_group) and env.env.active_agent.all()
    for t in range(13):obs,reward,term,trunc,_=env.step(np.zeros(5,int))
    assert trunc and not term
    result=dict(random_policy=rows,steps=steps,simultaneous_arrival=dict(initial=[1]*5,first_reward=-10,post_distance=[0]*5,active_groups=[False]*2,active_agents=[True]*5,ends_by_timeout=True),scope=cfg['scope'])
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    (out/'status.json').write_text(json.dumps(dict(stage='completed')))
    print({k:v for k,v in result.items() if k!='random_policy'})
if __name__=='__main__':main()
