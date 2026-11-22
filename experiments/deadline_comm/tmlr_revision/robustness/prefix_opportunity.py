"""Exact two-group completion opportunity after the recorded common prefix."""
import json
from pathlib import Path
import numpy as np
from ...common import load_config
from ...adapters.hallway import Hallway
from ...day2.io import atomic_json
from ..experiments import BASE
from ..editorial_audit import lines

def main():
    s=load_config(BASE/'configs/tmlr_prefix_opportunity.yaml');out=Path(s['output']);out.mkdir(exist_ok=False);atomic_json(out/'registration.json',s);rows=[];steps=0
    for r in lines(Path(s['source'])):
        env=Hallway(group_ids=[0,0,1,1,1]);env.reset(seed=14100000+r['episode'])
        for p in r['prefix']:
            _,reward,_,_,_=env.step(p['actions']);assert reward==p['reward'] and env.get_state().tolist()==p['state']
        remaining=14-len(r['prefix']);schedule={};last=0
        groups=sorted((int(max(env.env.state_n[env.env.group_ids==g])),g) for g,active in enumerate(env.env.active_group) if active)
        if r['survived']:
            for earliest,g in groups:
                assert earliest>=1
                arrival=max(earliest,last+1)
                if arrival<=remaining:schedule[g]=arrival;last=arrival
        target=10*len(schedule);actual=0
        if r['survived']:
            for local_t in range(1,remaining+1):
                actions=np.zeros(5,int)
                for i in range(5):
                    g=int(env.env.group_ids[i])
                    if g in schedule and env.env.active_group[g]:
                        distance=int(env.env.state_n[i]);actions[i]=int(distance>1 or (distance==1 and local_t==schedule[g]))
                _,reward,term,trunc,_=env.step(actions);steps+=1;actual+=reward
                if term or trunc:break
        assert actual==target
        rr=r['paired']['rr_cap']['suffix_return'];dac=r['paired']['dac']['suffix_return'];assert max(rr,dac)<=target
        rows.append(dict(seed=r['seed'],condition=r['condition'],arm=r['checkpoint_arm'],episode=r['episode'],survived=r['survived'],remaining_steps=remaining,unresolved=len(groups),oracle_return=actual,rr_return=rr,dac_return=dac,unused_opportunity=actual>max(rr,dac)))
    assert len(rows)==s['max_episodes'] and steps<=s['max_oracle_steps']
    atomic_json(out/'summary.json',dict(episodes=len(rows),oracle_steps=steps,records=rows,scope=s['scope'],contract='Each active group needs at least max(distance) steps. Distinct completion times avoid penalty. Earliest-deadline-free scheduling of two unit completions at these release times maximizes completed groups by horizon; all predicted rewards matched executed oracle suffixes.'))
    atomic_json(out/'status.json',dict(stage='completed'));print(out)
if __name__=='__main__':main()
