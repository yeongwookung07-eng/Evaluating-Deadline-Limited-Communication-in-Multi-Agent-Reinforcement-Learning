"""Constructive no-message diagnostic for the fixed, public Hallway task."""
import itertools,json
from pathlib import Path
import numpy as np
import torch
from ..common import load_config
from ..day2.io import atomic_json
from ..adapters.hallway import Hallway
s=load_config('experiments/deadline_comm/configs/tmlr_local_clock.yaml');out=Path(s['output']);out.mkdir(exist_ok=False);atomic_json(out/'registration.json',s)
meta=torch.load(s['checkpoint_metadata'],map_location='cpu',weights_only=True)['metadata'];cfg=meta['environment'];roster=cfg['group_ids'];maximum=max(cfg['state_numbers']);env=Hallway(**cfg);rows=[];steps=0
# Input signature enforces local observation only; roster entry is static public information.
def local_action(local_obs,t,group_id,maximum):
 distance,active=local_obs
 if not active:return 0
 if distance>1:return 1
 return int(distance==1 and t==maximum-1+group_id)
for initial in itertools.product(*[range(1,n+1) for n in cfg['state_numbers']]):
 env.reset(seed=0);env.env.state_n[:]=initial;obs=np.asarray(env.env.get_obs(),np.float32);ret=0;trace=[]
 for t in range(env.episode_length):
  a=np.array([local_action(obs[i],t,roster[i],maximum) for i in range(env.n_agents)])
  obs,r,term,trunc,_=env.step(a);ret+=r;steps+=1;trace.append(dict(t=t,actions=a.tolist(),reward=r))
  if term or trunc:break
 rows.append(dict(initial=initial,return_=ret,steps=len(trace),trace=trace))
assert len(rows)<=s['max_initial_states'] and steps<=s['max_steps']
atomic_json(out/'summary.json',dict(initial_states=len(rows),steps=steps,min_return=min(r['return_'] for r in rows),mean_return=float(np.mean([r['return_'] for r in rows])),all_optimal=all(r['return_']==20 for r in rows),communication_bytes=0,assumptions=s['contract'],interpretation='Constructive hand-designed local policy, not learned No-Comm baseline; task solvability differs from learnability under privileged demonstrations'))
with (out/'episodes.jsonl').open('w') as f:
 for r in rows:f.write(json.dumps(r)+'\n')
atomic_json(out/'status.json',dict(stage='completed'));print((out/'summary.json').read_text())
