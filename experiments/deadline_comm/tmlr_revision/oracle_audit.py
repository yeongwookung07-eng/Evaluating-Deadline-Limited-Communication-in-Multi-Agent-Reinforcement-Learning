"""Exhaustive reset Oracle check and exact local-history ambiguity diagnostic."""
from pathlib import Path
from collections import defaultdict
import itertools,json,platform,subprocess
import torch,numpy as np
from ..common import load_config
from ..adapters.hallway import Hallway
from ..hallway_oracle_check import oracle_actions
from ..day2.io import atomic_json
s=load_config('experiments/deadline_comm/configs/tmlr_oracle.yaml');out=Path(s['output']);out.mkdir(exist_ok=False);atomic_json(out/'registration.json',s)
meta=torch.load(s['checkpoint'],map_location='cpu',weights_only=True)['metadata'];env=Hallway(**meta['environment']);groups=env.env.group_ids;histories=defaultdict(set);examples={};returns=[];count=0
for pos in itertools.product(*[range(1,n+1) for n in meta['environment']['state_numbers']]):
 env.reset(seed=0);env.env.state_n[:]=pos;obs=np.asarray(env.env.get_obs(),np.float32);hs=[[] for _ in range(env.n_agents)];ret=0
 for t in range(env.episode_length):
  a=oracle_actions(obs,groups)
  for i in range(env.n_agents):
   hs[i].append(tuple(obs[i].tolist()));key=(i,tuple(hs[i]));histories[key].add(int(a[i]));examples.setdefault(key,[])
   if len(examples[key])<2:examples[key].append(dict(initial=list(pos),teacher_action=int(a[i]),t=t))
   hs[i].append(('previous_action',int(a[i])))
  obs,r,term,trunc,_=env.step(a);ret+=r;count+=1
  if term or trunc:break
 returns.append(ret)
assert len(returns)<=s['max_initial_states'] and count<=s['max_steps']
amb=[k for k,v in histories.items() if len(v)>1]
atomic_json(out/'summary.json',dict(initial_states=len(returns),steps=count,min_return=min(returns),max_return=max(returns),mean_return=float(np.mean(returns)),local_history_keys=len(histories),ambiguous_local_histories=len(amb),scope='Exact own observation/action histories under privileged Oracle trajectories; not proof against all locally learned policies',examples=[dict(agent=k[0],history=k[1],labels=sorted(histories[k])) for k in amb[:10]],environment=meta['environment']))
atomic_json(out/'hardware.json',dict(cpu=subprocess.check_output(['lscpu'],text=True),python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,kernel=platform.release()))
atomic_json(out/'status.json',dict(stage='completed'));print(json.loads((out/'summary.json').read_text())['ambiguous_local_histories'])
