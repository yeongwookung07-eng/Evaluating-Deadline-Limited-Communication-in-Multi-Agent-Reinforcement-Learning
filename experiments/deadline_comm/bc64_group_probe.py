"""Read-only group outcome diagnosis on the registered independent holdout seeds."""
from .timing import configure_cpu

def main():
 configure_cpu()
 import torch
 from collections import Counter
 from .common import ROOT,load_config,build_policy,save_json
 from .exploration_reward_audit import DiagnosticHallway
 from .harness import rollout
 p=ROOT/'runs/online_oracle_bc64_001';cfg=load_config(ROOT/'configs/timing_rank.yaml');cfg['policy']['hidden_dim']=64
 model=build_policy(cfg);model.load_state_dict(torch.load(p/'checkpoint_1500.pt',map_location='cpu',weights_only=True)['model'])
 env=DiagnosticHallway(**cfg['environment']);wins=Counter();failures=Counter();traces=[]
 for ep in range(30):
  _,_,_,summary=rollout(env,model,'full',cfg['harness'],980000+ep,False)
  for e in env.events:
   if not e['simultaneous_group_penalty']:wins.update(e['completed_groups_this_step'])
   failures.update(e['async_failed_groups'])
  traces.append(dict(episode=ep,summary=summary,steps=env.events))
 result=dict(group_success_events=dict(wins),group_async_failure_events=dict(failures),episodes=traces)
 save_json(p/'group_probe.json',result);print(dict(wins),dict(failures))
if __name__=='__main__':main()
