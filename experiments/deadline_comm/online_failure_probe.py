"""Read-only final-checkpoint behavior and successful-trajectory Q probe."""
from .timing import configure_cpu

def main():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument("--run-name",default="online_oracle_qmix_001");args=ap.parse_args()
 configure_cpu()
 import torch,numpy as np
 from collections import Counter
 from .common import ROOT,load_config,build_policy,save_json,save_csv
 from .adapters.hallway import Hallway
 from .harness import rollout
 from .td_diagnostic_data import generate_data
 from .learner import unroll
 p=ROOT/'runs'/args.run_name;cfg=load_config(ROOT/'configs/timing_rank.yaml');m=build_policy(cfg)
 m.load_state_dict(torch.load(p/'checkpoint_1500.pt',map_location='cpu',weights_only=True)['model']);env=Hallway(**cfg['environment'])
 actions=Counter();scores=[];qrows=[]
 for ep in range(30):
  r,_,_,s=rollout(env,m,'full',cfg['harness'],980000+ep,False);scores.append(dict(episode=ep,**s));actions.update(r.episode.actions.flatten().tolist())
  with torch.no_grad():
   q=unroll(m,r);values=[m.joint(q[t],torch.as_tensor(r.episode.actions[t])[None],torch.as_tensor(r.episode.states[t])[None]).item() for t in range(len(r.episode.actions))]
  qrows.extend(values)
 _,oracle=generate_data(m,env,cfg,990000,10);probe=[]
 with torch.no_grad():
  for r in oracle:
   q=unroll(m,r)
   for t in range(len(r.episode.actions)):
    s=torch.as_tensor(r.episode.states[t])[None];a=torch.as_tensor(r.episode.actions[t])[None];g=q[t].argmax(-1)
    probe.append(dict(q_oracle=m.joint(q[t],a,s).item(),q_greedy=m.joint(q[t],g,s).item(),oracle_reward_to_go=float(sum((.99**j)*v for j,v in enumerate(r.episode.rewards[t:]))),joint_action_match=bool(torch.equal(a,g))))
 result=dict(final_mean_return=float(np.mean([s['return'] for s in scores])),final_successes=sum(s['success'] for s in scores),action_counts=dict(actions),mean_predicted_executed_q=float(np.mean(qrows)),max_predicted_executed_q=max(qrows),oracle_action_match_rate=float(np.mean([r['joint_action_match'] for r in probe])),oracle_probe=probe)
 save_json(p/'failure_probe.json',result);save_csv(p/'final_holdout.csv',scores);print({k:v for k,v in result.items() if k!='oracle_probe'})
if __name__=='__main__':main()
