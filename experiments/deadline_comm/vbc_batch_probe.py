from .timing import configure_cpu

def main():
 configure_cpu()
 import torch,numpy as np,gc,json
 from time import perf_counter_ns
 from .common import ROOT,load_config,build_policy,save_json
 from .adapters.hallway import Hallway
 from .harness import rollout
 p=ROOT/'runs/bc64_preliminary_001';cfg=load_config(ROOT/'configs/timing_rank.yaml');cfg['policy']['hidden_dim']=64
 m=build_policy(cfg);m.load_state_dict(torch.load(ROOT/'runs/online_oracle_bc64_001/checkpoint_1500.pt',map_location='cpu',weights_only=True)['model']);m.eval()
 threshold=json.loads((p/'frozen_harness.json').read_text())['gate']['q_threshold'];env=Hallway(**cfg['environment']);bank=[]
 with torch.inference_mode():
  for ep in range(4):
   record=rollout(env,m,'full',cfg['harness'],1400000+ep,False)[0];h=m.initial_hidden(1);prev=torch.zeros(1,5,3)
   for t,obs in enumerate(record.episode.observations[:-1]):
    h=m.encode(torch.as_tensor(obs)[None],prev,h);bank.append(h.clone());prev=torch.nn.functional.one_hot(torch.as_tensor(record.episode.actions[t]),3).float()[None]
  bank=bank[:50]
  def scalar(h):
   return torch.stack([torch.diff(m.local_q(h[:,i:i+1])[0,0].topk(2).values).neg()[0] for i in range(5)])<=threshold
  def batched(h):
   values=m.local_q(h).topk(2,dim=-1).values
   return (values[:,:,0]-values[:,:,1]<=threshold)[0]
  assert all(torch.equal(scalar(h),batched(h)) for h in bank)
  for h in bank:scalar(h);batched(h)
  samples={'scalar':[],'batched':[]};gc.collect();was=gc.isenabled();gc.disable()
  try:
   for rep in range(10):
    for h in bank:
     order=[('scalar',scalar),('batched',batched)]
     if rep%2:order.reverse()
     for name,fn in order:
      start=perf_counter_ns();fn(h);samples[name].append(perf_counter_ns()-start)
  finally:
   if was:gc.enable()
 result=dict(states=len(bank),repetitions=10,all_gate_decisions_equal=True,scope='kernel-only local-Q plus top2; not end-to-end harness timing',timings={k:dict(median_us=float(np.median(v))/1000,p95_us=float(np.quantile(v,.95))/1000) for k,v in samples.items()})
 save_json(p/'vbc_batch_probe.json',result);print(result)
if __name__=='__main__':main()
