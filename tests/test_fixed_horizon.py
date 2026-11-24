from types import SimpleNamespace
import torch
import pytest
from experiments.deadline_comm.td_targets import td_target
from experiments.deadline_comm.replay import StratifiedReplay

@pytest.mark.parametrize('bootstrap,expected',[(False,[10.,0.,49.5]),(True,[10.,49.5,49.5])])
def test_horizon_mask(bootstrap,expected):
 out=td_target(torch.tensor([10.,0.,0.]),torch.tensor([50.,50.,50.]),[True,False,False],[False,True,False],.99,bootstrap)
 torch.testing.assert_close(out,torch.tensor(expected))

def test_terminal_inf_does_not_contaminate_target():
 assert td_target(10.,torch.tensor([float('inf')]),True,False,.99,False).item()==10.

def test_strata_exact_quota_retention_ownership():
 r=StratifiedReplay(12,0)
 for i in range(50):r.insert(SimpleNamespace(oracle_source=True,value=i))
 with pytest.raises(ValueError):r.sample(16)
 for i in range(40):
  r.insert(SimpleNamespace(oracle_source=False,value=i))
  batch=r.sample(16)
  assert sum(x.oracle_source for x in batch)==4
  batch[0].value=-999
 assert len(r.oracle)==50 and len(r.online)==12
 assert all(x.value!=-999 for x in r.items)

@pytest.mark.parametrize('batched',[False,True])
def test_double_q_uses_online_selection_not_target_argmax(batched):
 from experiments.deadline_comm.common import load_config,build_policy
 from experiments.deadline_comm.adapters.hallway import Hallway
 from experiments.deadline_comm.harness import rollout
 from experiments.deadline_comm.learner import CommLearner
 from experiments.deadline_comm.batched_learner import BatchedCommLearner
 torch.manual_seed(4);torch.set_num_threads(1)
 cfg=load_config('experiments/deadline_comm/configs/smoke.yaml');model=build_policy(cfg)
 record=rollout(Hallway(**cfg['environment']),model,'full',cfg['harness'],42,False)[0]
 learner=(BatchedCommLearner if batched else CommLearner)(model,cfg['learner'])
 with torch.no_grad():
  for net,bias in [(learner.model,[10.,0.,0.]),(learner.target,[0.,10.,0.])]:
   net.utility[-1].weight.zero_();net.utility[-1].bias.copy_(torch.tensor(bias))
 original=learner.target.joint;selected=[]
 def spy(q,actions,state):
  assert (q.argmax(-1)==1).all()  # target itself prefers action1
  selected.append(actions.detach().clone())
  return original(q,actions,state)
 learner.target.joint=spy
 learner.train_batch([record])
 assert selected and all((a==0).all() for a in selected)  # online selected0
