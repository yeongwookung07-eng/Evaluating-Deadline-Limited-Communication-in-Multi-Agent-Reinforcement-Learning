from copy import deepcopy
import torch
import pytest
from experiments.deadline_comm.common import load_config,build_policy
from experiments.deadline_comm.adapters.hallway import Hallway
from experiments.deadline_comm.td_diagnostic_data import generate_data
from experiments.deadline_comm.learner import CommLearner
from experiments.deadline_comm.batched_learner import BatchedCommLearner

@pytest.mark.parametrize("bootstrap", [True,False])
def test_matches_original_loss_gradients_and_weights(bootstrap):
 torch.set_num_threads(1);torch.manual_seed(3)
 cfg=load_config('experiments/deadline_comm/configs/timing_rank.yaml');cfg['learner']['bootstrap_timeouts']=bootstrap;m=build_policy(cfg)
 random,oracle=generate_data(m,Hallway(**cfg['environment']),cfg,42,3)
 records=random+oracle
 assert len(set(len(r.episode.actions) for r in records))>1
 a=CommLearner(deepcopy(m),cfg['learner']);b=BatchedCommLearner(deepcopy(m),cfg['learner'])
 for _ in range(2):
  x=a.train_batch(records);y=b.train_batch(records)
  assert abs(x['td_loss']-y['td_loss'])<=1e-4*max(1,x['td_loss'])
  for p,q in zip(a.model.parameters(),b.model.parameters()):
   torch.testing.assert_close(p.grad,q.grad,rtol=3e-3,atol=1e-5)
   torch.testing.assert_close(p,q,rtol=3e-3,atol=1e-5)
