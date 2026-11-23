from copy import deepcopy

import numpy as np
import torch
from torch import nn

from experiments.deadline_comm.receiver_adaptation.evaluate import SplitUtility, fit_head


def test_split_preserves_source_for_identical_inputs_and_original_forward():
    torch.manual_seed(31)
    head=nn.Sequential(nn.Linear(74,64),nn.ReLU(),nn.Linear(64,2))
    x=torch.randn(3,5,74)
    unchanged=SplitUtility(head,head)
    assert torch.equal(unchanged(x),head(x))
    changed=deepcopy(head)
    with torch.no_grad():
        changed[-1].bias.add_(torch.tensor([5.,-5.]))
    split=SplitUtility(head,changed)
    assert torch.equal(split(x)[:,:1],head(x)[:,:1])
    assert torch.equal(split(x)[:,1:],changed(x)[:,1:])


def test_fit_changes_only_copied_head_and_counts_label_budget():
    torch.manual_seed(32)
    head=nn.Sequential(nn.Linear(74,64),nn.ReLU(),nn.Linear(64,2))
    before=deepcopy(head.state_dict())
    x=np.random.default_rng(1).normal(size=(8,4,74))
    y=np.arange(8)%2
    settings=dict(updates=2,learning_rate=.001)
    fitted,info=fit_head(head,x,y,settings,seed=0,shuffle=False,check_budget=lambda:None)
    assert all(torch.equal(before[k],head.state_dict()[k]) for k in before)
    assert any(not torch.equal(before[k],fitted.state_dict()[k]) for k in before)
    assert info['labeled_timesteps']==8 and info['receiver_examples']==32
