import torch
from experiments.deadline_comm.oracle_bc_learner import oracle_bc_loss

def test_bc_only_oracle_valid_steps_receive_gradients():
 q=torch.randn(2,3,5,3,requires_grad=True);a=torch.zeros(2,3,5,dtype=torch.long)
 mask=torch.ones_like(q,dtype=torch.bool);filled=torch.tensor([[1.,1.,0.],[1.,1.,1.]])
 loss=oracle_bc_loss(q,a,mask,filled,torch.tensor([True,False]));loss.backward()
 assert q.grad[0,:2].abs().sum()>0
 assert q.grad[0,2].abs().sum()==0 and q.grad[1].abs().sum()==0

def test_bc_zero_without_oracle():
 q=torch.randn(2,3,5,3,requires_grad=True);a=torch.zeros(2,3,5,dtype=torch.long)
 loss=oracle_bc_loss(q,a,torch.ones_like(q,dtype=torch.bool),torch.ones(2,3),torch.zeros(2,dtype=torch.bool))
 assert loss.item()==0
 loss.backward();assert q.grad.abs().sum()==0
