import torch
from experiments.deadline_comm.external_biases.mnist_sum_learning import Policy,biases

def test_listening_stop_gradients():
    c={'entropy_target':1.,'ps_lambda':3.}
    s=torch.randn(8,20,requires_grad=True);l=torch.randn(8,19,requires_grad=True);z=torch.randn(8,19,requires_grad=True)
    ps,pl,ce=biases(s,l,z,c)
    g=torch.autograd.grad(pl,[l,z],allow_unused=True,retain_graph=True)
    assert g[0].abs().sum()>0 and g[1] is None
    g=torch.autograd.grad(ce,[l,z],allow_unused=True,retain_graph=True)
    assert g[0] is None and g[1].abs().sum()>0
    assert torch.autograd.grad(ps,s)[0].abs().sum()>0

def test_discrete_channel_blocks_listener_gradient_to_speaker():
    torch.set_num_threads(1);s=Policy();l=Policy(True);x=torch.rand(2,1,28,28)
    m=torch.distributions.Categorical(logits=s(x)).sample()
    l(x,torch.nn.functional.one_hot(m,20).float()).sum().backward()
    assert all(p.grad is None for p in s.parameters())
    assert any(p.grad is not None for p in l.parameters())

def test_original_power_metrics_on_known_perfect_protocol():
    from experiments.deadline_comm.external_biases.mnist_sum_learning import evaluate
    class Speaker:
        def __call__(self,x):
            d=torch.round(x[:,0,0,0]*255).long()
            return torch.nn.functional.one_hot(d,20).float()*60
    class Listener:
        def cnn(self,x): return torch.round(x[:,0,0,0]*255)[:,None]
        def head(self,z):
            digit=z[...,0].long();m=z[...,1:].argmax(-1)%10
            return torch.nn.functional.one_hot(digit+m,19).float()*60
    labels=torch.arange(10).repeat(100);images=labels[:,None,None,None].expand(-1,1,28,28).to(torch.uint8)
    result=evaluate(Speaker(),Listener(),images,labels,torch.arange(1000),1000,33)
    assert result['exact']>.9999
    assert .8<result['original_listener_power']<.95
    assert .8<result['original_speaker_power']<.95
    assert .07<result['uniform_speaker_expected']<.13
