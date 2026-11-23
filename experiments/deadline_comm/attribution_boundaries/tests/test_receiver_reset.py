from types import SimpleNamespace
import torch
from experiments.deadline_comm.attribution_boundaries.receiver_cache_reset_interaction import clone,reset_receiver,nonreceiver_bytes


class HarnessFixture:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_receiver_reset_invalidates_payload_and_timestamp_without_other_state_changes():
    h=HarnessFixture(model=object(),cache=torch.ones(1,5,8),source=torch.arange(5)[None],hidden=torch.randn(1,5,64),previous=torch.randn(1,5,3),dac=SimpleNamespace(age=torch.tensor([1,2,3,4,5]),cursor=3,credit=96),method='rr_cap')
    branch=clone(h);before=nonreceiver_bytes(branch)
    reset_receiver(branch)
    assert nonreceiver_bytes(branch)==before
    assert not branch.cache.any()
    assert (branch.source==-1).all()
    assert h.cache.all() and torch.equal(h.source,torch.arange(5)[None])
    assert branch.model is h.model
    assert torch.equal(branch.hidden,h.hidden)
    assert torch.equal(branch.previous,h.previous)
    assert torch.equal(branch.dac.age,h.dac.age)
    assert branch.dac.cursor==3 and branch.dac.credit==96
