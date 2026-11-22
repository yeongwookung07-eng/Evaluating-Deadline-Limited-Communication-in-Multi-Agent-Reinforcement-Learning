import pytest
import torch
from experiments.deadline_comm.day2.dac import DACGate,q_gap,link_cap,decay_messages
from experiments.deadline_comm.day2.policy import DecayedPolicy
from experiments.deadline_comm.day2.harness import Day2Harness
from experiments.deadline_comm.common import load_config


def test_dac_batched_equivalence():
    torch.manual_seed(41)
    m=DecayedPolicy(2,10,5,3,hidden_dim=64);h=torch.randn(1,5,64)
    mask=torch.ones(1,5,3,dtype=torch.bool);mask[0,2,1:]=False
    q=m.local_q(h);batched=q_gap(q,mask)
    scalar=[]
    for i in range(5):
        local=m.local_q(h[:,i:i+1])[0,0][mask[0,i]]
        scalar.append(local.topk(2).values.diff().neg()[0] if len(local)>=2 else local.new_tensor(0.))
    torch.testing.assert_close(batched[0],torch.stack(scalar),rtol=1e-5,atol=1e-6)
    g=DACGate(5,alpha=.1);g.age=torch.arange(5)
    torch.testing.assert_close(g.scores(q,mask)[0],torch.stack(scalar)+.1*torch.arange(5))
    # Sequential and batched VBC mask agrees, single-legal-action never sends.
    assert torch.equal((batched[0]<=.1)&(mask[0].sum(-1)>1),(torch.stack(scalar)<=.1)&(mask[0].sum(-1)>1))


def test_dac_congestion_cap():
    cfg=load_config('experiments/deadline_comm/configs/smoke.yaml')['harness']
    cfg.update(r_post_ns=450000,dac=dict(alpha=.1,threshold=0.,budget_bytes=120.,packet_bytes=48,burst_packets=3),decay_lambda=.9)
    m=DecayedPolicy(2,10,5,3,hidden_dim=64)
    for deadline,bandwidth,upper in [(5000000,20000,1),(1500000,20000,0),(1500000,100000,1),(5000000,1000000,3)]:
        cfg['deadline_ns']=deadline;cfg['link']['bandwidth_bytes_s']=bandwidth
        h=Day2Harness(m,'dac',cfg,0,False);spent=0
        for t in range(14):
            _,row,_=h.step(torch.ones(5,2),torch.ones(5,3,dtype=torch.bool),t)
            assert row['K_attempt']<=min(row['k_max'],upper)
            spent+=row['wire_bytes'];assert spent<=120*(t+1)
            assert row['drop_count']==0
    assert link_cap(1500000,450000,0,20000)[0]==0
    assert link_cap(5000000,450000,0,20000)[0]==1


def test_dac_aoi_reset():
    g=DACGate(5);g.start_step();g.commit(torch.tensor([True,False,True,False,False]))
    assert g.age.tolist()==[0,1,0,1,1]
    g.start_step();g.commit(torch.tensor([False,True,False,False,False]))
    assert g.age.tolist()==[1,0,1,2,2]
    # Commit is based on transmission attempts, no delivery ACK is consumed.
    g.start_step();g.commit(torch.zeros(5,dtype=torch.bool));assert g.age.tolist()==[2,1,2,3,3]


def test_dac_staleness_decay():
    cache=torch.ones(1,4,2);source=torch.tensor([[3,2,0,-1]])
    result=decay_messages(cache,source,3,.5)
    torch.testing.assert_close(result[0,:,0],torch.tensor([1.,.5,.125,0.]))
    torch.testing.assert_close(decay_messages(cache,source,3,1.)[0,:,0],torch.tensor([1.,1.,1.,0.]))
    m=DecayedPolicy(2,8,4,3,hidden_dim=16,message_dim=2);h=torch.randn(1,4,16)
    m.decay_lambda=1.;reference=m.q(h,cache,source,3)
    m.decay_lambda=.5;decayed=m.q(h,cache,source,3)
    assert not torch.allclose(reference,decayed)
