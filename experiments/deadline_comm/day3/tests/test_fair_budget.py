import torch
from experiments.deadline_comm.day2.dac import DACGate
from experiments.deadline_comm.day3.fair_budget import budget_config,nondominated


def test_high_budget_capacity_and_prefix():
    cfg={'dac':dict(alpha=.1,threshold=0.,budget_bytes=120.,packet_bytes=48,burst_packets=3)}
    for budget in (48,96,120,160,240):
        gate=DACGate(5,**budget_config(cfg,budget)['dac'])
        for t in range(60):
            k=gate.start_step()
            mask=gate.select(torch.ones(1,5),k)
            gate.commit(mask)
            assert gate.spent<=(t+1)*budget
        assert abs(gate.spent/60-budget)<1e-8
    assert cfg['dac']['burst_packets']==3


def test_pareto_direction_and_ties():
    ps=[{'x':1,'r':2},{'x':2,'r':1},{'x':3,'r':4},{'x':1,'r':2}]
    assert nondominated(ps,'x','r')==[ps[0],ps[2],ps[3]]
