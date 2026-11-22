from pathlib import Path
import torch
from experiments.deadline_comm.day2.decentralized_gate import DecentralizedDACGate
from experiments.deadline_comm.day2.aggregate_day2 import aggregate
from experiments.deadline_comm.day2.finetune_comm_aware import finetune, load_model
import json

ROOT=Path('experiments/deadline_comm/runs/day2_dac_marl_main_20261118_001/seed0')


def test_decentralized_budget():
    gate=DecentralizedDACGate(5,threshold=-100)
    for _ in range(100):
        cap=min(1,gate.start_step())
        chosen=gate.select(torch.ones(1,5),cap)
        assert chosen.sum()<=cap
        assert torch.equal(chosen,gate.select(torch.ones(1,5),cap))
        gate.commit(chosen)
        assert gate.spent<=120*gate.steps


def test_aggregate_dummy():
    from experiments.deadline_comm.day2.aggregate_day2 import METRICS
    rows=[dict(seed=i,category='main',condition='slow_5ms',method='dac',budget_eligible=False,
               **{m:float(i) for m in METRICS}) for i in range(3)]
    result=aggregate(rows,{0:9.,1:0.,2:9.})
    assert result[0]['metrics']['mean_return']==dict(mean=1.,std=1.)
    assert result[1]['seeds']==[0,2]


def test_finetune_load_update(tmp_path):
    path=ROOT/'train/checkpoint_1500.pt'
    cfg=json.loads((ROOT/'frozen_harness.json').read_text())
    before,_=load_model(path)
    after=finetune(path,cfg,tmp_path,0,updates=1,device='cuda:0')
    assert any(not torch.equal(a,b) for a,b in zip(before.parameters(),after.parameters()))
    loaded,_=load_model(tmp_path/'checkpoint_0001.pt')
    assert all(torch.equal(a,b) for a,b in zip(after.parameters(),loaded.parameters()))
