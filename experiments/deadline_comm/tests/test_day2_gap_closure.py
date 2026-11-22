import json
from pathlib import Path
import torch
from experiments.deadline_comm.day2.decentralized_slot_only import DecentralizedSlotOnlyGate
from experiments.deadline_comm.day2.close_day2_gaps import (
    train_full, evaluation_config, controlled, timing_statistics, PHASES)
from experiments.deadline_comm.day2.finetune_comm_aware import load_model
from experiments.deadline_comm.day2.aggregate_day2 import aggregate, METRICS
from experiments.deadline_comm.adapters.hallway import Hallway

ROOT=Path('experiments/deadline_comm/runs/day2_dac_marl_main_20261118_001/seed0')


def test_slot_cap_and_score_invariance():
    a=DecentralizedSlotOnlyGate(5,seed=3,probability=.7)
    b=DecentralizedSlotOnlyGate(5,seed=3,probability=.7)
    for t in range(30):
        cap=min(2,a.start_step())
        b.start_step()
        b.age.fill_(1000)
        x=a.select(torch.randn(1,5),cap)
        y=b.select(torch.full((1,5),10000.),cap)
        assert torch.equal(x,y) and x.sum()<=cap
        a.commit(x);b.commit(y)
        assert a.spent<=120*(t+1)


def test_full_continuation_update(tmp_path):
    checkpoint=ROOT/'train/checkpoint_1500.pt'
    cfg=json.loads((ROOT/'frozen_harness.json').read_text())
    before,_=load_model(checkpoint)
    after=train_full(checkpoint,cfg,tmp_path,0,1,'cuda:0')
    assert any(not torch.equal(a,b) for a,b in zip(before.parameters(),after.parameters()))
    saved=torch.load(tmp_path/'checkpoint_0001.pt',weights_only=True,map_location='cpu')
    assert saved['updates']==1 and saved['training_config']['bootstrap_timeouts'] is False
    assert saved['training_config']['target_update_interval']==100


def test_post_ft_lambda_one():
    cfg=json.loads((ROOT/'frozen_harness.json').read_text())
    local=evaluation_config(cfg,'comm_aware_lambda1')
    assert local['decay_lambda']==1 and cfg['decay_lambda']!=1
    model,meta=load_model(ROOT/'train/checkpoint_1500.pt')
    env=Hallway(**meta['environment'])
    controlled(env,model,'dac',local,2200000,False)
    assert model.decay_lambda==1.


def test_latency_and_unfiltered_aggregation():
    stats=timing_statistics([{p:v for p in PHASES} for v in (1000,2000,3000)])
    assert stats['gate_ns']['mean']==2 and stats['gate_ns']['median']==2
    assert abs(stats['gate_ns']['p95']-2.9)<1e-9
    rows=[dict(seed=i,category='main',condition='slow_5ms',method='dac',budget_eligible=False,
               **{m:float(i) for m in METRICS}) for i in range(3)]
    result=aggregate(rows,{0:9.,1:4.,2:0.})
    assert result[0]['seeds']==[0,1,2]
    assert result[0]['metrics']['mean_return']==dict(mean=1.,std=1.)
