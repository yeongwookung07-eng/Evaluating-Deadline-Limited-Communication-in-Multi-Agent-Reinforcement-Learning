"""Read saved smoke artifacts without launching additional training."""
import csv
import json
from pathlib import Path
import math
import pytest

ROOT=Path(__file__).parents[1]/'experiments/deadline_comm'

def validate_artifacts(path):
    required=['manifest.json','steps.csv','messages.jsonl','episodes.csv','summary.json']
    assert all((path/p).is_file() for p in required)
    meta=json.loads((path/'manifest.json').read_text())
    assert meta['status']=='completed'
    assert meta['threads']==meta['interop_threads']==1
    steps=list(csv.DictReader((path/'steps.csv').open()))
    episodes=list(csv.DictReader((path/'episodes.csv').open()))
    assert {r['method'] for r in episodes}=={'full','periodic','random','vbc','tmc'}
    for r in steps:
        cpu=float(r['cpu_total_ns']); comm=float(r['comm_model_ns'])
        assert cpu>0 and comm>=0
        assert math.isclose(cpu+comm,float(r['step_total_ns']))
        assert math.isclose(cpu,sum(float(r[k]) for k in ('encode_ns','gate_ns','cache_ns','act_ns','other_ns')))
        assert int(r['K_delivered'])<=int(r['K_attempt'])
        assert (r['action_miss']=='True')==(float(r['step_total_ns'])>float(r['deadline_ns']))
    for line in (path/'messages.jsonl').read_text().splitlines():
        r=json.loads(line)
        if r['accepted']: assert r['last_received_step']==r['t'] and r['age']==0
    result=json.loads((path/'summary.json').read_text())
    assert len(result['groups'])==len({(r['condition'],r['method']) for r in episodes})
    return dict(steps=len(steps),episodes=len(episodes),messages=len((path/'messages.jsonl').read_text().splitlines()))

def test_smoke_artifacts_if_present():
    path=ROOT/'runs/smoke_eval_003'
    if not path.exists(): pytest.skip('Integration smoke artifacts not created yet')
    validate_artifacts(path)
