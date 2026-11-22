"""Recompute held-out aggregates from raw native episode/step/message logs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from ..common import load_config
from ..day2.io import atomic_json
from .evaluate import seed_summary


def read_jsonl(path):return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def audit(spec,require_gate=True):
    root=Path(spec['output']);out=root/'evaluation'
    if require_gate:
        gate=json.loads((root/'learning_gate.json').read_text());assert gate['passed']
        assert len(gate['records'])==len(spec['seeds'])
        assert all(r['win_rate']>=spec['learning_gate_win_rate'] for r in gate['records'])
    summary=json.loads((out/'summary.json').read_text());assert summary['stage']=='completed'
    expected=len(spec['seeds'])*len(spec['evaluation']['conditions'])*2*len(spec['evaluation']['methods'])
    cells=sorted(out.glob('seed*/episodes.jsonl'));assert len(cells)==expected,(len(cells),expected)
    count=0;step_count=0;identity_count=0;hashes={}
    for cell in cells:
        erows=read_jsonl(cell);srows=read_jsonl(cell.parent/'steps.jsonl');mrows=read_jsonl(cell.parent/'messages.jsonl')
        assert len(erows)==spec['evaluation_episodes']
        assert {r['reset_seed'] for r in erows}==set(range(spec['heldout_seed_base'],spec['heldout_seed_base']+spec['evaluation_episodes']))
        for ep in erows:
            steps=[r for r in srows if r['reset_seed']==ep['reset_seed']];messages=[r for r in mrows if r['reset_seed']==ep['reset_seed']]
            assert len(steps)==ep['length'] and [r['t'] for r in steps]==list(range(ep['length']))
            assert np.isclose(sum(r['reward'] for r in steps),ep['return_'],atol=1e-8)
            assert steps[-1]['terminated'] or steps[-1]['truncated']
            assert ep['offered_bytes']==sum(r['wire_bytes'] for r in messages)
            assert ep['delivered_payload_bytes']==sum(r['payload_bytes'] for r in messages if r['accepted'])
            for row in steps:
                ms=[m for m in messages if m['t']==row['t']]
                accepted=sorted(m['sender'] for m in ms if m['accepted'])
                assert row['delivered_senders']==accepted
                assert row['K_attempt']==len(ms) and row['K_delivered']==len(accepted)
                assert row['wire_bytes']==48*len(ms) and row['payload_goodput_bytes']==32*len(accepted)
            if 'identity' in ep:
                identity_count+=1
                passed=ep['identity']['passed']
                if passed:
                    assert ep['identity']['first_mismatch'] is None
                    assert all(n==len(steps) for n in ep['identity']['matching_steps'].values())
            count+=1;step_count+=len(steps)
        recomputed=seed_summary(erows);stored=json.loads((cell.parent/'summary.json').read_text())
        for k,v in recomputed.items():
            if v is None:assert stored[k] is None
            else:assert np.isclose(v,stored[k],atol=1e-9),(cell,k)
        for path in [cell,cell.parent/'steps.jsonl',cell.parent/'messages.jsonl']:
            hashes[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
    assert identity_count==len(spec['seeds'])*len(spec['evaluation']['conditions'])*spec['evaluation_episodes']
    for seed in spec['seeds']:
        replay=json.loads((out/f'seed{seed}_replay_check.json').read_text());assert replay['passed']
        assert replay['first']['trace_sha256']==replay['repeat']['trace_sha256']
    result=dict(passed=True,require_learning_gate=require_gate,cells=len(cells),episodes=count,steps=step_count,identity_episodes=identity_count,raw_sha256=hashes)
    atomic_json(root/'raw_audit.json',result);print({k:v for k,v in result.items() if k!='raw_sha256'});return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();audit(load_config(a.config),not a.smoke)
