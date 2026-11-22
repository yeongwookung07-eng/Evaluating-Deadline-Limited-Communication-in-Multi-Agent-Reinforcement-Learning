"""Recover only evaluation from saved checkpoints after the known watchdog bug."""
import json
from pathlib import Path
import time
from ..timing import configure_cpu
from ..common import load_config
from ..day2.io import atomic_json
from ..training_controls.run import evaluate


def main():
    configure_cpu()
    parent=Path('experiments/deadline_comm/runs/training_controls_20261118_001')
    while True:
        status=json.loads((parent/'status.json').read_text())
        if status['stage']=='completed':return
        if status['stage']=='failed':break
        time.sleep(10)
    if 'Registered campaign wall limit reached during evaluation' not in status.get('error',''):
        raise RuntimeError('Unexpected failure: '+str(status))
    registration=json.loads((parent/'registration.json').read_text());jobs=registration['jobs'];spec=registration['spec']
    if not all((parent/j['name']/'training_complete.json').exists() for j in jobs):raise RuntimeError('Training incomplete')
    out=parent.with_name(parent.name+'_evaluation_recovery');out.mkdir(exist_ok=False)
    for job in jobs:
        dest=out/job['name'];dest.mkdir()
        (dest/'training_complete.json').write_bytes((parent/job['name']/'training_complete.json').read_bytes())
    atomic_json(out/'registration.json',dict(parent=str(parent),reason='Watchdog variable shadowing; no retraining',jobs=jobs,spec=spec))
    atomic_json(out/'status.json',dict(stage='sequential_cpu_evaluation'))
    try:
        evaluate(spec,jobs,out,time.monotonic()+6*3600)
        atomic_json(out/'status.json',dict(stage='completed'))
    except Exception as exc:
        atomic_json(out/'status.json',dict(stage='failed',error=repr(exc)));raise

if __name__=='__main__':main()
