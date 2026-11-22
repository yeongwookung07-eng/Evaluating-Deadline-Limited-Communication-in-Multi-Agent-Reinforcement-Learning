"""Sequential final evaluation after implementation/smoke activity has finished."""
import json
from pathlib import Path
import time
from ..common import load_config
from ..timing import configure_cpu
from ..day2.io import atomic_json
from ..training_controls.run import evaluate


def main():
    configure_cpu()
    parent=Path('experiments/deadline_comm/runs/training_controls_20261118_001')
    recovery=parent.with_name(parent.name+'_evaluation_recovery')
    end=time.monotonic()+6*3600
    while True:
        status=json.loads((parent/'status.json').read_text())
        if status['stage']=='completed':break
        r=recovery/'status.json'
        if r.exists():
            state=json.loads(r.read_text())
            if state['stage']=='completed':break
            if state['stage']=='failed':raise RuntimeError(str(state))
        if time.monotonic()>end:raise TimeoutError('Recovery wait budget')
        time.sleep(10)
    data=json.loads((parent/'registration.json').read_text());spec=data['spec'];jobs=data['jobs']
    out=parent.with_name(parent.name+'_evaluation_verified_002');out.mkdir(exist_ok=False)
    for job in jobs:
        p=out/job['name'];p.mkdir();(p/'training_complete.json').write_bytes((parent/job['name']/'training_complete.json').read_bytes())
    atomic_json(out/'registration.json',dict(parent=str(parent),reason='Final sequential evaluation; development smoke overlapped early recovery',jobs=jobs,spec=spec))
    atomic_json(out/'status.json',dict(stage='sequential_cpu_evaluation'))
    try:
        evaluate(spec,jobs,out,time.monotonic()+6*3600)
        atomic_json(out/'status.json',dict(stage='completed'))
    except Exception as exc:atomic_json(out/'status.json',dict(stage='failed',error=repr(exc)));raise

if __name__=='__main__':main()
