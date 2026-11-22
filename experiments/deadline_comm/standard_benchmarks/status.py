"""Read-only pilot status; no inference that completion means convergence."""
import json,time
from pathlib import Path
from ..common import load_config
def main():
    s=load_config('experiments/deadline_comm/configs/smacv2_pilot.yaml');root=Path(s['output'])
    if not root.exists():print('Pilot has not started.');return
    print(json.loads((root/'status.json').read_text()))
    for p in sorted(root.glob('*/status.json')):
        d=json.loads(p.read_text());print(p.parent.name,d)
    manifest=json.loads((root/'manifest.json').read_text())
    print('Registered campaign wall time remaining (not completion ETA):',round(max(0,manifest['deadline_epoch']-time.time())/3600,2),'hours')
if __name__=='__main__':main()
