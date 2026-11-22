"""Real engine continuation + all evaluation paths; never scientific learning evidence."""
import json,time
from pathlib import Path
from ..common import load_config
from ..day2.io import atomic_json
from .train import run as train
from .evaluate import run as evaluate

def main():
    spec=load_config('experiments/deadline_comm/configs/smacv2_protocol_smoke.yaml')
    checkpoint=json.loads((Path(spec['pilot'])/'terran_seed0_ideal_full/summary.json').read_text())['final_checkpoint']
    deadline=time.time()+1800
    train(spec,checkpoint,0,102000,'cuda:0',deadline)
    result=evaluate(spec,{'0':str(Path(spec['output'])/'phase_102000/seed0/checkpoint_final.pt')},deadline)
    atomic_json(Path(spec['output'])/'smoke_validation.json',dict(passed=True,scope='Technical smoke only; evaluation intentionally exercises all paths without learning-gate claim',evaluation_episodes=sum(m['n_seeds'] for m in result['means']),identity=result['identity']))
if __name__=='__main__':main()
