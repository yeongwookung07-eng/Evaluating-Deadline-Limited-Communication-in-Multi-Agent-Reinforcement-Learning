"""User-requested Q1/Q2 first; no broad extensions before these judgments."""
import os
import subprocess
import sys
from pathlib import Path
from ..day2.io import atomic_json

root=Path('experiments/deadline_comm/selection_validation')
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
commands=[('q1',[sys.executable,'-u','-m','experiments.deadline_comm.reviewer_validation.evaluate_endpoints','--config','experiments/deadline_comm/configs/q1_priority.yaml']),
          ('q2',[sys.executable,'-u','-m','experiments.deadline_comm.selection_validation.evaluate','--config','experiments/deadline_comm/configs/q2_priority.yaml']),
          ('ten_seed',[sys.executable,'-u','-m','experiments.deadline_comm.reviewer_validation.train_ten_seeds','--config','experiments/deadline_comm/configs/reviewer_ten_seeds.yaml','--output','experiments/deadline_comm/runs/reviewer_ten_seeds_20261118_001'])]
try:
 for stage,cmd in commands:
  atomic_json(root/'priority_status.json',dict(stage=stage))
  with (root/f'priority_{stage}.log').open('x') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
 atomic_json(root/'priority_status.json',dict(stage='completed'))
except Exception as exc:atomic_json(root/'priority_status.json',dict(stage='failed',error=repr(exc)));raise
