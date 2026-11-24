"""Local run registration and artifact helpers; never writes outside this component."""
import csv
import hashlib
import json
import platform
import sys
import os
from datetime import datetime, timezone
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent

def load_config(path):
    return yaml.safe_load(Path(path).read_text())

def hash_file(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def output_dir(name):
    path = (ROOT/'runs'/name).resolve()
    if not path.is_relative_to(ROOT/'runs'): raise ValueError('Output must be inside component runs/')
    path.mkdir(parents=True,exist_ok=False)
    return path

def save_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def save_csv(path,rows):
    if not rows: raise ValueError('Cannot write empty CSV')
    with Path(path).open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def manifest(config, phase, seed, checkpoint=None):
    import torch
    from .adapters.hallway import SOURCE
    sources = list(ROOT.rglob('*.py')) + [SOURCE,ROOT.parents[1]/'models/shared/agents/rnn_agent.py',
        ROOT.parents[1]/'models/shared/buffers/episode.py',ROOT.parents[1]/'models/shared/learners/td_learner.py',
        ROOT.parents[1]/'models/qmix/interface.py']
    return dict(status='running',phase=phase,created_utc=datetime.now(timezone.utc).isoformat(),
        config=config,config_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),seed=seed,python=sys.version,torch=torch.__version__,platform=platform.platform(),
        cpu=platform.processor() or next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), 'unknown'),affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
        threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
        source_hashes={str(p):hash_file(p) for p in sources},
        checkpoint=str(checkpoint) if checkpoint else None,checkpoint_hash=hash_file(checkpoint) if checkpoint else None,
        timing='measured CPU observation-to-action + virtual serial broadcast wait; no overlap',
        transport='full attempted packet charge; cutoff drops; flush at step boundary; no ACK',
        classification='local diagnostic VBC/TMC-style gates; not native paper reproductions')

def build_policy(config):
    from .policy import CommPolicy
    n = config['environment']['n_agents']
    return CommPolicy(2,2*n,n,3,**config['policy'])
