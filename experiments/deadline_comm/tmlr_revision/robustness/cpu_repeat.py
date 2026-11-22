"""Independent-process repeated timing plus identical-input selector snapshots."""
import argparse,os,json,subprocess,sys,gc,time
from pathlib import Path
from copy import deepcopy
import torch
from ...common import load_config
from ...day2.io import atomic_json
from ...day2.finetune_comm_aware import load_model
from ...adapters.hallway import Hallway
from ...selection_validation.harness import SelectionHarness
from ...timing import configure_cpu
from ..readiness import run as rollout_run
from ..experiments import BASE,setup,clone,switch,step

def worker(s,rep,core):
    os.sched_setaffinity(0,{core});configure_cpu();root=Path(s['output'])/f'repeat{rep}';root.mkdir(exist_ok=False)
    atomic_json(root/'process.json',dict(pid=os.getpid(),repeat=rep,cpu_affinity=sorted(os.sched_getaffinity(0)),wall_time=time.time(),scope='same host/core, fresh process; not exclusive host'))
    config=dict(s,output=str(root/'rollouts'))
    if rep%2:config['seeds']=list(reversed(s['seeds']));config['checkpoints']=list(reversed(s['checkpoints']))
    rollout_run(config)
    snapshots=[]
    with torch.inference_mode():
        for seed in s['seeds']:
            for arm in s['checkpoints']:
                model,meta=load_model(BASE/f'runs/reviewer_ten_seeds_20261118_001/scratch_medium_2ms_seed{seed}_{arm}/checkpoint_1500.pt');model.eval()
                for ep in range(s['episodes']):
                    env=Hallway(**meta['environment']);obs,_=env.reset(seed=s['reset_base']+ep);h=SelectionHarness(model,'rr_cap',setup(['medium',2.25]),ep,False)
                    for t in range(env.episode_length):
                        if t in s['snapshot_steps']:
                            order=['rr_cap','dac'] if (seed+ep+t+rep)%2==0 else ['dac','rr_cap']
                            # One unrecorded warmup for each cloned selector; cloning is outside timing.
                            for selector in order:
                                c=clone(h);switch(c,selector);c.cfg=deepcopy(h.cfg);c.cfg['fixed_ready']=False;c.measured=True
                                c.step(obs,env.available_actions(),t)
                            gc.collect();enabled=gc.isenabled();gc.disable()
                            try:
                                for selector in order:
                                    c=clone(h);switch(c,selector);c.cfg=deepcopy(h.cfg);c.cfg['fixed_ready']=False;c.measured=True
                                    _,r,_=c.step(obs,env.available_actions(),t)
                                    snapshots.append(dict(seed=seed,weights=arm,episode=ep,t=t,selector=selector,repeat=rep,**{k:r[k] for k in ['ready_ns','gate_ns','cpu_total_ns','K_attempt','K_delivered','k_budget','k_link_effective','cap_elapsed_ns']}))
                            finally:
                                if enabled:gc.enable()
                        obs,tr=step(env,h,obs,t)
                        if tr['done']:break
    with (root/'snapshots.jsonl').open('w') as f:
        for r in snapshots:f.write(json.dumps(r)+'\n')
    atomic_json(root/'status.json',dict(stage='completed',snapshots=len(snapshots)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int);p.add_argument('--core',type=int);args=p.parse_args();s=load_config(BASE/'configs/tmlr_cpu_repeats.yaml')
    if args.worker is not None:return worker(s,args.worker,args.core)
    root=Path(s['output']);root.mkdir(exist_ok=False);core=min(os.sched_getaffinity(0));atomic_json(root/'registration.json',dict(**s,cpu_core=core));started=time.monotonic()
    try:
        for rep in range(s['repeats']):
            atomic_json(root/'status.json',dict(stage='running',repeat=rep,completed=rep))
            with (root/f'repeat{rep}.log').open('w') as f:
                subprocess.run([sys.executable,'-m',__package__+'.cpu_repeat','--worker',str(rep),'--core',str(core)],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=max(1,s['wall_hours']*3600-(time.monotonic()-started)),env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
        atomic_json(root/'status.json',dict(stage='completed',repeats=s['repeats']))
    except Exception as exc:atomic_json(root/'status.json',dict(stage='failed',error=repr(exc)));raise
if __name__=='__main__':main()
