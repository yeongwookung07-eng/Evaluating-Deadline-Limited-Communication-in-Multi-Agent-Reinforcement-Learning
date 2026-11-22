"""Two GPU workers with a shared finite deadline; no automatic large sweep."""
import concurrent.futures,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
from ..common import load_config
from ..day2.io import atomic_json

BASE=Path('experiments/deadline_comm')
def main():
    s=load_config(BASE/'configs/smacv2_pilot.yaml');setup=load_config(BASE/'configs/smacv2_setup.yaml')
    assert json.loads((Path(setup['smoke_output'])/'status.json').read_text())['stage']=='completed'
    root=Path(s['output']);root.mkdir(exist_ok=False);deadline=time.time()+s['wall_hours']*3600
    runtime=Path(setup['runtime']).resolve();py=str(runtime/'venv/bin/python')
    jobs=[dict(race=race,arm=arm,seed=seed) for seed in s['seeds'] for j,race in enumerate(s['scenarios']) for arm in (s['arms'] if (seed+j)%2==0 else list(reversed(s['arms'])))]
    snapshot=root/'source_snapshot';snapshot.mkdir();hashes={}
    for p in sorted((BASE/'standard_benchmarks').glob('*.py')):
        data=p.read_bytes();(snapshot/p.name).write_bytes(data);hashes[str(p)]=hashlib.sha256(data).hexdigest()
    for p in [BASE/'configs/smacv2_pilot.yaml',BASE/'configs/smacv2_setup.yaml']:
        data=p.read_bytes();(snapshot/p.name).write_bytes(data);hashes[str(p)]=hashlib.sha256(data).hexdigest()
    freeze=subprocess.check_output([py,'-m','pip','freeze'],text=True);(root/'environment_freeze.txt').write_text(freeze)
    atomic_json(root/'manifest.json',dict(spec=s,jobs=jobs,deadline_epoch=deadline,started=time.time(),source_sha256=hashes,runtime_provenance=json.loads((runtime/'provenance.json').read_text()),scope='Bounded pilot; no automatic main benchmark expansion.'))
    def worker(device,jobs,phase):
        records=[]
        for job in jobs:
            if time.time()>=deadline:raise TimeoutError('registered campaign deadline')
            name=f"{job['race']}_seed{job['seed']}_{job['arm']}"
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(device),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',SC2PATH=str(runtime/'StarCraftII'),PYTHONUNBUFFERED='1')
            atomic_json(root/f'gpu{device}_status.json',dict(stage='running',phase=phase,job=job,completed=len(records)))
            with (root/(name+'.stdout.log')).open('w') as f:
                p=subprocess.Popen([py,'-m','experiments.deadline_comm.standard_benchmarks.train','--race',job['race'],'--arm',job['arm'],'--seed',str(job['seed']),'--device','cuda:0','--deadline',str(deadline)],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                try:p.wait(timeout=max(1,deadline-time.time()))
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid,signal.SIGTERM)
                    try:p.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                    raise
            if p.returncode:raise RuntimeError(f'{name} failed: {p.returncode}; see worker log')
            records.append(dict(job=job,summary=json.loads((root/name/'summary.json').read_text())))
            atomic_json(root/f'{phase}_gpu{device}_completed.json',records)
        atomic_json(root/f'gpu{device}_status.json',dict(stage='completed',completed=len(records)))
        return records
    def execute(selected,phase):
        atomic_json(root/'status.json',dict(stage='running',phase=phase,jobs=len(selected)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(worker,device,selected[i::2],phase) for i,device in enumerate(s['devices'])]
            with (root/'monitor.log').open('a',buffering=1) as monitor:
                while not all(f.done() for f in futures):
                    states=[]
                    for path in sorted(root.glob('*/status.json')):
                        d=json.loads(path.read_text());states.append(dict(job=path.parent.name,**d))
                    monitor.write(json.dumps(dict(time=time.time(),phase=phase,jobs=states))+'\n')
                    time.sleep(30)
            return [r for f in futures for r in f.result()]
    try:
        # User steering: learn Ideal first; do not run comparisons on unlearned tasks.
        ideal=execute([j for j in jobs if j['arm']=='ideal_full'],'ideal_learning_gate')
        gates={}
        for race in s['scenarios']:
            rr=[r['summary'] for r in ideal if r['job']['race']==race]
            gates[race]=all(r['learning_gate'] for r in rr) and sum(r['final_win_rate']-r['initial_win_rate'] for r in rr)>0
        atomic_json(root/'learning_gate.json',dict(passed_by_scenario=gates,ideal=ideal,criterion='All 3 Ideal seeds >=20% held-out wins and positive mean gain over initial; pilot only. Failed scenarios retained, not communication-effect evidence.'))
        comparisons=[j for j in jobs if j['arm']=='no_comm' and gates[j['race']]]
        results=ideal+(execute(comparisons,'no_comm_comparison') if comparisons else [])
        atomic_json(root/'summary.json',dict(stage='completed',jobs=results,learning_gates=gates,comparisons_started=len(comparisons),interpretation='Pilot gate only. Failure is not communication irrelevance; no selection for favorable seeds.'))
        atomic_json(root/'status.json',dict(stage='completed',jobs=len(results)))
    except Exception as exc:atomic_json(root/'status.json',dict(stage='failed',error=repr(exc)));raise
if __name__=='__main__':main()
