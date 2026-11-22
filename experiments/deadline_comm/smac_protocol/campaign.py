"""Registered bounded learning gates followed by serial held-out evaluation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from ..common import load_config
from ..day2.io import atomic_json


def gate_pass(records,spec):
    return len(records)==len(spec['seeds']) and all(r['win_rate']>=spec['learning_gate_win_rate'] for r in records) and sum(r['win_rate'] for r in records)>0


def record_ledger(spec,status):
    line=f"\n2026-11-21 smacv2_protocol — **{status}**. Config `{spec['_config']}`; results `{spec['output']}`; bounded user-authorized SMACv2 transfer, no assumed success. See `experiments/deadline_comm/smac_protocol/README.md`.\n"
    for name in ['README.md','docs/research_experiment_journal.md']:
        with Path(name).open('a') as f:f.write(line)
    with Path('experiments/deadline_comm/smac_protocol/README.md').open('a') as f:f.write('\nCampaign status: **'+status+'**, artifacts `'+spec['output']+'`.\n')


def run(config):
    spec=load_config(config);spec['_config']=config
    root=Path(spec['output']);root.mkdir(parents=True,exist_ok=False)
    start=time.time();deadline=start+spec['wall_hours']*3600
    runtime=Path(spec['runtime']).resolve();py=str(runtime/'venv/bin/python')
    snapshot=root/'source_snapshot';snapshot.mkdir();hashes={}
    inputs=list(Path('experiments/deadline_comm/smac_protocol').glob('*.py'))+[Path(config),Path('experiments/deadline_comm/smac_protocol/README.md'),Path('experiments/deadline_comm/batched_learner.py'),Path('experiments/deadline_comm/standard_benchmarks/train.py'),Path('experiments/deadline_comm/standard_benchmarks/smac_adapter.py'),Path('experiments/deadline_comm/selection_validation/harness.py')]
    for i,p in enumerate(inputs):
        data=p.read_bytes();hashes[str(p)]=hashlib.sha256(data).hexdigest();(snapshot/f'{i}_{p.name}').write_bytes(data)
    checkpoints={str(seed):json.loads((Path(spec['pilot'])/f"{spec['scenario']}_seed{seed}_ideal_full/summary.json").read_text())['final_checkpoint'] for seed in spec['seeds']}
    atomic_json(root/'manifest.json',dict(spec=spec,start_epoch=start,deadline_epoch=deadline,source_hashes=hashes,initial_checkpoints=checkpoints,checkpoint_hashes={k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in checkpoints.items()},pid=os.getpid(),paper_source_sha256=hashlib.sha256(Path('experiments/deadline_comm/publication/kbs/kbs_main.tex').read_bytes()).hexdigest()))
    envbase=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',SC2PATH=str(runtime/'StarCraftII'),PYTHONUNBUFFERED='1')
    def execute(command,env,log):
        if time.time()>=deadline:raise TimeoutError('Campaign wall limit')
        with log.open('w') as f:
            child=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            try:child.wait(timeout=max(1,deadline-time.time()))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise
            if child.returncode:raise RuntimeError(f'Worker failed {child.returncode}: {log}')
    record_ledger(spec,'running')
    try:
        passed=False
        for milestone in spec['training_milestones']:
            phase=root/f'phase_{milestone}';phase.mkdir()
            atomic_json(root/'status.json',dict(stage='training',milestone=milestone,seconds=time.time()-start))
            def worker(device,seeds):
                result=[]
                for seed in seeds:
                    command=[py,'-m','experiments.deadline_comm.smac_protocol.train','--config',config,'--checkpoint',checkpoints[str(seed)],'--seed',str(seed),'--milestone',str(milestone),'--device','cuda:0','--deadline',str(deadline)]
                    execute(command,dict(envbase,CUDA_VISIBLE_DEVICES=str(device)),phase/f'seed{seed}.stdout.log')
                    result.append(json.loads((phase/f'seed{seed}/gate.json').read_text()))
                return result
            with ThreadPoolExecutor(max_workers=len(spec['devices'])) as pool:
                futures=[pool.submit(worker,device,spec['seeds'][i::len(spec['devices'])]) for i,device in enumerate(spec['devices'])]
                with (root/'monitor.log').open('a',buffering=1) as monitor:
                    while not all(f.done() for f in futures):
                        states={p.parent.name:json.loads(p.read_text()) for p in phase.glob('seed*/status.json')}
                        monitor.write(json.dumps(dict(stage='training',milestone=milestone,elapsed_seconds=time.time()-start,jobs=states))+'\n')
                        time.sleep(10)
                records=[r for f in futures for r in f.result()]
            passed=gate_pass(records,spec)
            atomic_json(phase/'learning_gate.json',dict(passed=passed,criterion=spec['learning_gate_win_rate'],records=records))
            checkpoints={str(seed):str(phase/f'seed{seed}/checkpoint_final.pt') for seed in spec['seeds']}
            if passed:break
        atomic_json(root/'learning_gate.json',dict(passed=passed,milestone=milestone,records=records,checkpoint_selection='First common predeclared milestone passing all seeds; test reset set is disjoint'))
        if not passed:
            atomic_json(root/'summary.json',dict(stage='completed_gate_failed',scheduler_evaluation_started=False,records=records,interpretation='Learning criterion failed within registered budget; no communication-effect conclusion'))
            (root/'RESULTS_KO.md').write_text('# SMACv2 학습 기준 미통과\n\n등록된 최대 학습 예산에서 3-seed 공통 기준을 통과하지 못했습니다. scheduler 효과를 검증했다고 주장할 수 없습니다. learning_gate.json과 각 단계 원시 로그를 보존했습니다.\n')
            atomic_json(root/'status.json',dict(stage='completed_gate_failed',seconds=time.time()-start));record_ledger(spec,'completed; learning gate failed');return
        atomic_json(root/'checkpoints.json',checkpoints)
        atomic_json(root/'status.json',dict(stage='heldout_evaluation',milestone=milestone,seconds=time.time()-start))
        execute([py,'-m','experiments.deadline_comm.smac_protocol.evaluate','--config',config,'--checkpoints',str(root/'checkpoints.json'),'--deadline',str(deadline)],dict(envbase,CUDA_VISIBLE_DEVICES=''),root/'evaluation.stdout.log')
        execute([py,'-m','experiments.deadline_comm.smac_protocol.audit','--config',config],dict(envbase,CUDA_VISIBLE_DEVICES=''),root/'audit.stdout.log')
        execute([py,'-m','experiments.deadline_comm.smac_protocol.paper','--config',config],dict(envbase,CUDA_VISIBLE_DEVICES=''),root/'paper.stdout.log')
        result=json.loads((root/'evaluation/summary.json').read_text())
        atomic_json(root/'summary.json',dict(stage='completed',learning_gate_passed=True,evaluation=result))
        atomic_json(root/'status.json',dict(stage='completed',seconds=time.time()-start));record_ledger(spec,'completed; see actual evaluation results')
    except Exception as exc:
        atomic_json(root/'status.json',dict(stage='failed',error=repr(exc),seconds=time.time()-start));record_ledger(spec,'failed; retained logs');raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();run(a.config)
