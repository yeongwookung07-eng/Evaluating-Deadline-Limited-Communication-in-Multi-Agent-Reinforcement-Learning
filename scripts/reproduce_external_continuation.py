"""Optional no-PID continuation from six separately supplied common checkpoints."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--run',action='store_true',help='Execute the registered bounded training; default only validates/prints commands.')
    args=parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=''
    for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
        os.environ[k]='1'
    source=ROOT/'experiments/deadline_comm/external_biases/runs/mnist_sum_parallel_20261123_001'
    config=yaml.safe_load((ROOT/'experiments/deadline_comm/configs/external_biases_mnist_parallel.yaml').read_text())
    base=yaml.safe_load((ROOT/config['base_config']).read_text())
    output=Path(args.output).resolve()
    assert not output.exists(),'Choose a new output; original evidence must not be overwritten.'
    paths={(seed,arm):source/f'seed{seed}_{arm}_1500.pt' for seed in base['seeds'] for arm in base['arms']}
    assert len(paths)==6 and all(p.is_file() for p in paths.values())
    commands=[[sys.executable,'-m','experiments.deadline_comm.external_biases.mnist_parallel_continuation',
               '--config',str(output/'continuation.yaml'),'--worker',str(seed),arm,str(path),'1500','3000']
              for (seed,arm),path in paths.items()]
    if not args.run:
        print(json.dumps({'mode':'dry_run_no_training','common_checkpoint_count':len(paths),'first_stage_commands':commands,'gate':'All three both-bias seeds pass validation before stopping; original no-bias controls advance equally.'},indent=2))
        return
    import torch
    from experiments.deadline_comm.external_biases.mnist_sum_learning import Policy, read_data, evaluate
    torch.set_num_threads(1)
    data,_=read_data(ROOT/base['data'])
    output.mkdir(parents=True)
    base['data']=str(ROOT/base['data'])
    copied_base=output/'base.yaml';copied_base.write_text(yaml.safe_dump(base))
    config.update(output=str(output),base_config=str(copied_base))
    copied_config=output/'continuation.yaml';copied_config.write_text(yaml.safe_dump(config))
    (output/'registration.json').write_text(json.dumps({'mode':'standalone no-PID continuation','config':config,'source_milestone':1500},indent=2))
    start=time.monotonic();latest=1500;passed=False;children=[]
    try:
        for target in config['milestones']:
            children=[];handles=[]
            try:
                for (seed,arm),path in paths.items():
                    log=(output/f'seed{seed}_{arm}_{target}.log').open('x');handles.append(log)
                    command=[sys.executable,'-m','experiments.deadline_comm.external_biases.mnist_parallel_continuation',
                             '--config',str(copied_config),'--worker',str(seed),arm,str(path),str(latest),str(target)]
                    children.append(subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT))
                for child in children:
                    remaining=config['wall_seconds']-(time.monotonic()-start)
                    if remaining<=0:raise TimeoutError('Registered continuation wall budget')
                    assert child.wait(timeout=remaining)==0,'Worker failed'
            finally:
                for child in children:
                    if child.poll() is None:child.terminate()
                for handle in handles:handle.close()
            rows=[json.loads((output/f'seed{seed}_{arm}_{target}_validation.json').read_text()) for seed,arm in paths]
            (output/f'validation_{target}.json').write_text(json.dumps(rows,indent=2))
            latest=target
            paths={(seed,arm):output/f'seed{seed}_{arm}_{target}.pt' for seed,arm in paths}
            passed=all(r['validation']['sampled']>.2 and r['validation']['expected']>r['validation']['shuffled'] for r in rows if r['arm']=='both')
            if passed:break
        images,labels=data['test'];records=[]
        for (seed,arm),path in paths.items():
            saved=torch.load(path,map_location='cpu',weights_only=True)
            speaker=Policy();listener=Policy(True)
            speaker.load_state_dict(saved['speaker']);listener.load_state_dict(saved['listener'])
            metrics=evaluate(speaker,listener,images,labels,torch.arange(10000),base['evaluation_pairs'],base['evaluation_seed']+1)
            records.append(dict(seed=seed,arm=arm,updates=latest,checkpoint=str(path),test=metrics))
        (output/'summary.json').write_text(json.dumps(dict(status='completed',validation_gate_pass=passed,selected_updates=latest,records=records),indent=2))
        (output/'status.json').write_text(json.dumps(dict(stage='completed',validation_gate_pass=passed)))
    except Exception as exc:
        (output/'status.json').write_text(json.dumps(dict(stage='failed_or_budget_exhausted',last_common_update=latest,error=repr(exc))))
        raise


if __name__=='__main__':main()
