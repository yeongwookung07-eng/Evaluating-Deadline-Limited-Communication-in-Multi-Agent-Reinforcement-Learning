"""Bounded day2 train -> calibration -> shared-weight matrix -> ablations."""
import argparse
import gc
import itertools
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from ..timing import configure_cpu
from .io import atomic_json,CSVLog,JSONLLog


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);args=ap.parse_args()
    configure_cpu()
    import numpy as np
    import torch
    import yaml
    from ..common import ROOT,load_config,output_dir,manifest,hash_file
    from ..adapters.hallway import Hallway
    from .harness import METHODS,ABLATIONS,rollout
    from .policy import DecayedPolicy
    spec=load_config(args.config);out=output_dir(spec['run_name'])
    status=manifest(spec,'day2_dac_main',spec['seeds'][0]);status.update(pid=os.getpid(),config_path=str(Path(args.config).resolve()),phase='initializing',completed_seeds=[])
    atomic_json(out/'manifest.json',status)
    all_condition_summaries=[]
    def progress(**values):
        status.update(values);atomic_json(out/'manifest.json',status)
    def controlled(env,model,method,hc,seed):
        gc.collect();was=gc.isenabled();gc.disable()
        try:return rollout(env,model,method,hc,seed,True)
        finally:
            if was:gc.enable()
    def summarize(episodes,steps):
        total=len(steps);attempt=sum(r['K_attempt'] for r in steps);byte_mean=float(np.mean([r['wire_bytes'] for r in steps]));valid_ages=[r['mean_age'] for r in steps if r['mean_age'] is not None]
        result=dict(episodes=len(episodes),steps=total,mean_return=float(np.mean([r['episode_return'] for r in episodes])),success_rate=float(np.mean([r['success'] for r in episodes])),mean_wire_bytes=byte_mean,
            target_budget_eligible=abs(byte_mean-spec['calibration']['target_bytes'])<=spec['calibration']['tolerance']*spec['calibration']['target_bytes'],
            miss_count=sum(r['action_miss'] for r in steps),action_miss_rate=float(np.mean([r['action_miss'] for r in steps])),
            message_drop_rate=sum(r['drop_count'] for r in steps)/max(attempt,1),mean_age=float(np.mean(valid_ages)) if valid_ages else None,
            invalid_cache_fraction=float(np.mean([r['invalid_cache_fraction'] for r in steps])),mean_K=float(np.mean([r['K_attempt'] for r in steps])),
            timings={key:dict(mean=float(np.mean([r[key] for r in steps])),p95=float(np.quantile([r[key] for r in steps],.95)),p99=float(np.quantile([r[key] for r in steps],.99))) for key in ['encode_ns','gate_ns','cache_ns','act_ns','other_ns','cpu_total_ns','comm_model_ns','step_total_ns']})
        for r in steps:
            assert abs(r['step_total_ns']-r['cpu_total_ns']-r['comm_model_ns'])<1e-5
            assert r['other_ns']>=0
            if r['method'].startswith('dac'):assert r['K_attempt']<=r['k_max']
        return result
    try:
        for seed in spec['seeds']:
            seedout=out/f'seed{seed}';seedout.mkdir()
            progress(phase='training',seed=seed)
            train_cfg={**spec['training'],'seed':seed,'output_name':f"{spec['run_name']}/seed{seed}/train"}
            config_path=seedout/'train.yaml';config_path.write_text(yaml.safe_dump(train_cfg,sort_keys=False))
            envvars={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
            with (seedout/'training_stdout.log').open('x') as log:
                subprocess.run([sys.executable,'-m','experiments.deadline_comm.day2.train_shared','--config',str(config_path)],cwd=ROOT.parents[1],env=envvars,stdout=log,stderr=subprocess.STDOUT,check=True)
            checkpoint=seedout/'train'/f"checkpoint_{spec['training']['episodes']:04}.pt"
            saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
            metadata=saved['metadata'];n=metadata['environment']['n_agents']
            model=DecayedPolicy(2,2*n,n,3,**metadata['policy']);model.load_state_dict(saved['model']);model.eval()
            env=Hallway(**metadata['environment']);hc=deepcopy(spec['harness'])
            seedmeta=dict(seed=seed,checkpoint=str(checkpoint),checkpoint_hash=hash_file(checkpoint),shared_checkpoint_for_all_methods=True)
            atomic_json(seedout/'checkpoint.json',seedmeta)
            progress(phase='calibration',seed=seed)
            for method in METHODS:
                for ep in range(spec['evaluation']['warmup_episodes']):controlled(env,model,method,hc,2500000+ep)
            calibration=[];cal_log=JSONLLog(seedout/'calibration_candidates.jsonl');selected={}
            for method,key,increasing,upper in [('random','p',True,1.),('vbc','q_threshold',True,20.),('tmc','message_threshold',False,20.),('dac','threshold',False,20.)]:
                candidates=[];lo=0.;hi=upper
                for iteration in range(spec['calibration']['iterations']+2):
                    value=lo if iteration==0 else hi if iteration==1 else (lo+hi)/2
                    candidate=deepcopy(hc);candidate['dac' if method=='dac' else 'gate'][key]=value
                    ss=[]
                    for ep in range(spec['calibration']['episodes']):
                        rows,_,_=controlled(env,model,method,candidate,spec['calibration']['seed_start']+seed*10000+ep);ss.extend(rows)
                    rate=float(np.mean([r['wire_bytes'] for r in ss]));target=spec['calibration']['target_bytes']
                    row=dict(seed=seed,method=method,iteration=iteration,parameter=key,value=value,mean_wire_bytes=rate,error=abs(rate-target))
                    candidates.append(row);cal_log.append([row])
                    if iteration>=2:
                        if (rate<target)==increasing:lo=value
                        else:hi=value
                best=min(candidates,key=lambda r:r['error']);hc['dac' if method=='dac' else 'gate'][key]=best['value'];selected[method]={**best,'eligible':best['error']<=target*spec['calibration']['tolerance']}
                atomic_json(seedout/'calibration.json',dict(selected=selected,frozen_harness=hc));print('CALIBRATION',seed,method,selected[method],flush=True)
            cal_log.close();atomic_json(seedout/'frozen_harness.json',hc)
            pairs=list(itertools.product(spec['evaluation']['links'],spec['evaluation']['deadlines_ms']))
            jobs=[('main',link,deadline,list(METHODS)) for link,deadline in pairs]
            jobs += [('ablation',link,deadline,list(ABLATIONS[1:])) for link,deadline in spec['evaluation']['ablations']]
            for category,link,deadline,methods in jobs:
                label=f'{category}_{link}_{deadline:g}ms';conditionout=seedout/label;conditionout.mkdir()
                local=deepcopy(hc);local['deadline_ns']=int(deadline*1e6);local['link']['bandwidth_bytes_s']=spec['evaluation']['links'][link]
                theoretical_capacity=max(0,int((local['deadline_ns']-local['r_post_ns']-local['link']['latency_ns'])*1e-9*local['link']['bandwidth_bytes_s']//48))
                progress(phase='evaluation',seed=seed,condition=label)
                sl=CSVLog(conditionout/'steps.csv');el=CSVLog(conditionout/'episodes.csv');ml=JSONLLog(conditionout/'messages.jsonl')
                steps={m:[] for m in methods};eps={m:[] for m in methods}
                try:
                    for repeat in range(spec['evaluation']['repeats']):
                        for ep in range(spec['evaluation']['episodes']):
                            offset=(repeat+ep)%len(methods);order=methods[offset:]+methods[:offset]
                            for method in order:
                                rows,msg,summary=controlled(env,model,method,local,spec['evaluation']['seed_start']+ep)
                                tags=dict(training_seed=seed,category=category,condition=f'{link}_{deadline:g}ms',repeat=repeat,episode=ep)
                                sl.append([{**tags,**r} for r in rows]);el.append([{**tags,**summary}]);ml.append([{**tags,**r} for r in msg])
                                steps[method].extend(rows);eps[method].append(summary)
                    condition_rows=[]
                    for method in methods:
                        aggregate=summarize(eps[method],steps[method])
                        aggregate.update(training_seed=seed,category=category,condition=f'{link}_{deadline:g}ms',method=method,
                            theoretical_no_cpu_packet_cap=theoretical_capacity,
                            target_infeasible_under_link_cap=method.startswith('dac') and method!='dac_no_cap' and theoretical_capacity*48<114)
                        condition_rows.append(aggregate);all_condition_summaries.append(aggregate)
                    atomic_json(conditionout/'summary.json',condition_rows);atomic_json(out/'seed_condition_summaries.json',all_condition_summaries)
                    aggregate_seeds(out,all_condition_summaries)
                    print('CONDITION_DONE',seed,label,flush=True)
                finally:sl.close();el.close();ml.close()
            status['completed_seeds'].append(seed);progress(phase='seed_completed',seed=seed)
        progress(status='completed',phase='completed')
    except Exception as exc:
        progress(status='failed',error=repr(exc));raise


def aggregate_seeds(out,records):
    import numpy as np
    result=[]
    for key in sorted({(r['category'],r['condition'],r['method']) for r in records}):
        rows=[r for r in records if (r['category'],r['condition'],r['method'])==key]
        metrics={}
        for metric in ['mean_return','success_rate','mean_wire_bytes','action_miss_rate','message_drop_rate','invalid_cache_fraction','mean_age']:
            values=[r[metric] for r in rows if r[metric] is not None]
            metrics[metric]=dict(mean=float(np.mean(values)) if values else None,seed_sample_sd=float(np.std(values,ddof=1)) if len(values)>1 else None)
        result.append(dict(category=key[0],condition=key[1],method=key[2],completed_seeds=[r['training_seed'] for r in rows],all_seeds_budget_eligible=all(r['target_budget_eligible'] for r in rows),metrics=metrics))
    atomic_json(out/'aggregate.json',result)

if __name__=='__main__':main()
