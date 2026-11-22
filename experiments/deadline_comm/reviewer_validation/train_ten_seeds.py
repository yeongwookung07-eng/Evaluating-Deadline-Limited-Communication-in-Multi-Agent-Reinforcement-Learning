"""Ten-seed, three-condition Hallway controls; copied isolated training contract."""
import argparse
from copy import deepcopy
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

from ..timing import configure_cpu

ARMS = ('full_continuation', 'dac', 'dac_no_cap', 'dac_no_aoi', 'dac_no_decay')


def configure_arm(base, arm, link, deadline):
    cfg = deepcopy(base)
    cfg['link']['bandwidth_bytes_s'] = link
    cfg['deadline_ns'] = int(deadline*1e6)
    method = arm
    if arm in ('rr_cap','random_cap','age_cap'):
        cfg['selector']=arm
        method='dac'
    if arm in ('no_comm','ideal_full'):
        cfg['endpoint']=arm
        method='full'
    if arm == 'full_continuation':
        method = 'full'
        cfg['link'].update(bandwidth_bytes_s=1000000000, latency_ns=0)
        cfg['deadline_ns'] = 1000000000
    if not method.startswith('dac') or method == 'dac_no_decay':
        cfg['decay_lambda'] = 1.
    if method == 'dac_no_aoi':
        cfg['dac']['alpha'] = 0.
    return method, cfg


def trajectory(env, model, method, cfg, seed, oracle=False, epsilon=.05):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from framework.buffers.episode import Episode
    from ..replay import CommEpisode
    from ..hallway_oracle_check import oracle_actions
    from .endpoints import EndpointHarness as Day2Harness
    model.eval()
    obs, _ = env.reset(seed=seed)
    h = Day2Harness(model, method, cfg, seed+100000, measured=False)
    rng = np.random.default_rng(seed)
    observations, states, masks = [obs], [env.get_state()], [env.available_actions()]
    actions, rewards, terms, truncs, attempts, deliveries, sources = [], [], [], [], [], [], []
    for t in range(env.episode_length+1):
        a, _, messages = h.step(obs, masks[-1], t)
        attempt, delivery = np.zeros(env.n_agents, bool), np.zeros(env.n_agents, bool)
        for message in messages:
            attempt[message['sender']] = True
            delivery[message['sender']] = message['accepted']
        attempts.append(attempt); deliveries.append(delivery)
        sources.append(h.source[0].numpy().copy())
        if t == env.episode_length or (terms and (terms[-1] or truncs[-1])):
            break
        if oracle:
            a = oracle_actions(obs, env.env.group_ids)
        else:
            for i in range(env.n_agents):
                if rng.random() < epsilon:
                    a[i] = rng.choice(np.flatnonzero(masks[-1][i]))
        h.previous = F.one_hot(torch.as_tensor(a), model.n_actions).float()[None]
        obs, reward, term, trunc, _ = env.step(a)
        actions.append(a); rewards.append(reward); terms.append(term); truncs.append(trunc)
        observations.append(obs); states.append(env.get_state()); masks.append(env.available_actions())
    ep = Episode(np.array(observations,np.float32),np.array(states,np.float32),np.array(masks,bool),
                 np.array(actions),np.array(rewards,np.float32),np.array(terms),np.array(truncs))
    record = CommEpisode(ep,np.array(attempts),np.array(deliveries),np.array(sources),
                        np.zeros((env.n_agents,model.message_dim),np.float32),np.full(env.n_agents,-1))
    record.oracle_source = oracle
    return record


def controlled(fn, *args, **kwargs):
    gc.collect()
    enabled = gc.isenabled(); gc.disable()
    try:
        return fn(*args, **kwargs)
    finally:
        if enabled: gc.enable()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker(spec, job, out, device):
    import torch
    from ..common import load_config
    from ..day2.finetune_comm_aware import load_model
    from ..day2.policy import DecayedPolicy
    from ..day2.io import atomic_json, JSONLLog
    from ..oracle_bc_learner import OracleBCLearner
    from ..replay import StratifiedReplay
    from ..adapters.hallway import Hallway
    seed, arm, family = job['seed'], job['arm'], job['family']
    torch.manual_seed(seed)
    checkpoint = Path(spec['base_run'])/'seed0/train/checkpoint_1500.pt'  # metadata only for fresh initialization
    if family == 'continuation':
        model, metadata = load_model(checkpoint)
    else:
        saved = torch.load(checkpoint,map_location='cpu',weights_only=True)
        metadata = deepcopy(saved['metadata'])
        n = metadata['environment']['n_agents']
        model = DecayedPolicy(2,2*n,n,3,**metadata['policy'])
    base = load_config('experiments/deadline_comm/configs/day2_dac_marl_main.yaml')['harness']
    _, arm_cfg = configure_arm(base,arm,20000,5.)
    model.decay_lambda = arm_cfg['decay_lambda']
    learner_cfg = load_config('experiments/deadline_comm/configs/timing_rank.yaml')['learner']
    learner_cfg.update(learning_rate=1e-4 if family=='continuation' else 5e-4,
                       gamma=.99, grad_clip=10., target_update_interval=100,
                       bootstrap_timeouts=False,bc_weight=.5)
    initial_hash = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    model.to(device)
    learner = OracleBCLearner(model,learner_cfg)
    assert learner.target.decay_lambda == model.decay_lambda
    rollout_model = deepcopy(model).cpu()
    env = Hallway(**metadata['environment'])
    replay = StratifiedReplay(2000,seed)
    metadata = deepcopy(metadata)
    metadata.update(seed=seed,training_control=job,decay_lambda=model.decay_lambda,
                    starting_checkpoint_sha256=digest(checkpoint) if family=='continuation' else None,
                    initial_parameter_sha256=initial_hash,learner=learner_cfg)
    atomic_json(out/'manifest.json',dict(job=job,device=device,metadata=metadata,status='running'))
    def collect(index, oracle=False):
        rollout_model.load_state_dict(model.state_dict())
        schedules = [(100000,5.),(20000,5.),(100000,2.)] if family=='continuation' else [tuple(job['training_condition'])]
        link,deadline = schedules[index%len(schedules)]
        method,cfg = configure_arm(base,arm,link,deadline)
        epsilon = .05 if family=='continuation' else 1.-.95*max(0,index-100)/max(1,job['updates']-1)
        return controlled(trajectory,env,rollout_model,method,cfg,
                          spec['training_seed_start']+seed*10000+index,oracle,epsilon)
    online_transitions=0
    start = time.monotonic()
    for i in range(spec['oracle_episodes']): replay.insert(collect(i,True))
    log = JSONLLog(out/'training.jsonl')
    try:
        for update in range(job['updates']):
            record = collect(100+update); replay.insert(record)
            online_transitions+=len(record.episode.actions)
            batch = replay.sample(spec['batch_size'])
            assert sum(r.oracle_source for r in batch)==spec['batch_size']//4
            metrics = learner.train_batch(batch)
            log.append([dict(update=update+1,episode_return=float(record.episode.rewards.sum()),
                             seconds=time.monotonic()-start,online_transitions=online_transitions,batch_size=spec['batch_size'],oracle_batch_count=spec['batch_size']//4,**metrics)])
            if (update+1)%spec['checkpoint_interval']==0 or update+1==job['updates']:
                metadata['update'] = (1500 if family=='continuation' else 0)+update+1
                learner.save_checkpoint(out/f'checkpoint_{update+1:04}.pt',metadata)
                print(job['name'],update+1,metrics,'seconds',time.monotonic()-start,flush=True)
        atomic_json(out/'training_complete.json',dict(status='completed',seconds=time.monotonic()-start,
                    final_checkpoint=str(out/f"checkpoint_{job['updates']:04}.pt"),job=job))
    finally:
        log.close()


def evaluate(spec, jobs, out, wall_deadline=None):
    from ..common import load_config
    from ..day2.finetune_comm_aware import load_model
    from .rollouts import rollout
    from ..day2.aggregate_day2 import measure
    from ..day2.io import atomic_json,CSVLog,JSONLLog
    from ..adapters.hallway import Hallway
    base = load_config('experiments/deadline_comm/configs/day2_dac_marl_main.yaml')['harness']
    records=[]
    for job in jobs:
        job_out=out/job['name']
        saved=json.loads((job_out/'training_complete.json').read_text())
        model,meta=load_model(saved['final_checkpoint'])
        env=Hallway(**meta['environment'])
        arm=job['arm']
        deployments=['dac'] if arm=='full_continuation' else [arm]
        if job['family']=='continuation' and arm not in ('full_continuation','dac'): deployments.append('dac')
        for deployment in deployments:
            for link,deadline in spec['conditions']:
                method,cfg=configure_arm(base,deployment,spec['links'][link],deadline)
                # Common deployment restores DAC alpha/decay even for ablated training.
                destination=job_out/f'eval_{deployment}_{link}_{deadline:g}ms';destination.mkdir()
                for ep in range(spec['warmup_episodes']):
                    controlled(rollout,env,model,method,cfg,spec['evaluation_seed_start']-100+ep,True)
                sl,el,ml=CSVLog(destination/'steps.csv'),CSVLog(destination/'episodes.csv'),JSONLLog(destination/'messages.jsonl')
                steps,episodes=[],[]
                try:
                    for ep in range(spec['evaluation_episodes']):
                        if wall_deadline is not None and time.monotonic()>wall_deadline:
                            raise TimeoutError('Registered campaign wall limit reached during evaluation')
                        ss,mm,ee=controlled(rollout,env,model,method,cfg,spec['evaluation_seed_start']+ep,True)
                        sl.append([dict(episode=ep,**s) for s in ss]);el.append([dict(episode=ep,**ee)])
                        ml.append([dict(episode=ep,**m) for m in mm]);steps.extend(ss);episodes.append(ee)
                    row=dict(seed=job['seed'],family=job['family'],arm=arm,deployment=deployment,
                             condition=f'{link}_{deadline:g}ms',**measure(episodes,steps))
                    atomic_json(destination/'summary.json',row);records.append(row)
                    atomic_json(out/'evaluation_progress.json',records)
                    print('EVALUATED',job['name'],deployment,row['condition'],row['mean_return'],flush=True)
                finally: sl.close();el.close();ml.close()
    summarize(records,out)


def summarize(records,out):
    from ..day2.io import atomic_json
    from collections import defaultdict
    def csv_write(name,rows):
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    csv_write('by_seed.csv',records)
    groups=defaultdict(list)
    for row in records:groups[tuple(row[k] for k in ('family','arm','deployment','condition'))].append(row)
    aggregates=[]
    for key,rows in groups.items():
        r=dict(zip(('family','arm','deployment','condition'),key));r['n_seeds']=len(rows)
        for metric in ('mean_return','offered_B_step','goodput_payload_B_step','wasted_pct','cache_invalid_pct','drop_pct'):
            values=[v[metric] for v in rows if v[metric] is not None]
            r[metric]=statistics.mean(values) if values else None
            r[metric+'_sd']=statistics.stdev(values) if len(values)>1 else None
        r['all_budget_eligible']=all(v['budget_eligible'] for v in rows);aggregates.append(r)
    csv_write('aggregate.csv',aggregates)
    deltas=[]
    for row in records:
        if row['arm']=='dac':continue
        baseline=next((r for r in records if r['family']==row['family'] and r['seed']==row['seed'] and r['condition']==row['condition'] and r['arm']=='dac' and r['deployment']=='dac'),None)
        if baseline:deltas.append(dict(family=row['family'],seed=row['seed'],arm=row['arm'],deployment=row['deployment'],condition=row['condition'],dac_minus_arm_return=baseline['mean_return']-row['mean_return']))
    csv_write('paired_deltas.csv',deltas)
    from .stats import paired_bootstrap
    comparisons=[]
    for family,condition in sorted({(r['family'],r['condition']) for r in records}):
        for left,right in [('dac','periodic'),('dac','no_comm'),('ideal_full','no_comm'),('full','no_comm'),('dac','rr_cap')]:
            a={r['seed']:r['mean_return'] for r in records if r['family']==family and r['condition']==condition and r['arm']==left}
            b={r['seed']:r['mean_return'] for r in records if r['family']==family and r['condition']==condition and r['arm']==right}
            if a and b:comparisons.append(dict(family=family,condition=condition,left=left,right=right,**paired_bootstrap(a,b)))
    atomic_json(out/'paired_bootstrap.json',comparisons)
    csv_write('paired_ci.csv',comparisons)
    atomic_json(out/'summary.json',dict(status='completed',records=records,aggregates=aggregates,paired_deltas=deltas))
    lines=['# Training controls','', 'All seeds retained; SD across seeds. Final checkpoints only. Oracle-assisted scratch is not demonstration-free RL. Full scratch attempts all packets under the constrained link; Full continuation is lossless training followed by DAC evaluation. No native VBC/TMC training comparison.','', '|Family|Training arm|Deployment|Condition|Return mean ± SD|Offered B|Payload B|Waste %|Budget eligible|','|---|---|---|---|---|---|---|---|---|']
    for r in aggregates:
        sd=r['mean_return_sd'];w=r['wasted_pct']
        lines.append(f"|{r['family']}|{r['arm']}|{r['deployment']}|{r['condition']}|{r['mean_return']:.2f} ± {sd if sd is not None else 'NA'}|{r['offered_B_step']:.2f}|{r['goodput_payload_B_step']:.2f}|{w if w is not None else 'NA'}|{r['all_budget_eligible']}|")
    lines += ['', 'Interpretation: positive DAC-minus-arm return deltas favor DAC only in that paired seed/condition. Read paired_deltas.csv before any consistency claim. Different offered loads are not equal-budget evidence. Experimental outcomes do not automatically alter the manuscript.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def make_jobs(spec):
    jobs=[]
    for link,deadline in spec['training_conditions']:
        family=f'scratch_{link}_{deadline:g}ms'
        for seed in spec['seeds']:
            for arm in spec.get('methods',('full','periodic','dac','no_comm','ideal_full')):
                jobs.append(dict(name=f'{family}_seed{seed}_{arm}',family=family,seed=seed,arm=arm,
                                 training_condition=[spec['links'][link],deadline],updates=spec['scratch_updates']))
    return jobs


def main():
    configure_cpu()
    from ..common import load_config
    from ..day2.io import atomic_json
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--smoke',action='store_true');ap.add_argument('--worker');ap.add_argument('--device');a=ap.parse_args()
    spec=load_config(a.config);out=Path(a.output)
    if a.worker:
        job=json.loads(a.worker)
        try:worker(spec,job,out,a.device)
        except Exception:
            atomic_json(out/'failure.json',dict(error=traceback.format_exc()));raise
        return
    if not a.smoke and spec.get('preflight_selection_config') and not (Path(load_config(spec['preflight_selection_config'])['output'])/'summary.json').exists():
        subprocess.run([sys.executable,'-u','-m','experiments.deadline_comm.selection_validation.evaluate','--config',spec['preflight_selection_config']],check=True)
    if a.smoke:
        spec.update(seeds=[0],continuation_updates=2,scratch_updates=2,oracle_episodes=4,
                    evaluation_episodes=2,warmup_episodes=1,conditions=[['slow',5.]],training_conditions=[['slow',5.]],checkpoint_interval=2)
    out.mkdir(parents=True,exist_ok=False)
    import yaml
    resolved=out/'resolved.yaml';resolved.write_text(yaml.safe_dump(spec))
    jobs=make_jobs(spec)
    atomic_json(out/'registration.json',dict(spec=spec,jobs=jobs,total_updates=sum(j['updates'] for j in jobs),smoke=a.smoke,
               entrypoint_sha256=digest(__file__),registered_unix=time.time()))
    pending=list(jobs);running=[];start=time.monotonic();completed=[]
    env=os.environ.copy()
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
    try:
        while pending or running:
            if time.monotonic()-start>spec['wall_limit_hours']*3600:raise TimeoutError('Registered campaign wall limit reached')
            used={r[2] for r in running}
            for device in spec['devices']:
                if device in used or not pending:continue
                job=pending.pop(0);dest=out/job['name'];dest.mkdir();log=(dest/'stdout.log').open('x')
                command=[sys.executable,'-m','experiments.deadline_comm.reviewer_validation.train_ten_seeds','--config',str(resolved),'--output',str(dest),'--worker',json.dumps(job),'--device',device]
                p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
                running.append((p,job,device,log));print('START',job['name'],device,flush=True)
            for record in list(running):
                p,job,device,log=record;code=p.poll()
                if code is not None:
                    log.close();running.remove(record)
                    if code:raise RuntimeError(f"{job['name']} exited {code}")
                    completed.append(job['name']);print('DONE',job['name'],flush=True)
            atomic_json(out/'status.json',dict(stage='training',completed=completed,running=[r[1]['name'] for r in running],pending=len(pending),elapsed_seconds=time.monotonic()-start))
            time.sleep(2)
        atomic_json(out/'status.json',dict(stage='sequential_cpu_evaluation',training_completed=len(completed)))
        evaluate(spec,jobs,out,start+spec['wall_limit_hours']*3600)
        atomic_json(out/'status.json',dict(stage='completed',elapsed_seconds=time.monotonic()-start))
    except Exception:
        for p,_,_,log in running:
            if p.poll() is None:p.terminate()
        for p,_,_,log in running:p.wait();log.close()
        atomic_json(out/'status.json',dict(stage='failed',error=traceback.format_exc(),completed=completed));raise

if __name__=='__main__':main()
