"""Bounded non-instrumented training; outputs a CPU checkpoint for evaluation."""
import argparse
from time import perf_counter_ns
from .timing import configure_cpu

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',required=True); ap.add_argument('--run-name',required=True)
    ap.add_argument('--seed',type=int,default=0)
    args=ap.parse_args()
    configure_cpu()
    import torch
    from .common import load_config,output_dir,manifest,save_json,save_csv,build_policy
    from .adapters.hallway import Hallway
    from .gates import METHODS
    from .harness import rollout
    from .replay import CommReplay
    from .learner import CommLearner
    cfg=load_config(args.config); torch.manual_seed(args.seed)
    out=output_dir(args.run_name); meta=manifest(cfg,'train',args.seed); save_json(out/'manifest.json',meta)
    model=build_policy(cfg); learner=CommLearner(model,cfg['learner'])
    env=Hallway(**cfg['environment'],seed=args.seed)
    replay=CommReplay(cfg['training']['replay_capacity'],args.seed)
    rows=[]; transitions=0; started=perf_counter_ns()
    try:
        for ep in range(cfg['training']['episodes']):
            method=cfg['training'].get('method','mixed')
            if method=='mixed': method=METHODS[ep%len(METHODS)]
            record,_,_,summary=rollout(env,model,method,cfg['harness'],args.seed*10000+ep,False,cfg['training']['epsilon'])
            replay.insert(record); transitions+=len(record.episode.actions)
            metrics={}
            if len(replay)>=cfg['training']['batch_size']:
                metrics=learner.train_batch(replay.sample(cfg['training']['batch_size']))
            rows.append(dict(episode=ep,method=method,episode_return=summary['return'],transitions=transitions,
                td_loss=metrics.get('td_loss'),grad_norm=metrics.get('grad_norm'),updates=learner.updates))
        learner.save_checkpoint(out/'checkpoint.pt',{'policy':cfg['policy'],'environment':cfg['environment'],'seed':args.seed})
        save_csv(out/'training.csv',rows)
        meta.update(status='completed',transitions=transitions,updates=learner.updates,wall_ns=perf_counter_ns()-started,
            limitation='Training uses configured virtual ready time, not measured deployment CPU. Smoke is not convergence evidence.')
    except Exception as exc:
        meta.update(status='failed',error=repr(exc)); raise
    finally: save_json(out/'manifest.json',meta)
    print(out)
if __name__=='__main__': main()
