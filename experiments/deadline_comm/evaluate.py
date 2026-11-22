"""One CPU evaluation path for all five gates, with matched environment seeds."""
import argparse
from .timing import configure_cpu

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True)
    ap.add_argument('--checkpoint',required=True,nargs='+'); ap.add_argument('--run-name',required=True)
    args=ap.parse_args(); configure_cpu()
    import copy
    import json
    import time
    import torch
    from .common import load_config,output_dir,manifest,save_json,save_csv,build_policy,hash_file
    from .adapters.hallway import Hallway
    from .harness import rollout
    from .gates import METHODS
    from .analyze import analyze
    cfg=load_config(args.config); out=output_dir(args.run_name)
    meta=manifest(cfg,'evaluate',cfg['evaluation']['seed'],args.checkpoint[0]);
    meta['checkpoints']={p:hash_file(p) for p in args.checkpoint}; save_json(out/'manifest.json',meta)
    all_steps,all_messages,all_episodes=[],[],[]; start=time.perf_counter_ns()
    try:
        seen=set()
        for checkpoint in args.checkpoint:
            saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
            if saved['metadata']['policy']!=cfg['policy'] or saved['metadata']['environment']!=cfg['environment']:
                raise ValueError('Checkpoint architecture/environment does not match config')
            training_seed=saved['metadata']['seed']
            if training_seed in seen: raise ValueError('Duplicate training seed')
            seen.add(training_seed)
            model=build_policy(cfg); model.load_state_dict(saved['model']); model.eval()
            env=Hallway(**cfg['environment'])
            # Discard warmup states, caches and RNG; measured episodes construct fresh harnesses.
            for i in range(cfg['evaluation']['warmup_episodes']):
                for method in METHODS: rollout(env,model,method,cfg['harness'],900000+i,True)
            for condition in cfg['evaluation']['conditions']:
                hc=copy.deepcopy(cfg['harness']); hc['link'].update(condition.get('link',{}))
                for key in ('deadline_ns','r_post_ns'):
                    if key in condition: hc[key]=condition[key]
                for ep in range(cfg['evaluation']['episodes']):
                    seed=cfg['evaluation']['seed']+ep
                    # Rotate method ordering to reduce consistent first/last effects.
                    order=METHODS[ep%5:]+METHODS[:ep%5]
                    for method in order:
                        _,steps,messages,summary=rollout(env,model,method,hc,seed,True)
                        tags=dict(condition=condition['name'],episode=ep,seed=seed,training_seed=training_seed)
                        all_steps.extend([{**tags,**r} for r in steps]); all_messages.extend([{**tags,**r} for r in messages])
                        all_episodes.append({**tags,**summary})
        save_csv(out/'steps.csv',all_steps); save_csv(out/'episodes.csv',all_episodes)
        with (out/'messages.jsonl').open('w') as f:
            for row in all_messages: f.write(json.dumps(row,allow_nan=False)+'\n')
        analyze(out,cfg['evaluation'].get('budget_tolerance',.05))
        meta.update(status='completed',steps=len(all_steps),episodes=len(all_episodes),wall_ns=time.perf_counter_ns()-start)
    except Exception as exc:
        meta.update(status='failed',error=repr(exc)); raise
    finally: save_json(out/'manifest.json',meta)
    print(out)
if __name__=='__main__': main()
