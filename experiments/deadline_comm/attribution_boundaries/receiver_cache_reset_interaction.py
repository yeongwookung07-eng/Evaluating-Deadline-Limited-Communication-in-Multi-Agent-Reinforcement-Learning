"""Genuine common-prefix receiver reset crossed with subsequent RR/DAC access."""
import argparse
from collections import defaultdict, Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import statistics as st
import time
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
import yaml
from scipy.stats import t as student_t
from ..tmlr_revision.experiments import clone,setup,step,suffix,switch
from ..adapters.hallway import Hallway
from ..day2.finetune_comm_aware import load_model
from ..selection_validation.harness import SelectionHarness
from ..timing import configure_cpu


def nonreceiver_bytes(h):
    # Compare the same cloned object's complete remaining state before/after mutation.
    return pickle.dumps({k:v for k,v in vars(h).items() if k not in ['cache','source','model']})


def reset_receiver(h):
    before=nonreceiver_bytes(h)
    h.cache.zero_();h.source.fill_(-1)
    assert nonreceiver_bytes(h)==before,'Receiver intervention modified other state'
    assert not h.cache.any() and (h.source==-1).all()


def ci(values):
    values=list(values);m=st.mean(values);d=student_t.ppf(.975,len(values)-1)*st.stdev(values)/math.sqrt(len(values)) if len(values)>1 else 0
    return dict(mean=m,t95=[float(m-d),float(m+d)],seed_values=values,n_seeds=len(values))


def main(config):
    c=yaml.safe_load(Path(config).read_text());out=Path(c['output']);out.mkdir(exist_ok=False);(out/'registration.json').write_text(json.dumps(c,indent=2)+'\n')
    configure_cpu();torch.set_num_threads(c['threads']);start=time.monotonic();original={}
    source=Path(c['source_prefix'])
    for line in source.open():
        r=json.loads(line)
        if r['condition']=='slow_5ms' and r['checkpoint_arm']==c['checkpoint_arm']:original[r['seed'],r['episode']]=r
    results=[];checks=0;hashes={str(source):hashlib.sha256(source.read_bytes()).hexdigest()}
    try:
      with (out/'episodes.jsonl').open('w') as log,torch.inference_mode():
        for seed in c['seeds']:
          ck=Path(c['checkpoint_template'].format(seed=seed));hashes[str(ck)]=hashlib.sha256(ck.read_bytes()).hexdigest();model,meta=load_model(ck);model.eval()
          for episode in range(c['episodes']):
            if time.monotonic()-start>c['wall_seconds']:raise TimeoutError('Registered CPU budget exhausted')
            saved=original[seed,episode];reset=c['reset_base']+episode;env=Hallway(**meta['environment']);obs,_=env.reset(seed=reset);h=SelectionHarness(model,'rr_cap',setup(c['condition']),reset,False)
            prefix=[]
            for t in range(len(saved['prefix'])):
                obs,tr=step(env,h,obs,t);prefix.append(tr)
            assert prefix==saved['prefix'],'Genuine prefix replay differs'
            survived=bool((h.source>=0).all()) and not prefix[-1]['done'];assert survived==saved['survived']
            branches={};cache_before=h.cache.clone();source_before=h.source.clone()
            for memory in ['retained','reset']:
              for selector in ['rr_cap','dac']:
                hh=clone(h)
                if memory=='reset':reset_receiver(hh)
                switch(hh,selector)
                value,trace=suffix(deepcopy(env),hh,obs.copy(),len(prefix)) if survived else (0.,[])
                if memory=='retained':
                    assert trace==saved['paired'][selector]['trace'] and value==saved['paired'][selector]['suffix_return'],'Retained no-op suffix differs';checks+=1
                branches[memory+'_'+selector]=dict(suffix_return=value,total_return=sum(v['reward'] for v in prefix)+value,trace=trace)
            assert torch.equal(h.cache,cache_before) and torch.equal(h.source,source_before),'Branch leaked into factual prefix'
            get=lambda m,a:branches[m+'_'+a]['suffix_return']
            retained=get('retained','dac')-get('retained','rr_cap');resetdiff=get('reset','dac')-get('reset','rr_cap')
            actions=lambda m,a:[q['actions'] for q in branches[m+'_'+a]['trace']]
            metrics=dict(seed=seed,episode=episode,survived=survived,prefix_steps=len(prefix),retained_selector_contrast=retained,reset_selector_contrast=resetdiff,interaction=resetdiff-retained,rr_reset_effect=get('reset','rr_cap')-get('retained','rr_cap'),dac_reset_effect=get('reset','dac')-get('retained','dac'),reset_changes_rr_return=get('reset','rr_cap')!=get('retained','rr_cap'),reset_changes_dac_return=get('reset','dac')!=get('retained','dac'),reset_changes_rr_actions=actions('reset','rr_cap')!=actions('retained','rr_cap'),reset_changes_dac_actions=actions('reset','dac')!=actions('retained','dac'),reset_selector_changes_return=resetdiff!=0)
            log.write(json.dumps(dict(**metrics,branches=branches))+'\n');log.flush();metrics['suffix_returns']={k:v['suffix_return'] for k,v in branches.items()};results.append(metrics)
          (out/'status.json').write_text(json.dumps(dict(stage='running',episodes=len(results),seed=seed)))
      keys=[k for k in results[0] if k not in ['seed','episode','survived','prefix_steps','suffix_returns']];summaries={}
      for subset in ['all_episodes','prefix_survivors']:
        rr=[r for r in results if subset=='all_episodes' or r['survived']]
        summaries[subset]={k:ci(st.mean(r[k] for r in rr if r['seed']==s) for s in c['seeds'] if any(r['seed']==s for r in rr)) for k in keys}
      result=dict(status='completed',episodes=len(results),survivors=sum(r['survived'] for r in results),retained_noop_checks=checks,summary=summaries,counts={k:sum(r[k] for r in results) for k in keys if isinstance(results[0][k],bool)},suffix_return_histograms={b:dict(Counter(r['suffix_returns'][b] for r in results)) for b in results[0]['suffix_returns']},elapsed_seconds=time.monotonic()-start,scope='Receiver memory invalidation at the same reached joint-history boundary; not cold-start mediation. Source timestamps encode validity and age; contents and metadata change jointly. Fixed readiness, modeled communication cost; no new CPU timing claim.')
      (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');(out/'input_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n');(out/'status.json').write_text(json.dumps(dict(stage='completed',episodes=len(results),noop_checks=checks),indent=2)+'\n');print(json.dumps(result),flush=True)
    except Exception as exc:
      (out/'status.json').write_text(json.dumps(dict(stage='failed',error=repr(exc),episodes=len(results),elapsed_seconds=time.monotonic()-start),indent=2)+'\n');raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);main(ap.parse_args().config)
