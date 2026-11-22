"""Episode-disjoint, receiver-specific linear probes of actual action-head inputs."""
import argparse,json,time
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
import numpy as np
import torch
from ...common import load_config
from ...timing import configure_cpu
from ...day2.io import atomic_json
from ...day2.policy import DecayedPolicy
from ...day2.dac import decay_messages
from ...reviewer_validation.hidden_source import HiddenSource
from ...reviewer_validation.train_ten_seeds import configure_arm
from ...reviewer_validation.endpoints import EndpointHarness
from ..experiments import BASE,setup,clone,step
from ..editorial_audit import lines

def ridge_prediction(train_x,train_y,test_x,ridge):
    mean=train_x.mean(0);scale=np.maximum(train_x.std(0),1e-6)
    x=np.c_[(train_x-mean)/scale,np.ones(len(train_x))];z=np.c_[(test_x-mean)/scale,np.ones(len(test_x))]
    penalty=np.eye(x.shape[1])*ridge;penalty[-1,-1]=0
    return z@np.linalg.solve(x.T@x+penalty,x.T@train_y)>=.5

def main():
    configure_cpu();p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');args=p.parse_args();s=load_config(BASE/'configs/tmlr_receiver_probe.yaml')
    if args.smoke:s.update(output=s['output']+'_smoke',seeds=[0],methods=['ideal_full','rr_cap'],conditions=[['slow',5.]],episodes=4)
    out=Path(s['output']);out.mkdir(exist_ok=False);atomic_json(out/'registration.json',s);expected={};records=[];start=time.monotonic()
    for r in lines(BASE/'runs/tmlr_broadcast_20261119_001/steps.jsonl'):
        if r['seed'] in s['seeds'] and r['method'] in s['methods'] and r['episode']<s['episodes']:
            expected[r['seed'],r['method'],r['condition'],r['scheduler'],r['episode'],r['t']]=(r['bit'],r['followers_accuracy'],r['k'],r['source_delivered'])
    checked=0
    try:
        with torch.inference_mode():
            for seed in s['seeds']:
                for method in s['methods']:
                    model=DecayedPolicy(4,10,5,2,hidden_dim=64,message_dim=8,mixer='qmix')
                    model.load_state_dict(torch.load(BASE/f'runs/hidden_mechanism_20261119_001/broadcast_n5_seed{seed}_{method}/final.pt',map_location='cpu',weights_only=True)['model']);model.eval()
                    captured={}
                    def hook(_,inputs):captured['x']=inputs[0].detach().cpu().numpy().copy()
                    handle=model.utility.register_forward_pre_hook(hook)
                    try:
                        for link,d in s['conditions']:
                            cond=f'{link}_{d:g}ms';actual,cfg=configure_arm(setup([link,d]),method,{'slow':20000,'medium':100000}[link],d)
                            for sched in (['normal'] if method=='ideal_full' else ['normal','source_aware']):
                                features=[];messages=[];labels=[];correct=[];received=[];ids=[]
                                for ep in range(s['episodes']):
                                    assert time.monotonic()-start<s['wall_seconds']
                                    env=HiddenSource();obs,_=env.reset(seed=s['reset_base']+ep);h=EndpointHarness(model,actual,cfg,ep,False)
                                    for t in range(env.episode_length):
                                        chosen=None
                                        if sched=='source_aware':
                                            _,probe=step(deepcopy(env),clone(h),obs.copy(),t);k=len(probe['senders']);chosen=sorted(([0]+[i for i in probe['senders'] if i!=0])[:k]) if k else None
                                        def choose(scores,kmax):
                                            mask=torch.zeros(5,dtype=torch.bool);mask[chosen]=True;assert len(chosen)<=kmax;return mask
                                        if chosen is None:action,row,logs=h.step(obs,env.available_actions(),t)
                                        else:
                                            with patch.object(h.dac,'select',side_effect=choose):action,row,logs=h.step(obs,env.available_actions(),t)
                                        # Read immediately: the helper step() also evaluates local_q later.
                                        utility=captured['x'][0];cache=decay_messages(h.cache,h.source,t,model.decay_lambda) if model.decay_lambda!=1 else h.cache
                                        mask=(h.source>=0)[:,None,:]&~torch.eye(5,dtype=torch.bool)[None];count=mask.sum(-1,keepdim=True).clamp_min(1)
                                        aggregate=torch.einsum('brs,bsm->brm',mask.float(),cache)/count
                                        np.testing.assert_allclose(utility[:,64:72],aggregate[0].numpy(),atol=1e-6,rtol=1e-5)
                                        bit=int(env.bits[0]);got=any(x['sender']==0 and x['accepted'] for x in logs)
                                        correct.append((action[1:]==bit).astype(float));features.append(utility[1:]);messages.append(model.message(h.hidden)[0,0].numpy().copy());labels.append(bit);received.append(got);ids.append(ep)
                                        obs,reward,_,_,_=env.step(action)
                                        assert (bit,reward,row['K_delivered'],got)==expected[seed,method,cond,sched,ep,t];checked+=1
                                x=np.asarray(features,dtype=np.float64);source=np.asarray(messages,dtype=np.float64);y=np.asarray(labels,float);recv=np.asarray(received,bool);episode=np.asarray(ids);acc=np.asarray(correct);train=episode<s['episodes']//2;test=~train
                                cell=dict(seed=seed,method=method,condition=cond,scheduler=sched,train_episodes=s['episodes']//2,test_episodes=s['episodes']-s['episodes']//2,actual_accuracy=float(acc[test].mean()),all_episode_accuracy=float(acc.mean()),q=float(recv[test].mean()),test_steps=int(test.sum()),received_test_steps=int(recv[test].sum()),probes={})
                                for name in ['source','aggregate','local_cache_metadata','full_input']:
                                    predictions=[]
                                    for receiver in range(4):
                                        z=source if name=='source' else x[:,receiver,64:72] if name=='aggregate' else np.c_[x[:,receiver,:64],x[:,receiver,72:]] if name=='local_cache_metadata' else x[:,receiver]
                                        predictions.append(ridge_prediction(z[train],y[train],z[test],s['ridge']))
                                    pred=np.stack(predictions,axis=1);hits=pred==y[test,None]
                                    cell['probes'][name]=dict(accuracy=float(hits.mean()),received_accuracy=float(hits[recv[test]].mean()) if recv[test].any() else None,missing_accuracy=float(hits[~recv[test]].mean()) if (~recv[test]).any() else None,receiver_accuracy=hits.mean(0).tolist())
                                np.savez_compressed(out/f'seed{seed}_{method}_{cond}_{sched}.npz',utility_input=x,source_message=source,bit=y,source_received=recv,episode=episode,policy_correct=acc)
                                records.append(cell);atomic_json(out/'summary.json',dict(records=records,verified_steps=checked,scope=s['scope']));atomic_json(out/'status.json',dict(stage='running',cells=len(records)))
                    finally:handle.remove()
        atomic_json(out/'status.json',dict(stage='completed',cells=len(records),verified_steps=checked));print(out)
    except Exception as exc:atomic_json(out/'status.json',dict(stage='failed',cells=len(records),error=repr(exc)));raise
if __name__=='__main__':main()
