"""Gated, exact one-step deadline attribution on held-out external MNIST-sum."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics as st
import time
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
import yaml
from scipy.stats import t as student_t
from ..external_biases.mnist_sum_learning import Policy, read_data, image_batch


def expected_listener(speaker, listener_by_message):
    """Exact marginal action distribution; inputs [episodes,m], [episodes,m,a]."""
    return np.einsum('nm,nma->na',speaker,listener_by_message)


def influence(speaker, listener_by_message):
    marginal=expected_listener(speaker,listener_by_message)
    kl=(listener_by_message*(np.log(np.maximum(listener_by_message,1e-30))-np.log(np.maximum(marginal[:,None],1e-30)))).sum(-1)
    return (speaker*kl).sum(-1)


def original_powers(speaker, listener_by_message, source_labels, listener_labels):
    """Published one-step label-power concepts, estimated on this empirical image set."""
    n=len(source_labels);joint=np.zeros((10,20));payoff=np.zeros((10,20))
    for digit in range(10):
        joint[digit]=speaker[np.asarray(source_labels)==digit].sum(0)/n
        payoff[digit]=listener_by_message[np.arange(n),:,digit+np.asarray(listener_labels)].mean(0)
    prior=joint.sum(-1)
    return dict(original_listener_power=float(joint.max(0).sum()-prior.max()),original_speaker_power=float((prior*(payoff.max(-1)-payoff.mean(-1))).sum()))


def reception_mask(readiness_us, deadline_us, wire_bytes, bandwidth, propagation_us):
    return np.asarray(readiness_us)+wire_bytes/bandwidth*1e6+propagation_us<=deadline_us


def random_admission(timely, seed):
    return np.asarray(timely)[np.random.default_rng(seed).permutation(len(timely))]


def after_delivery(delivered, full, zero):
    return np.where(np.asarray(delivered)[:,None],full,zero)


def mean_ci(values):
    values=list(map(float,values));m=st.mean(values)
    d=float(student_t.ppf(.975,len(values)-1)*st.stdev(values)/math.sqrt(len(values))) if len(values)>1 else None
    return dict(mean=m,t95=[m-d,m+d] if d is not None else None,n_seeds=len(values),seed_values=values)


def gate_status(root,seeds):
    p=root/'summary.json'
    if not p.exists():return False,'External summary is absent; no evaluation permitted.'
    summary=json.loads(p.read_text())
    if summary.get('status')!='completed' or summary.get('validation_gate_pass') is not True:return False,'External run did not complete with its registered learning gate passed.'
    validation=root/f"validation_{summary['selected_updates']}.json"
    if not validation.exists():return False,'Selected-milestone validation is absent.'
    selected=[r for r in json.loads(validation.read_text()) if r['arm']=='both']
    if sorted(r['seed'] for r in selected)!=sorted(seeds):return False,'Validation seed coverage differs from preregistered seeds.'
    if not all(r['validation']['sampled']>.2 and r['validation']['expected']>r['validation']['shuffled'] for r in selected):return False,'Independent check of the original per-seed validation criterion failed.'
    return True,dict(selected_updates=summary['selected_updates'],summary_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),validation_sha256=hashlib.sha256(validation.read_bytes()).hexdigest(),checkpoints={str(r['seed']):r['checkpoint'] for r in summary.get('records',[]) if r.get('arm')=='both' and 'checkpoint' in r})


def main(config):
    c=yaml.safe_load(Path(config).read_text());out=Path(c['output']);out.mkdir(parents=True,exist_ok=False)
    (out/'registration.json').write_text(json.dumps(c,indent=2)+'\n')
    allowed,gate=gate_status(Path(c['checkpoint_run']),c['seeds'])
    if not allowed:
        (out/'status.json').write_text(json.dumps(dict(stage='blocked_by_learning_gate',reason=gate,scientific_conclusions=False),indent=2)+'\n')
        print(gate);return
    (out/'gate.json').write_text(json.dumps(gate,indent=2)+'\n')
    torch.set_num_threads(c['threads']);torch.set_num_interop_threads(1);start=time.monotonic()
    data,hashes=read_data(Path(c['data']));images,labels=data['test'];n=c['pair_count'];rng=np.random.default_rng(c['pair_seed'])
    indices=rng.integers(0,len(images),(2,n));target=(labels[indices[0]]+labels[indices[1]]).numpy()
    ready=np.resize(np.array(c['readiness_us'],float),n);rng.shuffle(ready)
    # A nonzero cyclic permutation exactly preserves the empirical sender marginal.
    permutation=np.roll(np.arange(n),n//2+1);assert np.all(permutation!=np.arange(n))
    seed_rows=[];cell_rows=[];checkpoint_hashes={}
    try:
      for seed in c['seeds']:
        path=Path(gate['checkpoints'].get(str(seed),str(Path(c['checkpoint_run'])/f"seed{seed}_{c['arm']}.pt")));checkpoint_hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        saved=torch.load(path,weights_only=True);assert saved['updates']==gate['selected_updates']
        speaker=Policy();listener=Policy(True);speaker.load_state_dict(saved['speaker']);listener.load_state_dict(saved['listener']);speaker.eval();listener.eval()
        sparts=[];lparts=[];zparts=[]
        with torch.inference_mode():
          for b in range(0,n,64):
            if time.monotonic()-start>c['wall_seconds']:raise TimeoutError('Registered evaluation budget exhausted')
            si,li=indices[:,b:b+64];sp=speaker(image_batch(images,si)).softmax(-1)
            z=listener.cnn(image_batch(images,li));messages=torch.eye(20)[None].expand(len(si),-1,-1)
            allp=listener.head(torch.cat([z[:,None].expand(-1,20,-1),messages],-1)).softmax(-1)
            zero=listener.head(torch.cat([z,torch.zeros(len(si),20)],-1)).softmax(-1)
            sparts.append(sp.numpy());lparts.append(allp.numpy());zparts.append(zero.numpy())
        ps=np.concatenate(sparts).astype(float);lp=np.concatenate(lparts).astype(float);zero=np.concatenate(zparts).astype(float)
        full=expected_listener(ps,lp);shuffle=expected_listener(ps[permutation],lp);ci=influence(ps,lp);j=np.arange(n)
        assert np.allclose(full.sum(-1),1,atol=1e-6)
        base=dict(seed=seed,pairs=n,full_expected=float(full[j,target].mean()),shuffle_expected=float(shuffle[j,target].mean()),zero_expected=float(zero[j,target].mean()),shuffle_gain=float((full[j,target]-shuffle[j,target]).mean()),zero_gain=float((full[j,target]-zero[j,target]).mean()),causal_influence_nats=float(ci.mean()))
        base.update(original_powers(ps,lp,labels[indices[0]].numpy(),labels[indices[1]].numpy()))
        seed_rows.append(base)
        arrays=dict(source_indices=indices[0],listener_indices=indices[1],target=target,shuffle_permutation=permutation,ready_us=ready,speaker_probabilities=ps,listener_by_message=lp,zero_probabilities=zero,full_expected=full[j,target],shuffle_expected=shuffle[j,target],influence_nats=ci)
        for deadline in c['deadlines_us']:
          mask=reception_mask(ready,deadline,c['wire_bytes'],c['bandwidth_bytes_s'],c['propagation_us'])
          restored=reception_mask(np.full(n,c['restore_readiness_us']),deadline,c['wire_bytes'],c['bandwidth_bytes_s'],c['propagation_us'])
          assert restored.all(),'Registered restoration must deliver for every tested deadline'
          factual=after_delivery(mask,full,zero)
          admitted=after_delivery(mask,full,zero) # same delivered input by explicitly omitted late attempts
          assert np.array_equal(factual,admitted)
          random_offers=random_admission(mask,c['random_admission_seed']+deadline)
          random_received=random_offers & mask
          assert int(random_offers.sum())==int(mask.sum())
          random_policy=after_delivery(random_received,full,zero)
          sr=after_delivery(mask,shuffle,zero)
          values=dict(seed=seed,deadline_us=deadline,pairs=n,delivery_rate=float(mask.mean()),full_offered_bytes=float(c['wire_bytes']),admit_offered_bytes=float(mask.mean()*c['wire_bytes']),late_offered_bytes=float((~mask).mean()*c['wire_bytes']),delivered_payload_bytes=float(mask.mean()*c['symbol_payload_bytes']),full_expected=float(factual[j,target].mean()),admit_expected=float(admitted[j,target].mean()),restored_expected=base['full_expected'],admit_minus_full=float((admitted[j,target]-factual[j,target]).mean()),restored_minus_full=float((full[j,target]-factual[j,target]).mean()),shuffle_gap=float((factual[j,target]-sr[j,target]).mean()),causal_influence_nats=float((ci*mask).mean()),max_full_admit_distribution_error=float(np.abs(factual-admitted).max()))
          values.update(random_offered_bytes=float(random_offers.mean()*c['wire_bytes']),random_late_offered_bytes=float((random_offers & ~mask).mean()*c['wire_bytes']),random_delivery_rate=float(random_received.mean()),random_expected=float(random_policy[j,target].mean()),informed_minus_random=float((admitted[j,target]-random_policy[j,target]).mean()),informed_random_action_tv=float((.5*np.abs(admitted-random_policy).sum(-1)).mean()),random_causal_influence_nats=float((ci*random_received).mean()))
          assert values['random_offered_bytes']==values['admit_offered_bytes']
          cell_rows.append(values);arrays[f'delivered_{deadline}us']=mask;arrays[f'expected_{deadline}us']=factual[j,target];arrays[f'random_offers_{deadline}us']=random_offers;arrays[f'random_expected_{deadline}us']=random_policy[j,target]
        np.savez_compressed(out/f'seed{seed}_episodes.npz',**arrays)
        print(json.dumps(base),flush=True)
      baseline={k:mean_ci(r[k] for r in seed_rows) for k in seed_rows[0] if k not in ['seed','pairs']}
      comparisons=[]
      for deadline in c['deadlines_us']:
        rows=[r for r in cell_rows if r['deadline_us']==deadline]
        comparisons.append(dict(deadline_us=deadline,metrics={k:mean_ci(r[k] for r in rows) for k in rows[0] if k not in ['seed','deadline_us','pairs']}))
      result=dict(status='completed',gate=gate,seed_records=seed_rows,deadline_records=cell_rows,baseline_seed_intervals=baseline,deadline_seed_intervals=comparisons,elapsed_seconds=time.monotonic()-start,scope='Exact expected one-step frozen-policy diagnostic under declared modeled reception; no measured CPU/radio timing, cache, sender selection or original-score reproduction.')
      (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');(out/'input_hashes.json').write_text(json.dumps(dict(data=hashes,checkpoints=checkpoint_hashes),indent=2)+'\n')
      (out/'status.json').write_text(json.dumps(dict(stage='completed',gate_passed=True,identity_pairs=n*len(c['seeds'])*len(c['deadlines_us'])),indent=2)+'\n')
    except Exception as exc:
      (out/'status.json').write_text(json.dumps(dict(stage='failed',error=repr(exc),elapsed_seconds=time.monotonic()-start),indent=2)+'\n');raise

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--config',required=True);main(a.parse_args().config)
