"""Exact frozen-policy sensitivity to noisy readiness estimates and decision cost."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch
import yaml
from scipy.special import ndtr
from ..external_deadline.mnist_deadline_attribution import gate_status, mean_ci
from ..external_biases.mnist_sum_learning import Policy, read_data, image_batch
from ..timing import configure_cpu


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def admission_probability(ready, deadline, sd, bias, cost):
    slack = deadline - 220. - cost - np.asarray(ready) - bias
    return (slack >= 0).astype(float) if sd == 0 else ndtr(slack / sd)


def evaluate_cell(ready, full, zero, deadline, sd, bias, cost):
    p = admission_probability(ready, deadline, sd, bias, cost)
    timely = np.asarray(ready) + cost + 220 <= deadline
    cheap_timely = np.asarray(ready) + 220 <= deadline
    q = p.mean()
    delta = full - zero
    return dict(offer_probability=p, actual_arrival_us=np.asarray(ready)+cost+220,
                timely=timely, cheap_timely=cheap_timely,
                informed=zero+p*timely*delta,
                random_same_cost=zero+q*timely*delta,
                random_no_extra_cost=zero+q*cheap_timely*delta,
                full_same_cost=zero+timely*delta,
                full_no_extra_cost=zero+cheap_timely*delta)


def main(config):
    configure_cpu()
    c=yaml.safe_load(Path(config).read_text()); out=Path(c['output']);out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    def write(name, data): (out/name).write_text(json.dumps(data,indent=2)+'\n')
    write('registration.json',dict(**c,entry_sha256=digest(__file__),config_sha256=digest(config)))
    source=Path(c['source_run']); source_c=json.loads((source/'registration.json').read_text())
    hashes=json.loads((source/'input_hashes.json').read_text())
    inputs={str(source/f):digest(source/f) for f in ['registration.json','input_hashes.json','results.json']}
    rows=[];checks=0
    try:
        allowed,gate=gate_status(Path(c['checkpoint_run']),c['seeds']);assert allowed,gate
        for path,value in hashes['checkpoints'].items(): assert digest(path)==value;inputs[path]=value
        data,data_hash=read_data(Path(c['data']));assert data_hash==hashes['data']
        images,_=data['test']
        write('input_hashes.json',dict(files=inputs,data=data_hash,gate=gate))
        cells=list(itertools.product(c['deadlines_us'],c['noise_sd_us'],c['bias_us'],c['decision_cost_us']))
        for seed in c['seeds']:
            ap=source/f'seed{seed}_episodes.npz';inputs[str(ap)]=digest(ap)
            with np.load(ap,allow_pickle=False) as saved:
                arrays={k:saved[k] for k in saved.files}
            checkpoint=Path(gate['checkpoints'][str(seed)])
            saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
            assert saved['updates']==gate['selected_updates']
            speaker=Policy().eval();listener=Policy(True).eval()
            speaker.load_state_dict(saved['speaker']);listener.load_state_dict(saved['listener'])
            full_parts=[];zero_parts=[]
            with torch.inference_mode():
                for b in range(0,len(arrays['target']),64):
                    assert time.monotonic()-start<c['wall_seconds']
                    si=arrays['source_indices'][b:b+64];li=arrays['listener_indices'][b:b+64]
                    ps=speaker(image_batch(images,si)).softmax(-1).numpy().astype(float)
                    z=listener.cnn(image_batch(images,li));m=torch.eye(20)[None].expand(len(si),-1,-1)
                    lp=listener.head(torch.cat([z[:,None].expand(-1,20,-1),m],-1)).softmax(-1).numpy().astype(float)
                    zero=listener.head(torch.cat([z,torch.zeros(len(si),20)],-1)).softmax(-1).numpy().astype(float)
                    np.testing.assert_allclose(ps,arrays['speaker_probabilities'][b:b+64],rtol=0,atol=1e-7)
                    np.testing.assert_allclose(lp,arrays['listener_by_message'][b:b+64],rtol=0,atol=1e-7)
                    np.testing.assert_allclose(zero,arrays['zero_probabilities'][b:b+64],rtol=0,atol=1e-7)
                    # Explicit message sum independently verifies historical einsum.
                    full_parts.append(sum(ps[:,m,None]*lp[:,m,:] for m in range(20)));zero_parts.append(zero)
            full_dist=np.concatenate(full_parts);zero_dist=np.concatenate(zero_parts)
            y=arrays['target'];j=np.arange(len(y));full=full_dist[j,y];zero=zero_dist[j,y];ready=arrays['ready_us']
            np.testing.assert_allclose(full,arrays['full_expected'],rtol=0,atol=1e-7)
            records={key:[] for key in evaluate_cell(ready,full,zero,*cells[0])}
            for deadline,sd,bias,cost in cells:
                assert time.monotonic()-start<c['wall_seconds']
                result=evaluate_cell(ready,full,zero,deadline,sd,bias,cost)
                for key,value in result.items():records[key].append(value)
                p=result['offer_probability'];timely=result['timely'];q=p.mean()
                # Independently form full action distributions to verify scalar rewards.
                expected=np.where(timely[:,None],p[:,None]*full_dist+(1-p[:,None])*zero_dist,zero_dist)
                np.testing.assert_allclose(expected[j,y],result['informed'],rtol=0,atol=1e-12)
                if sd==bias==0:
                    np.testing.assert_array_equal(p,timely.astype(float))
                    np.testing.assert_allclose(result['informed'],result['full_same_cost'],atol=1e-12)
                r=dict(seed=seed,deadline_us=deadline,noise_sd_us=sd,bias_us=bias,decision_cost_us=cost,
                       offered_B=17*q,random_offered_B=17*q,delivered_payload_B=float((p*timely).mean()),
                       random_same_payload_B=float(q*timely.mean()),random_cheap_payload_B=float(q*result['cheap_timely'].mean()),
                       late_offered_B=float(17*(p*~timely).mean()),false_reject_rate=float(((1-p)*timely).mean()),
                       informed=float(result['informed'].mean()),random_same_cost=float(result['random_same_cost'].mean()),
                       random_no_extra_cost=float(result['random_no_extra_cost'].mean()),
                       full_same_cost=float(result['full_same_cost'].mean()),full_no_extra_cost=float(result['full_no_extra_cost'].mean()),
                       gain_same_cost_pp=float(100*(result['informed']-result['random_same_cost']).mean()),
                       gain_practical_pp=float(100*(result['informed']-result['random_no_extra_cost']).mean()),
                       informed_minus_full_pp=float(100*(result['informed']-result['full_no_extra_cost']).mean()))
                rows.append(r);checks+=len(y)
            np.savez_compressed(out/f'seed{seed}_cells.npz',cells=np.asarray(cells),ready_us=ready,full_reward=full,zero_reward=zero,
                                **{k:np.stack(v) for k,v in records.items()})
        summary=[]
        for cell in cells:
            group=[r for r in rows if tuple(r[k] for k in ['deadline_us','noise_sd_us','bias_us','decision_cost_us'])==cell]
            metrics={k:mean_ci(r[k] for r in group) for k in group[0] if k not in ['seed','deadline_us','noise_sd_us','bias_us','decision_cost_us']}
            summary.append(dict(zip(['deadline_us','noise_sd_us','bias_us','decision_cost_us'],cell),metrics=metrics))
        primary=next(r for r in summary if all(r[k]==v for k,v in c['primary'].items()))
        for path,value in inputs.items():assert digest(path)==value
        write('input_hashes.json',dict(files=inputs,data=data_hash,gate=gate))
        write('results.json',dict(rows=rows,summary=summary,primary=primary,
              practical_primary_pass=primary['metrics']['gain_practical_pp']['t95'][0]>0,
              elapsed_seconds=time.monotonic()-start))
        write('validation.json',dict(passed=True,checkpoint_reinference=True,paired_example_cell_checks=checks,
              original_inputs_unchanged=True,known_timing_identity=True,equal_expected_wire_budget=True))
        write('status.json',dict(stage='completed',cells=len(rows),wall_seconds=time.monotonic()-start))
        print(json.dumps(primary,indent=2))
    except Exception as exc:
        write('status.json',dict(stage='failed',error=repr(exc),wall_seconds=time.monotonic()-start));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);main(p.parse_args().config)
