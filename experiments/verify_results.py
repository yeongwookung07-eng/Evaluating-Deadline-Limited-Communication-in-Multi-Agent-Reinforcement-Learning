"""Verify recorded analysis results and model inputs without training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def run(command):
    subprocess.run([sys.executable,*command],cwd=ROOT,check=True)


def receiver_audit():
    p=ROOT/'experiments/deadline_comm/runs/receiver_head_adaptation_20261123_001'
    data=[json.loads(line) for line in (p/'episodes.jsonl').open()]
    indexed={(r['seed'],r['scheduler'],r['arm'],r['episode']):r for r in data}
    for r in data:
        reference=indexed[r['seed'],r['scheduler'],'original',r['episode']]
        assert len(r['trace'])==20
        rewards=[]
        for actual, base in zip(r['trace'],reference['trace']):
            assert actual['bit']==base['bit']
            assert actual['offered_wire_bytes']==96 and actual['delivered_payload_bytes']==64
            assert actual['delivered_senders']==base['delivered_senders']
            assert actual['source_timestamps']==base['source_timestamps']
            reward=sum(a==actual['bit'] for a in actual['actions'][1:])/4
            assert reward==actual['reward']
            rewards.append(reward)
        assert sum(rewards)==r['return_']
        assert abs(sum(rewards)/len(rewards)-r['accuracy'])<1e-12
    assert len(data)==1200
    print('PASS: receiver actual actions/rewards, return and matched delivery; 1,200 episodes.')


def structural_recompute():
    import yaml
    from verify_statistics import compare
    config=ROOT/'experiments/deadline_comm/configs/attribution_structural_history_v2.yaml'
    settings=yaml.safe_load(config.read_text())
    expected=json.loads((ROOT/settings['output']/'results.json').read_text())
    with tempfile.TemporaryDirectory(prefix='reviewer-structural-') as tmp:
        settings['output']=str(Path(tmp)/'result')
        copied=Path(tmp)/'config.yaml'
        copied.write_text(yaml.safe_dump(settings))
        run(['-m','experiments.deadline_comm.attribution_boundaries.structural_history_audit','--config',str(copied)])
        actual=json.loads((Path(settings['output'])/'results.json').read_text())
        for value in [actual,expected]:
            value.pop('elapsed_seconds')
        compare(actual,expected)
    print('PASS: structural/history/timing summary recomputes from exact raw inputs.')


def external_models():
    import torch
    import yaml
    from experiments.deadline_comm.external_biases.mnist_sum_learning import Policy
    config=yaml.safe_load((ROOT/'experiments/deadline_comm/configs/external_deadline_mnist_v2.yaml').read_text())
    directory=ROOT/config['checkpoint_run']
    result=json.loads((directory/'summary.json').read_text())
    files=[ROOT/r['checkpoint'] for r in result['records']]
    assert len(files)==6
    for path in files:
        saved=torch.load(path,map_location='cpu',weights_only=True)
        Policy().load_state_dict(saved['speaker'],strict=True)
        Policy(True).load_state_dict(saved['listener'],strict=True)
        assert saved['updates']==result['selected_updates']
    print('PASS: six external final speaker/listener checkpoints match selected milestone.')
    common=list(directory.glob('seed*_1500.pt'))
    assert len(common)==6
    for path in common:
        saved=torch.load(path,map_location='cpu',weights_only=True)
        assert saved['updates']==1500
        Policy().load_state_dict(saved['speaker'],strict=True)
        Policy(True).load_state_dict(saved['listener'],strict=True)
    print('PASS: six common1500 checkpoints for optional no-PID continuation.')


def deadline_saved_inputs():
    import numpy as np
    import torch
    import yaml
    from experiments.deadline_comm.external_deadline.mnist_deadline_attribution import expected_listener,reception_mask,random_admission,after_delivery,influence
    config=yaml.safe_load((ROOT/'experiments/deadline_comm/configs/external_deadline_mnist_v2.yaml').read_text())
    directory=ROOT/config['output']
    status=json.loads((directory/'status.json').read_text())
    if status['stage']=='blocked_by_learning_gate':
        training=json.loads((ROOT/config['checkpoint_run']/'summary.json').read_text())
        assert training['status']!='completed' or training['validation_gate_pass'] is not True
        assert status['scientific_conclusions'] is False
        print('PASS: external diagnostic correctly blocked by failed/incomplete learning gate; no benchmark success asserted.')
        return
    result=json.loads((directory/'results.json').read_text())
    learning=yaml.safe_load((ROOT/'experiments/deadline_comm/configs/external_biases_mnist.yaml').read_text())
    assert config['pair_seed'] not in [learning['evaluation_seed'],learning['evaluation_seed']+1]
    generator=torch.Generator().manual_seed(learning['evaluation_seed']+1)
    earlier_pairs=torch.randint(10000,(2,learning['evaluation_pairs']),generator=generator).numpy()
    earlier_set=set(zip(earlier_pairs[0].tolist(),earlier_pairs[1].tolist()))
    cells={(r['seed'],r['deadline_us']):r for r in result['deadline_records']}
    for seed in config['seeds']:
        with np.load(directory/f'seed{seed}_episodes.npz',allow_pickle=False) as data:
            new_pairs=set(zip(data['source_indices'].tolist(),data['listener_indices'].tolist()))
            print(f'External seed {seed}: independent deployment pair RNG; realized old/new test-pair overlap={len(new_pairs & earlier_set)}.')
            assert np.all(data['source_indices']<10000) and np.all(data['listener_indices']<10000)
            ps=data['speaker_probabilities'];lp=data['listener_by_message'];target=data['target'];j=np.arange(len(target));zero=data['zero_probabilities']
            full=expected_listener(ps,lp)
            np.testing.assert_allclose(full[j,target],data['full_expected'],rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(influence(ps,lp),data['influence_nats'],rtol=1e-12,atol=1e-12)
            for deadline in config['deadlines_us']:
                row=cells[seed,deadline]
                mask=reception_mask(data['ready_us'],deadline,config['wire_bytes'],config['bandwidth_bytes_s'],config['propagation_us'])
                offers=random_admission(mask,config['random_admission_seed']+deadline)
                actual=after_delivery(mask,full,zero)
                random_policy=after_delivery(offers & mask,full,zero)
                np.testing.assert_array_equal(mask,data[f'delivered_{deadline}us'])
                np.testing.assert_array_equal(offers,data[f'random_offers_{deadline}us'])
                np.testing.assert_allclose(actual[j,target],data[f'expected_{deadline}us'],rtol=1e-12,atol=1e-12)
                np.testing.assert_allclose(random_policy[j,target],data[f'random_expected_{deadline}us'],rtol=1e-12,atol=1e-12)
                assert abs(float(actual[j,target].mean())-row['full_expected'])<1e-12
                assert abs(float((actual[j,target]-random_policy[j,target]).mean())-row['informed_minus_random'])<1e-12
                assert int(mask.sum())==int(offers.sum())
    print('PASS: external exact marginal, influence, reception and rate-matched random controls from saved arrays.')


def regenerate_external(data_root):
    import numpy as np
    import yaml
    from verify_statistics import compare
    settings=yaml.safe_load((ROOT/'experiments/deadline_comm/configs/external_deadline_mnist_v2.yaml').read_text())
    recorded=ROOT/settings['output']
    status=json.loads((recorded/'status.json').read_text())
    if status['stage']=='blocked_by_learning_gate':
        print('SKIP: external inference regeneration blocked by original learning gate; no new training.')
        return
    data_root=Path(data_root).resolve()
    run(['experiments/fetch_mnist.py','--destination',str(data_root),'--check-only'])
    with tempfile.TemporaryDirectory(prefix='reviewer-external-inference-') as tmp:
        settings['data']=str(data_root)
        settings['output']=str(Path(tmp)/'results')
        config=Path(tmp)/'config.yaml';config.write_text(yaml.safe_dump(settings))
        run(['-m','experiments.deadline_comm.external_deadline.mnist_deadline_attribution','--config',str(config)])
        regenerated=Path(settings['output'])
        expected=json.loads((recorded/'results.json').read_text())
        actual=json.loads((regenerated/'results.json').read_text())
        for result in [expected,actual]:result.pop('elapsed_seconds')
        compare(actual,expected)
        for seed in settings['seeds']:
            with np.load(recorded/f'seed{seed}_episodes.npz',allow_pickle=False) as before,np.load(regenerated/f'seed{seed}_episodes.npz',allow_pickle=False) as after:
                assert set(before.files)==set(after.files)
                for key in before.files:
                    np.testing.assert_allclose(after[key],before[key],rtol=1e-10,atol=1e-12,err_msg=key)
    print('PASS: external predictions and full deadline results regenerate from final models and verified original MNIST inputs.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--recompute',action='store_true')
    parser.add_argument('--tests',action='store_true')
    parser.add_argument('--external-data',help='Optional local four-gzip MNIST directory for full external inference regeneration; no download/training.')
    args=parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=''
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
        os.environ[key]='1'
    from prepare_inputs import prepare
    prepare()
    if args.recompute:
        run(['experiments/inspect_checkpoints.py'])
        run(['experiments/verify_statistics.py'])
        run(['experiments/verify_models.py','--probes','--smoke'])
        structural_recompute()
        receiver_audit()
        external_models()
        deadline_saved_inputs()
        run(['-m','experiments.deadline_comm.deployment_robustness.verify_results','--mnist','experiments/deadline_comm/runs/mnist_uncertain_readiness_20261126_001','--broadcast','experiments/deadline_comm/runs/broadcast_hard_deadline_20261126_001'])
        run(['experiments/reproduce_external_continuation.py','--output',str(Path(tempfile.gettempdir())/'communication-external-preview-unused')])
    if args.tests:
        run(['-m','pytest','-q','-p','no:cacheprovider','tests'])
    if args.external_data:
        regenerate_external(args.external_data)


if __name__=='__main__':
    main()
