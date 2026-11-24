"""Verify recorded checkpoints and receiver inputs; optionally recompute probes and replay."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def verify_inputs():
    import torch
    from experiments.deadline_comm.day2.finetune_comm_aware import load_model
    from experiments.deadline_comm.day2.policy import DecayedPolicy
    rows = json.loads((ROOT / 'evidence/reviewer/manifest.json').read_text())
    counts = {}
    for row in rows:
        data = (ROOT / row['path']).read_bytes()
        if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError(f"Artifact mismatch: {row['path']}")
        if 'raw_sha256' in row:
            raw = gzip.decompress(data)
            if len(raw) != row['raw_bytes'] or hashlib.sha256(raw).hexdigest() != row['raw_sha256']:
                raise ValueError(f"Raw mismatch: {row['path']}")
        if row['kind'] == 'checkpoint':
            path = ROOT / row['path']
            if path.name == 'checkpoint_1500.pt':
                model, _ = load_model(path)
            else:
                model = DecayedPolicy(4, 10, 5, 2, hidden_dim=64, message_dim=8, mixer='qmix')
                model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)['model'])
            model.eval()
        counts[row['kind']] = counts.get(row['kind'], 0) + 1
    print('PASS: hashes, decompression and model state loading:', counts)


def verify_probes():
    import numpy as np
    from verify_statistics import compare
    from experiments.deadline_comm.tmlr_revision.robustness.aggregate_probe import ridge_prediction
    base = ROOT / 'experiments/deadline_comm/runs/tmlr_receiver_probe_20261119_001'
    records = json.loads((base / 'summary.json').read_text())['records']
    for row in records:
        name = 'seed{seed}_{method}_{condition}_{scheduler}.npz'.format(**row)
        with np.load(base / name, allow_pickle=False) as data:
            x, source, y = data['utility_input'], data['source_message'], data['bit']
            episode, recv, acc = data['episode'], data['source_received'], data['policy_correct']
            train = episode < row['train_episodes']; test = ~train
            if set(episode[train]) & set(episode[test]):
                raise ValueError('Episode leakage')
            compare(float(acc[test].mean()), row['actual_accuracy'], name + '.policy')
            compare(float(acc.mean()), row['all_episode_accuracy'], name + '.all_policy')
            compare(float(recv[test].mean()), row['q'], name + '.q')
            compare(int(test.sum()), row['test_steps'], name + '.steps')
            compare(int(recv[test].sum()), row['received_test_steps'], name + '.received_steps')
            for kind, expected in row['probes'].items():
                predictions = []
                for receiver in range(4):
                    z = source if kind == 'source' else x[:, receiver, 64:72] if kind == 'aggregate' else np.c_[x[:, receiver, :64], x[:, receiver, 72:]] if kind == 'local_cache_metadata' else x[:, receiver]
                    predictions.append(ridge_prediction(z[train], y[train], z[test], .001))
                hits = np.stack(predictions, axis=1) == y[test, None]
                actual = dict(accuracy=float(hits.mean()), received_accuracy=float(hits[recv[test]].mean()) if recv[test].any() else None, missing_accuracy=float(hits[~recv[test]].mean()) if (~recv[test]).any() else None, receiver_accuracy=hits.mean(0).tolist())
                compare(actual, expected, name + '.' + kind)
    print(f'PASS: all four probe types and policy accounting match in {len(records)} cells')


def replay_smoke():
    import yaml
    config = ROOT / 'experiments/deadline_comm/configs/tmlr_fresh.yaml'
    with tempfile.TemporaryDirectory(prefix='reviewer-replay-') as directory:
        settings = yaml.safe_load(config.read_text())
        settings['output'] = str(Path(directory) / 'fresh')
        path = Path(directory) / 'config.yaml'; path.write_text(yaml.safe_dump(settings))
        subprocess.run([sys.executable, '-m', 'experiments.deadline_comm.tmlr_revision.experiments', '--kind', 'fresh', '--config', str(path), '--smoke'], cwd=ROOT, check=True)
        result = json.loads((Path(settings['output'] + '_smoke') / 'status.json').read_text())
        if result['stage'] != 'completed':
            raise ValueError(result)
    print('PASS: seed-0 RR/DAC fresh-subset replay, Slow/5 ms and Medium/2 ms; temporary outputs removed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probes', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
        os.environ[key] = '1'
    verify_inputs()
    if args.probes:
        verify_probes()
    if args.smoke:
        replay_smoke()


if __name__ == '__main__':
    main()
