"""Load bundled PyTorch checkpoints on CPU and report their tensor contents."""
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch

ROOT = Path(__file__).resolve().parents[1]


def tensor_count(value):
    if isinstance(value, torch.Tensor):
        return 1
    if isinstance(value, dict):
        return sum(tensor_count(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_count(item) for item in value)
    return 0


def main():
    torch.set_num_threads(1)
    rows = json.loads((ROOT / 'results/manifest.json').read_text())['files']
    checkpoints = [row for row in rows if row['path'].startswith('checkpoints/') and Path(row['path']).suffix == '.pt']
    tensors = 0
    for row in checkpoints:
        value = torch.load(ROOT / row['path'], map_location='cpu', weights_only=True)
        count = tensor_count(value)
        if not isinstance(value, dict) or not count:
            raise ValueError(f'Checkpoint has no model tensors: {row["path"]}')
        tensors += count
        del value
    print(f'PASS: {len(checkpoints)} PyTorch checkpoints load on CPU ({tensors} tensors).')


if __name__ == '__main__':
    main()
