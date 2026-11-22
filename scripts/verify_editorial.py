"""Recompute frozen editorial evidence in a temporary directory, without training."""
import contextlib
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.deadline_comm.tmlr_revision import editorial_audit


def compare(actual, expected, path='result'):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            compare(actual[key], expected[key], f'{path}.{key}')
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, f'{path}[{i}]')
    elif isinstance(expected, float):
        assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), path
    else:
        assert actual == expected, path


def main():
    evidence = ROOT / 'evidence/editorial'
    base = Path('experiments/deadline_comm')
    with tempfile.TemporaryDirectory(prefix='deadline-editorial-') as directory:
        temporary = Path(directory)
        for row in json.loads((evidence / 'manifest.json').read_text()):
            packed = (evidence / row['archive']).read_bytes()
            assert hashlib.sha256(packed).hexdigest() == row['compressed_sha256'], row['archive']
            data = gzip.decompress(packed)
            assert len(data) == row['bytes'], row['source']
            assert hashlib.sha256(data).hexdigest() == row['sha256'], row['source']
            relative = Path(row['source'])
            assert not relative.is_absolute() and '..' not in relative.parts
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        hidden = base / 'runs/hidden_mechanism_20261119_001'
        shutil.copytree(ROOT / hidden, temporary / hidden)
        editorial_audit.BASE = temporary / base
        editorial_audit.HERE = temporary / base / 'tmlr_revision'
        (editorial_audit.HERE / 'results').mkdir(parents=True)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            editorial_audit.main()
        actual = json.loads((editorial_audit.HERE / 'results/editorial_audit.json').read_text())
        expected = json.loads((ROOT / base / 'tmlr_revision/results/editorial_audit.json').read_text())
        compare(actual, expected)
        print(output.getvalue().strip())
        print('PASS: all editorial audit fields match the frozen result (float tolerance 1e-10 relative / 1e-12 absolute).')
        print('No training, checkpoint evaluation, or modification of recorded results.')


if __name__ == '__main__':
    main()
