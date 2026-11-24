"""Check this source edition; optionally run its CPU tests in an isolated copy."""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tests', action='store_true')
    args = parser.parse_args()
    files = [p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.relative_to(ROOT).parts]
    source = [p for p in files if p.suffix in {'.py', '.md'}]
    python = [p for p in source if p.suffix == '.py']
    markdown = [p for p in source if p.suffix == '.md']
    for path in python:
        compile(path.read_bytes(), str(path), 'exec')
    for path in markdown:
        for target in re.findall(r'(?<!!)\[[^\]]+\]\(([^)]+)\)', path.read_text()):
            if target.startswith(('https://', 'http://', '#', 'mailto:')):
                continue
            target = target.split('#', 1)[0]
            if not (path.parent / target).exists():
                raise FileNotFoundError(f'{path.relative_to(ROOT)}: {target}')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    subprocess.run([sys.executable, str(ROOT / 'scripts/materialize_configs.py'), '--check'],
                   cwd=ROOT, env=env, check=True)
    print(f'PASS: {len(python)} Python sources compile; {len(markdown)} Markdown documents have valid local links.', flush=True)
    if not args.tests:
        return
    with tempfile.TemporaryDirectory(prefix='deadline-analysis-check-') as directory:
        isolated = Path(directory)
        for path in source:
            target = isolated / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        env['PYTHONPATH'] = str(isolated)
        subprocess.run([sys.executable, str(isolated / 'scripts/materialize_configs.py')],
                       cwd=isolated, env=env, check=True)
        subprocess.run([sys.executable, '-m', 'pytest', '-q', '-c', '/dev/null',
                        '-p', 'no:cacheprovider', 'experiments/deadline_comm'],
                       cwd=isolated, env=env, check=True)


if __name__ == '__main__':
    main()
