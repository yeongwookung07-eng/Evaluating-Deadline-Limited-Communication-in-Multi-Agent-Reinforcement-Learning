"""Verify bundled assets and restore the paths used by the experiment code."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def inside(relative, allowed):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts or relative.parts[0] not in allowed:
        raise ValueError(f'Invalid asset path: {relative}')
    path = ROOT / relative
    if not path.resolve().is_relative_to(ROOT):
        raise ValueError(f'Asset path leaves this repository: {relative}')
    return path


def prepare(check_only=False):
    manifest = json.loads((ROOT / 'results/manifest.json').read_text())
    copied = 0
    for row in manifest['files']:
        source = inside(row['path'], {'results', 'checkpoints'})
        if source.stat().st_size != row['bytes'] or digest(source) != row['sha256']:
            raise ValueError(f'Bundled asset mismatch: {row["path"]}')
        if check_only:
            continue
        target = inside(row['original'], {'experiments', 'evidence'})
        if target.exists():
            if target.stat().st_size != row['bytes'] or digest(target) != row['sha256']:
                raise FileExistsError(f'Local input differs; preserve or move it before preparing: {target}')
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied += 1
    print(f'PASS: {len(manifest["files"])} asset hashes; {copied} local input copies prepared.')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    prepare(parser.parse_args().check_only)
