"""Restore verbatim experiment YAML from the Markdown source catalogue."""
import argparse
import hashlib
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def configurations():
    for document in sorted((ROOT / 'docs/configs').glob('*.md')):
        if document.name == 'README.md':
            continue
        text = document.read_text()
        match = re.search(r'```yaml\n(.*?)```', text, re.S)
        digest = re.search(r'SHA256: `([0-9a-f]{64})`', text)
        if not match or not digest:
            raise ValueError(f'Missing configuration block or hash: {document.name}')
        data = match[1].encode()
        if hashlib.sha256(data).hexdigest() != digest[1]:
            raise ValueError(f'Configuration hash mismatch: {document.name}')
        if not isinstance(yaml.safe_load(data), dict):
            raise ValueError(f'Expected a YAML mapping: {document.name}')
        yield document.stem + '.yaml', data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Verify without creating YAML files.')
    parser.add_argument('--output-root', type=Path, default=ROOT)
    args = parser.parse_args()
    count = 0
    for name, data in configurations():
        count += 1
        if args.check:
            continue
        target = args.output_root / 'experiments/deadline_comm/configs' / name
        if target.exists() and target.read_bytes() != data:
            raise FileExistsError(f'Refusing to overwrite changed settings: {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    print(f'{"Verified" if args.check else "Materialized"} {count} configurations; no experiment launched.')


if __name__ == '__main__':
    main()
