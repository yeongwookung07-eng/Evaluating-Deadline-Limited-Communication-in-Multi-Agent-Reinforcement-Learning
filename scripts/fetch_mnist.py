"""Optional upstream download; MNIST files are deliberately not redistributed."""
import argparse
import hashlib
from pathlib import Path
import urllib.request

# Mirror and MD5s verified against PyTorch torchvision's official MNIST loader:
# https://raw.githubusercontent.com/pytorch/vision/main/torchvision/datasets/mnist.py
# Dataset origin: http://yann.lecun.com/exdb/mnist/
MIRROR = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
FILES = {
    'train-images-idx3-ubyte.gz': 'f68b3c2dcbeaaa9fbdd348bbdeb94873',
    'train-labels-idx1-ubyte.gz': 'd53e105ee54ea40749a09fcbcd1e9432',
    't10k-images-idx3-ubyte.gz': '9fb629c4189551a2d022fa330f9573f3',
    't10k-labels-idx1-ubyte.gz': 'ec29112dd5afa0611ce80d1b7f02629c',
}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--destination', default='experiments/deadline_comm/external_biases/data')
    parser.add_argument('--check-only', action='store_true')
    args=parser.parse_args()
    root=Path(args.destination)
    root.mkdir(parents=True,exist_ok=True)
    for name, expected in FILES.items():
        path=root/name
        if not path.exists():
            if args.check_only:
                raise FileNotFoundError(path)
            data=urllib.request.urlopen(MIRROR+name,timeout=60).read()
            assert hashlib.md5(data).hexdigest()==expected, name
            path.write_bytes(data)
        assert hashlib.md5(path.read_bytes()).hexdigest()==expected, name
    print('PASS: all four MNIST gzip files match upstream MD5s; no training executed.')


if __name__=='__main__':
    main()
