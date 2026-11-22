"""Incremental durable records and atomic status snapshots."""
import csv
import json
import os
from pathlib import Path


def atomic_json(path,data):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(data,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

class CSVLog:
    def __init__(self,path):
        self.path=Path(path);self.file=self.path.open('x',newline='');self.writer=None
    def append(self,rows):
        for row in rows:
            if self.writer is None:
                self.writer=csv.DictWriter(self.file,fieldnames=list(row));self.writer.writeheader()
            self.writer.writerow(row)
        self.file.flush();os.fsync(self.file.fileno())
    def close(self):self.file.close()

class JSONLLog:
    def __init__(self,path):self.file=Path(path).open('x')
    def append(self,rows):
        for row in rows:self.file.write(json.dumps(row,allow_nan=False)+'\n')
        self.file.flush();os.fsync(self.file.fileno())
    def close(self):self.file.close()
