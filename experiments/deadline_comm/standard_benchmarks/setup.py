"""Isolated pinned SMACv2 setup; never alter shared Python or upstream trees."""
import concurrent.futures, hashlib, json, subprocess, time, zipfile
from pathlib import Path
import yaml

def main():
    cfg=yaml.safe_load(Path('experiments/deadline_comm/configs/smacv2_setup.yaml').read_text())
    root=Path(cfg['runtime']).resolve();root.mkdir(exist_ok=True,parents=True)
    started=time.monotonic();status=root/'setup_status.json'
    def update(stage,**kw):status.write_text(json.dumps(dict(stage=stage,elapsed_seconds=time.monotonic()-started,**kw),indent=2))
    def download(key,name):
        target=root/name
        if not target.exists():
            subprocess.run(['curl','-L','--fail','--retry','2','--connect-timeout','30','--max-time',str(cfg['setup_wall_seconds']),'-o',str(target)+'.partial',cfg[key]],check=True,stdout=subprocess.DEVNULL,stderr=(root/(name+'.download.log')).open('w'))
            Path(str(target)+'.partial').rename(target)
        return target
    try:
        update('downloading')
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            jobs=[pool.submit(download,*x) for x in [('sc2_url','SC2.4.10.zip'),('source_url','smacv2.zip'),('maps_url','SMAC_Maps.zip')]]
            venv=root/'venv'
            if not (venv/'bin/python').exists():subprocess.run([cfg['python'],'-m','venv','--system-site-packages',str(venv)],check=True)
            source=jobs[1].result()
            with zipfile.ZipFile(source) as z:z.extractall(root/'upstream')
            src=root/'upstream'/('smacv2-'+cfg['upstream_commit'])
            with (root/'pip.log').open('w') as log:
                subprocess.run([str(venv/'bin/python'),'-m','pip','install','numpy==1.26.4','protobuf==3.20.3','PyYAML','pytest',str(src)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
            for job in jobs:job.result()
        update('extracting')
        # Native unzip avoids Python's very slow pure-Python ZipCrypto on 4 GB.
        with (root/'unzip.log').open('w') as log:
            subprocess.run(['unzip','-q','-o','-P','iagreetotheeula',str(root/'SC2.4.10.zip'),'-d',str(root)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
        # Python zip extraction does not preserve POSIX executable bits.
        for p in (root/'StarCraftII/Versions').glob('Base*/SC2_x64'):p.chmod(0o755)
        maps=root/'StarCraftII/Maps/SMAC_Maps';maps.mkdir(exist_ok=True,parents=True)
        with zipfile.ZipFile(root/'SMAC_Maps.zip') as z:
            for item in z.infolist():
                if item.filename.endswith('.SC2Map'):(maps/Path(item.filename).name).write_bytes(z.read(item))
        hashes={}
        for p in root.glob('*.zip'):
            h=hashlib.sha256()
            with p.open('rb') as f:
                for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
            hashes[p.name]=h.hexdigest()
        (root/'provenance.json').write_text(json.dumps(dict(config=cfg,sha256=hashes,sc2path=str(root/'StarCraftII')),indent=2))
        update('installed');print(root,flush=True)
    except Exception as exc:update('failed',error=repr(exc));raise
if __name__=='__main__':main()
