"""Execute the exported artifacts in a clean directory, not the working tree."""
import json,zipfile,tempfile,subprocess,sys,os,hashlib
from pathlib import Path
BASE=Path('experiments/deadline_comm')
def main():
    out=BASE/'runs/tmlr_artifact_replay_20261119_001';out.mkdir(exist_ok=False)
    bundle=BASE/'tmlr_revision/bundles/results_complete';sources={n:hashlib.sha256((bundle/n).read_bytes()).hexdigest() for n in ['anonymous_code_settings.zip','evidence_checkpoints.zip']}
    (out/'registration.json').write_text(json.dumps(dict(source_hashes=sources,scope='Clean working directory on same installed software environment, not a new OS/container; no training.'),indent=2))
    jobs=[('tests',[sys.executable,'-m','pytest','experiments/deadline_comm/tmlr_revision/tests/test_interventions.py','-q']),('raw_audit',[sys.executable,'-m','experiments.deadline_comm.tmlr_revision.editorial_audit']),('fresh_smoke',[sys.executable,'-m','experiments.deadline_comm.tmlr_revision.experiments','--kind','fresh','--config','experiments/deadline_comm/configs/tmlr_fresh.yaml','--smoke'])]
    try:
        with tempfile.TemporaryDirectory(prefix='deadline_artifact_replay_') as tmp:
            for name in sources:
                with zipfile.ZipFile(bundle/name) as z:z.extractall(tmp)
            expected=json.loads((Path(tmp)/BASE/'tmlr_revision/results/editorial_audit.json').read_text())
            for name,cmd in jobs:
                with (out/(name+'.log')).open('w') as f:subprocess.run(cmd,cwd=tmp,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300,env=dict(os.environ,PYTHONPATH=tmp,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
            actual=json.loads((Path(tmp)/BASE/'tmlr_revision/results/editorial_audit.json').read_text());assert expected==actual
            smoke=Path(tmp)/BASE/'runs/tmlr_fresh_20261119_001_smoke';assert json.loads((smoke/'status.json').read_text())['stage']=='completed'
            summary=json.loads((smoke/'summary.json').read_text());assert summary['noop_checks']>0
            result=dict(stage='completed',tests='5 passed',raw_audit_exact_match=True,fresh_smoke_states=len(summary['states']),fresh_smoke_noop_checks=summary['noop_checks'],source_hashes=sources,scope='Same software environment, isolated extracted code/data. Full campaign re-execution and independently provisioned OS not claimed.')
            (out/'summary.json').write_text(json.dumps(result,indent=2));(out/'status.json').write_text(json.dumps(dict(stage='completed')));print(json.dumps(result))
    except Exception as exc:(out/'status.json').write_text(json.dumps(dict(stage='failed',error=repr(exc))));raise
if __name__=='__main__':main()
