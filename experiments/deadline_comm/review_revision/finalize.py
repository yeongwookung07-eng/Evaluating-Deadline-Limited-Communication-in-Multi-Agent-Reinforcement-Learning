"""Bounded completion aggregation; never turn an unfinished job into a result."""
import json,time,statistics,subprocess
from pathlib import Path
root=Path('experiments/deadline_comm/runs');out=root/'review_revision_20261119_001';deadline=time.monotonic()+8.5*3600
names=['review_identity_20261119_001','review_oracle_free_20261119_001','review_native_20261119_001']
while True:
 states={}
 for name in names:
  p=root/name/'status.json';states[name]=json.loads(p.read_text()) if p.exists() else {'stage':'initializing'}
 (out/'followup_status.json').write_text(json.dumps(states,indent=2))
 if any(s['stage']=='failed' for s in states.values()):raise RuntimeError('Follow-up failed; inspect followup_status.json')
 if all(s['stage']=='completed' for s in states.values()):break
 if time.monotonic()>deadline:raise TimeoutError('Follow-up completion deadline')
 time.sleep(30)
identity=json.loads((root/names[0]/'summary.json').read_text())['records']
learning=[json.loads(p.read_text()) for p in sorted((root/names[1]).glob('seed*/result.json'))];assert len(learning)==12
native=json.loads((root/names[2]/'summary.json').read_text())
report=['# Additional reviewer controls completed','', 'These are bounded diagnostics, not proof of generalization or native DAC superiority.','', '## Demonstration-free Hallway (3 seeds; fixed readiness)','|Arm|Mean ± sample SD|Mean online transitions|','|---|---:|---:|']
for arm in ['ideal_full','no_comm','rr_cap','dac']:
 x=[r for r in learning if r['arm']==arm];v=[r['mean_return'] for r in x];report.append(f"|{arm}|{statistics.mean(v):.3f} ± {statistics.stdev(v):.3f}|{statistics.mean(r['online_transitions'] for r in x):.0f}|")
report+=['','## Fixed-checkpoint Full/RR identity','|Condition|Decay|Same sets/common steps|Same cache|Same actions|','|---|---:|---:|---:|---:|']
for c in ['slow_5ms','medium_2ms','medium_5ms']:
 for d in [1.,.9]:
  x=[r for r in identity if r['condition']==c and r['decay']==d and r['method']=='rr_cap'];total=sum(r['common_steps'] for r in x)
  report.append(f"|{c}|{d}|{sum(r['same_sets'] for r in x)}/{total}|{sum(r['same_cache'] for r in x)}|{sum(r['same_actions'] for r in x)}|")
report+=['','## Native author-task result',json.dumps(native,indent=2),'','Native task score and Hallway return are different objectives; do not pool. Standard benchmark matched DAC and second-machine timing remain unresolved.']
(out/'FOLLOWUP_RESULTS.md').write_text('\n'.join(report)+'\n')
(out/'followup_results.json').write_text(json.dumps(dict(identity=identity,oracle_free=learning,native=native),indent=2))
with Path('experiments/deadline_comm/review_revision/README.md').open('a') as f:f.write('\nFollow-up diagnostics completed; see ../runs/review_revision_20261119_001/FOLLOWUP_RESULTS.md. No automatic adoption or journal-readiness claim.\n')
with Path('README.md').open('a') as f:f.write('\n2026-11-19 reviewer follow-up diagnostics — **completed**; actual additional results: experiments/deadline_comm/runs/review_revision_20261119_001/FOLLOWUP_RESULTS.md. Broader review objections are not automatically resolved.\n')
with Path('docs/research_experiment_journal.md').open('a') as f:f.write('\nReviewer follow-up raw diagnostics completed. Artifacts: experiments/deadline_comm/runs/review_revision_20261119_001/followup_results.json and FOLLOWUP_RESULTS.md. Adoption deferred to evidence interpretation; native and Hallway results not pooled.\n')
print(out/'FOLLOWUP_RESULTS.md',flush=True)
