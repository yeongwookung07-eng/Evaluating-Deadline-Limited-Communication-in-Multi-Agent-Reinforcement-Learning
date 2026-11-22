"""Extract missing settings and primary online-interaction learning curves."""
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path('experiments/deadline_comm');HERE=BASE/'tmlr_revision';out=HERE/'repro_assets';out.mkdir(exist_ok=True)
rows=[]
for p in sorted((BASE/'runs/day3_fair_budget_20261118_002').glob('seed*/*/calibration.json')):
 r=json.loads(p.read_text());rows.append(dict(seed=int(p.parent.parent.name[4:]),method=r['method'],budget=r['budget'],threshold_or_period=r['selected']['parameter'],calibration_offered=r['selected']['offered_B_step'],eligible=r['eligible']))
with (out/'historical_calibration.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
tex=[r'\paragraph{Historical calibration, separately from the primary cohort.} Calibration uses the fast/5ms link on ten resets starting at $5100000+10000s$, with evaluation on separate resets starting at $5200000+10000s$. Periodic evaluates periods1--6. VBC/TMC evaluate anchors $0,.01,.03,.1,.3,1,3,10,30,100$ followed by ten neighbor-midpoint refinement iterations around the smallest byte-error candidate. All trials are retained; monotonicity is not assumed. The chosen setting minimizes absolute difference from the target. VBC-batched reuses the VBC threshold. DAC uses threshold0 and budget credit; it does not threshold-tune its priority score. Full selected settings are listed below and in the supplementary CSV.',r'\begin{longtable}{llrrr}\caption{Historical calibration parameters, not primary ten-seed settings.}\\\toprule Method & Target B & Seed0 & Seed1 & Seed2\\\midrule\endfirsthead\toprule Method & Target B & Seed0 & Seed1 & Seed2\\\midrule\endhead']
for method in ['periodic','vbc','tmc','dac']:
 for budget in [48,96,120,160,240]:
  v=[next(r['threshold_or_period'] for r in rows if (r['method'],r['budget'],r['seed'])==(method,budget,s)) for s in range(3)];tex.append(method+' & '+str(budget)+' & '+' & '.join(f'{x:.8g}' for x in v)+r'\\')
tex.append(r'\bottomrule\end{longtable}');(out/'calibration.tex').write_text('\n'.join(tex))
fig,axes=plt.subplots(1,3,figsize=(11.8,3.6),sharey=True)
for ax,cond in zip(axes,['slow_5ms','medium_2ms','medium_5ms']):
 for arm in ['no_comm','ideal_full','full','periodic','rr_cap','dac']:
  curves=[]
  for seed in range(10):
   path=BASE/f'runs/reviewer_ten_seeds_20261118_001/scratch_{cond}_seed{seed}_{arm}/training.jsonl';rr=[json.loads(x) for x in path.read_text().splitlines()];x=np.array([r['online_transitions'] for r in rr]);y=np.convolve([r['episode_return'] for r in rr],np.ones(100)/100,mode='valid');curves.append((x[99:],y))
  grid=np.linspace(max(x[0] for x,y in curves),min(x[-1] for x,y in curves),150);a=np.array([np.interp(grid,x,y) for x,y in curves]);ax.plot(grid,a.mean(0),label=arm);ax.fill_between(grid,a.mean(0)-a.std(0,ddof=1),a.mean(0)+a.std(0,ddof=1),alpha=.08)
 ax.set_title(cond);ax.set_xlabel('Online environment transitions');ax.set_ylim(-5,20)
axes[0].set_ylabel('Rolling exploratory training return (100 episodes)');axes[-1].legend(fontsize=7);fig.tight_layout();fig.savefig(out/'learning_curves.pdf');fig.savefig(out/'learning_curves.png',dpi=160)
print(out)
