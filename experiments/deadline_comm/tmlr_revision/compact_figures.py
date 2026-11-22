"""Render the compact paper's two figures from frozen seed aggregates."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path('experiments/deadline_comm');OUT=B/'publication/tmlr/figures'
def main():
 d=json.loads((B/'tmlr_revision/results/summary.json').read_text())
 old=json.loads((B/'runs/review_revision_20261119_001/summary.json').read_text())
 plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
 colors={'rr_cap':'tab:blue','dac':'tab:orange','full':'tab:green','periodic':'tab:purple'}
 names={'rr_cap':'RR','dac':'DAC','full':'Full','periodic':'Periodic'}
 fig,axs=plt.subplots(1,3,figsize=(11.5,2.7),sharey=True)
 for ax,cond,title in zip(axs,['slow_5ms','medium_2ms','medium_5ms'],['Slow / 5 ms','Medium / 2 ms','Medium / 5 ms']):
  for i,arm in enumerate(['full','periodic','rr_cap','dac']):
   v=[r['mean'] for r in old['raw_seed_results'] if r['condition']==cond and r['arm']==arm];assert len(v)==10
   ax.scatter(i+np.linspace(-.12,.12,10),v,color=colors[arm],s=19,alpha=.8);ax.hlines(np.mean(v),i-.27,i+.27,color='black')
  ax.set_xticks(range(4),['Full','Periodic','RR','DAC']);ax.set_title(title);ax.set_ylim(-.5,20.5);ax.grid(axis='y',alpha=.2)
 axs[0].set_ylabel('Episode return');fig.tight_layout();fig.savefig(OUT/'seed_distribution.pdf');plt.close(fig)
 fig,axs=plt.subplots(1,3,figsize=(12,3.4))
 for i,arm in enumerate(['rr_cap','dac']):
  for j,c in enumerate(['slow_5ms','medium_2ms']):
   v=[r['range'] for r in d['fresh_seed'] if r['condition']==c and r['checkpoint_arm']==arm];assert len(v)==10
   axs[0].scatter(np.linspace(i*2+j-.12,i*2+j+.12,10),v,color=colors[arm],marker=['o','^'][j],s=22)
 axs[0].set_xticks(range(4),['RR\nSlow/5 ms','RR\nMedium/2 ms','DAC\nSlow/5 ms','DAC\nMedium/2 ms'],fontsize=8);axs[0].set_ylabel('Mean suffix-return range')
 for arm in ['rr_cap','dac']:
  v=[r for r in d['prefix_seed'] if r['condition']=='slow_5ms' and r['checkpoint_arm']==arm]
  axs[1].scatter([r['cold_delta'] for r in v],[r['warm_unconditional_delta'] for r in v],label=names[arm],color=colors[arm],marker='o' if arm=='rr_cap' else '^')
 axs[1].axhline(0,color='grey',lw=.5);axs[1].axvline(0,color='grey',lw=.5);axs[1].set_xlabel('Cold DAC minus RR return');axs[1].set_ylabel('Prefix DAC minus RR return');axs[1].legend()
 for arm in ['rr_cap','dac']:
  for mode,ls in [('measured','-'),('fixed','--')]:
   xs=sorted({r['deadline_ms'] for r in d['readiness_seed']});ys=[np.mean([r['mean_k'] for r in d['readiness_seed'] if r['checkpoint_arm']==arm and r['mode']==mode and r['deadline_ms']==x]) for x in xs]
   axs[2].plot(xs,ys,ls,label=names[arm]+' '+mode,color=colors[arm])
 axs[2].set_xlabel('Deadline (ms)');axs[2].set_ylabel('Delivered packets / step');axs[2].legend(fontsize=8)
 axs[0].set_title('Feasible sender subsets');axs[1].set_title('Slow / 5 ms');axs[2].set_title('Medium (trained at 2 ms)')
 fig.tight_layout();fig.savefig(OUT/'diagnostics.pdf');plt.close(fig)
if __name__=='__main__':main()
