"""Render KBS figures at final print width with a common 10 TeX-point text size.

Uses frozen figure_inputs.json; runs no training or statistical estimation.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
OUT=HERE/'figures'
TEXT=10*72/72.27
MAIN_WIDTH=(210-2*18.1)/25.4
SUPP_WIDTH=(210-2*24)/25.4
def main():
 inputs=json.loads((HERE/'figure_inputs.json').read_text())
 d=inputs['diagnostics']
 old={'raw_seed_results':inputs['seeds']}
 plt.rcParams.update({'font.size':TEXT,'axes.labelsize':TEXT,'axes.titlesize':TEXT,'xtick.labelsize':TEXT,'ytick.labelsize':TEXT,'legend.fontsize':TEXT,'pdf.fonttype':42})
 colors={'rr_cap':'tab:blue','dac':'tab:orange','full':'tab:green','periodic':'tab:purple'}
 names={'rr_cap':'RR','dac':'DAC','full':'Full','periodic':'Periodic'}
 fig,axs=plt.subplots(1,3,figsize=(SUPP_WIDTH,2.8),sharey=True)
 for ax,cond,title in zip(axs,['slow_5ms','medium_2ms','medium_5ms'],['Slow / 5 ms','Medium / 2 ms','Medium / 5 ms']):
  for i,arm in enumerate(['full','periodic','rr_cap','dac']):
   v=[r['mean'] for r in old['raw_seed_results'] if r['condition']==cond and r['arm']==arm];assert len(v)==10
   ax.scatter(i+np.linspace(-.12,.12,10),v,color=colors[arm],s=19,alpha=.8);ax.hlines(np.mean(v),i-.27,i+.27,color='black')
  ax.set_xticks(range(4),['Full','Periodic','RR','DAC'],rotation=35,ha='right');ax.set_title(title);ax.set_ylim(-.5,20.5);ax.grid(axis='y',alpha=.2)
 axs[0].set_ylabel('Episode return');fig.tight_layout();fig.savefig(OUT/'seed_distribution.pdf',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
 fig,axs=plt.subplots(1,3,figsize=(MAIN_WIDTH,3.5))
 for i,arm in enumerate(['rr_cap','dac']):
  for j,c in enumerate(['slow_5ms','medium_2ms']):
   v=[r['range'] for r in d['fresh_seed'] if r['condition']==c and r['checkpoint_arm']==arm];assert len(v)==10
   axs[0].scatter(np.linspace(i*2+j-.12,i*2+j+.12,10),v,color=colors[arm],marker=['o','^'][j],s=22)
 axs[0].set_xticks(range(4),['RR\nSlow/5 ms','RR\nMedium/2 ms','DAC\nSlow/5 ms','DAC\nMedium/2 ms'],fontsize=TEXT,rotation=55,ha='right');axs[0].set_ylabel('Mean suffix-return\nrange')
 for arm in ['rr_cap','dac']:
  v=[r for r in d['prefix_seed'] if r['condition']=='slow_5ms' and r['checkpoint_arm']==arm]
  axs[1].scatter([r['cold_delta'] for r in v],[r['warm_unconditional_delta'] for r in v],label=names[arm],color=colors[arm],marker='o' if arm=='rr_cap' else '^')
 axs[1].axhline(0,color='grey',lw=.5);axs[1].axvline(0,color='grey',lw=.5);axs[1].set_xlabel('Cold DAC minus RR\nreturn');axs[1].set_ylabel('Prefix DAC minus RR\nreturn');axs[1].legend(loc="upper center",bbox_to_anchor=(.5,-.63),ncol=2,frameon=False,handlelength=1,columnspacing=.8)
 for arm in ['rr_cap','dac']:
  for mode,ls in [('measured','-'),('fixed','--')]:
   xs=sorted({r['deadline_ms'] for r in d['readiness_seed']});ys=[np.mean([r['mean_k'] for r in d['readiness_seed'] if r['checkpoint_arm']==arm and r['mode']==mode and r['deadline_ms']==x]) for x in xs]
   axs[2].plot(xs,ys,ls,label=names[arm]+' '+mode,color=colors[arm])
 axs[2].set_xlabel('Deadline (ms)');axs[2].set_ylabel('Delivered packets\n/ step');axs[2].legend(fontsize=TEXT,loc="upper center",bbox_to_anchor=(.5,-.49),borderaxespad=0,handlelength=1.3,handletextpad=.4,frameon=False)
 axs[0].set_title('Feasible sender\nsubsets');axs[1].set_title('Slow / 5 ms');axs[2].set_title('Medium\n(trained at 2 ms)')
 fig.subplots_adjust(left=.10,right=.98,bottom=.44,top=.84,wspace=.85);fig.savefig(OUT/'diagnostics.pdf',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)


 fig, ax = plt.subplots(figsize=((MAIN_WIDTH-18/72.27)/2, 2.5), layout="constrained")
 for level, (access, color) in enumerate([("normal", "#215e8c"), ("source_aware", "#966422")]):
     result = inputs["receiver"][access]["adapted_minus_original"]
     values = np.array(result["seed_values"])
     ax.scatter(values, level + np.linspace(-.12, .12, len(values)),
                s=22, facecolors="none", edgecolors=color, linewidths=1.2)
     lo, hi = result["ci95"]
     ax.plot([lo, hi], [level + .22] * 2, color=color, linewidth=2.5)
     ax.scatter(result["mean"], level + .22, marker="D", color=color, s=32, zorder=3)
 ax.axvline(0, color=".35", linestyle=":", linewidth=1)
 ax.set_yticks([0, 1], ["Normal RR", "Source-aware"])
 ax.set_xlabel("Accuracy change after\nhead adaptation\n(percentage points)")
 ax.set_ylim(1.48, -.4)
 ax.spines[["top", "right", "left"]].set_visible(False)
 ax.tick_params(axis="y", length=0)
 ax.grid(axis="x", alpha=.18)
 fig.savefig(HERE / "figures" / "receiver_adaptation.pdf",
             metadata={"CreationDate": None, "ModDate": None})
 plt.close(fig)

if __name__=='__main__':main()
