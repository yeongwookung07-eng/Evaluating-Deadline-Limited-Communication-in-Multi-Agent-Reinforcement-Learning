"""Render editorial figures from printed, previously validated manuscript values.

No evaluation or training. Table rows are parsed directly to avoid retyping data.
Identity summarizes the observed delivery comparison and 48-byte packet accounting.
"""
from pathlib import Path
import re, json, hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
P=Path(__file__).resolve().parent
s=(P/'jaamas_main.tex').read_text()
plt.rcParams.update({'font.size':9,'axes.labelsize':10,'axes.titlesize':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
blue='#23658a'; orange='#c36930'
records={}
def save(fig,name,data):
 fig.savefig(P/(name+'.pdf'),bbox_inches='tight',pad_inches=.05)
 fig.savefig(P/(name+'.png'),dpi=150,bbox_inches='tight',pad_inches=.05)
 plt.close(fig);records[name]=data
def rows(label):
 block=s[s.index('\\label{'+label+'}'):];block=block[:block.index('\\end{table}')]
 return [l for l in block.splitlines() if '&' in l and re.match(r'(\d|Slow/|Medium/|Normal RR)',l)]
fig,ax=plt.subplots(figsize=(4.65,2.25));ax.barh(['Full','Round-robin'],[240,48],color=[blue,orange]);ax.invert_yaxis();ax.set(xlim=(0,290),xlabel='Offered wire bytes per decision')
for y,v in enumerate([240,48]):ax.text(v+4,y,str(v),va='center')
ax.set_title('80% fewer offered bytes; 400/400 identical pairs',pad=12)
save(fig,'delivery_identity',{'source':'Section sec:identity; 5 or 1 packets, 48 B each','offered':[240,48],'identical_pairs':400})
a=[]
for row in rows('tab:external_deadlines'):
 cells=row.split('&');a.append([float(c.replace(',','').strip()) for c in cells[:4]])
a=np.array(a)
fig,ax=plt.subplots(figsize=(4.65,2.8));ax.plot(a[:,0],a[:,2],'-o',color=blue,label='Full / timing-informed');ax.plot(a[:,0],a[:,3],'--s',color=orange,label='Random');ax.set(xlabel='Reception deadline (µs)',ylabel='Expected accuracy (%)',ylim=(0,60),xticks=a[:,0]);ax.legend(frameon=False,fontsize=8)
save(fig,'equal_cost_admission',{'source':'tab:external_deadlines','rows':a.tolist(),'columns':['deadline_us','informed_random_bytes','full_informed_accuracy','random_accuracy']})
a=[]; labels=[]
for row in rows('tab:sender_weighting'):
 cells=row.split('&');labels.append(cells[0].strip()+' / '+cells[1].strip());a.append([[float(v) for v in re.findall(r'-?\d+\.\d+',c)] for c in [cells[3],cells[5]]])
a=np.array(a);fig,axes=plt.subplots(1,2,figsize=(4.65,3),gridspec_kw={'width_ratios':[1,1.2]})
for k,ax in enumerate(axes):
 v=a[:,k,:];ax.errorbar(v[:,0],range(4),xerr=np.array([v[:,0]-v[:,1],v[:,2]-v[:,0]]),fmt='o',color=[blue,orange][k],capsize=3);ax.set_ylim(3.5,-.5)
axes[0].set_yticks(range(4),labels,fontsize=7);axes[1].set_yticks([]);axes[0].set_xlabel('Sensitive states (%)');axes[1].set_xlabel('Episode-weighted\nreturn difference');axes[1].axvline(0,color='gray',lw=.8,ls='--');axes[0].set_title('Sender sensitivity');axes[1].set_title('Factual − uniform');fig.tight_layout(w_pad=1)
save(fig,'sender_selection',{'source':'tab:sender_weighting','labels':labels,'sensitivity_and_episode_contrast_mean_ci':a.tolist()})
vals=[]
for row in rows('tab:receiver_controls'):
 vals.append(float(row.split('&')[2].strip()))
# Printed normal-RR rows: unchanged, true labels, shuffled labels, probe.
assert len(vals)==4
fig,ax=plt.subplots(figsize=(4.65,2.6));bars=ax.barh(['Original actor','Shuffled-label actor','Adapted actor','Probe (decoding)'],[vals[0],vals[2],vals[1],vals[3]],color=[blue,blue,blue,'white'],edgecolor=blue);bars[-1].set_hatch('///');ax.invert_yaxis();ax.set(xlim=(0,85),xlabel='Accuracy (%)');ax.axhline(2.5,color='gray',lw=.7,ls='--')
for bar in bars:ax.text(bar.get_width()+1,bar.get_y()+bar.get_height()/2,f'{bar.get_width():.2f}',va='center')
save(fig,'receiver_utilization',{'source':'tab:receiver_controls, normal RR only','acting':[vals[0],vals[2],vals[1]],'probe_decoding':vals[3],'note':'Probe is not an acting policy; original fixed-readiness cohort, not hard fallback.'})
assert '5.63-point gain [1.80, 9.46]' in s and '5.45-point gain [1.99, 8.92]' in s
fig,ax=plt.subplots(figsize=(4.65,2.7));v=np.array([5.63,5.45,0]);lo=np.array([1.80,1.99,0]);hi=np.array([9.46,8.92,0]);ax.errorbar([0,250,500],v,yerr=[v-lo,hi-v],fmt='o-',capsize=4,color=blue);ax.axhline(0,color='gray',lw=.8);ax.set(xlabel='Added postprocessing (µs)',ylabel='Adapted − original (pp)',xticks=[0,250,500],ylim=(-1,11));ax.set_xlim(-25,580);ax.annotate('All candidates\nlate', xy=(500,0), xytext=(0,24), textcoords='offset points', ha='center', va='bottom')
save(fig,'execution_boundary',{'source':'sec:deployment_robustness; previous-action fallback','added_us':[0,250,500],'gain':v.tolist(),'ci_low':lo.tolist(),'ci_high':hi.tolist(),'note':'500 µs: every head has 47.75% accuracy; difference identically zero.'})
(P/'findings_figure_inputs.json').write_text(json.dumps(records,indent=2)+'\n')
