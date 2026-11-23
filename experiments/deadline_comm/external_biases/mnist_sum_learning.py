"""Paper-based Eccles MNIST-sum learning, with disjoint validation/test control."""
from __future__ import annotations
import argparse, gzip, hashlib, json, math, os, time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
import yaml

class Policy(nn.Module):
    def __init__(self, listener=False):
        super().__init__(); self.listener=listener
        self.cnn=nn.Sequential(nn.Conv2d(1,32,5),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,5),nn.ReLU(),nn.MaxPool2d(2),nn.Flatten())
        self.head=nn.Sequential(nn.Linear(1024+(20 if listener else 0),1024),nn.ReLU(),nn.Linear(1024,19 if listener else 20))
    def forward(self,x,message=None):
        z=self.cnn(x)
        if self.listener: z=torch.cat([z,message],dim=-1)
        return self.head(z)

def biases(speaker_logits,listener_logits,zero_logits,c):
    ps=speaker_logits.softmax(-1); p=listener_logits.softmax(-1); p0=zero_logits.softmax(-1)
    hs=Categorical(logits=speaker_logits).entropy()
    marginal=-(ps.mean(0)*ps.mean(0).clamp_min(1e-12).log()).sum()
    return ((hs-c['entropy_target']).square().mean()-c['ps_lambda']*marginal,
            -(p-p0.detach()).abs().sum(-1).mean(),
            -(p.detach()*zero_logits.log_softmax(-1)).sum(-1).mean())

def read_data(root):
    ans={}; hashes={}
    for split,prefix in [('train','train'),('test','t10k')]:
        arr=[]
        for kind,offset in [('images-idx3-ubyte',16),('labels-idx1-ubyte',8)]:
            p=root/f'{prefix}-{kind}.gz'; blob=p.read_bytes(); hashes[p.name]=hashlib.sha256(blob).hexdigest()
            vals=np.frombuffer(gzip.decompress(blob)[offset:],dtype=np.uint8).copy()
            arr.append(torch.from_numpy(vals.reshape(-1,1,28,28) if offset==16 else vals.astype(np.int64)))
        ans[split]=arr
    assert len(ans['train'][0])==60000 and len(ans['test'][0])==10000
    return ans,hashes

def image_batch(images,idx): return images[idx].float()/255.

@torch.no_grad()
def evaluate(speaker,listener,images,labels,indices,n,seed):
    gen=torch.Generator().manual_seed(seed); idx=indices[torch.randint(len(indices),(2,n),generator=gen)]
    raw=[]; signal_mass=torch.zeros(10,20); listener_payoff=torch.zeros(10,20)
    for start in range(0,n,128):
        i,j=idx[:,start:start+128]; ys=labels[i]; yl=labels[j]; target=ys+yl
        sp=speaker(image_batch(images,i)).softmax(-1); m=torch.multinomial(sp,1,generator=gen).squeeze(-1)
        onehot=torch.nn.functional.one_hot(m,20).float(); z=listener.cnn(image_batch(images,j))
        lp=listener.head(torch.cat([z,onehot],-1)).softmax(-1)
        # Shuffling uses independent speaker image from the same held-out split.
        other=indices[torch.randint(len(indices),(len(i),),generator=gen)]
        op=speaker(image_batch(images,other)).softmax(-1)
        om=torch.multinomial(op,1,generator=gen).squeeze(-1)
        shuffled=listener.head(torch.cat([z,torch.nn.functional.one_hot(om,20).float()],-1)).softmax(-1)
        zero=listener.head(torch.cat([z,torch.zeros(len(i),20)],-1)).softmax(-1)
        all_messages=torch.eye(20)[None].expand(len(i),-1,-1)
        all_probs=listener.head(torch.cat([z[:,None].expand(-1,20,-1),all_messages],-1)).softmax(-1)
        # Exact sender-policy marginal: one-step causal influence, not speaker information.
        marginal=(sp[:,:,None]*all_probs).sum(1)
        for digit in range(10):
            signal_mass[digit]+=sp[ys==digit].sum(0)
            listener_payoff[digit]+=all_probs[torch.arange(len(i)),:,digit+yl].sum(0)
        cic=(sp*(all_probs*(all_probs.clamp_min(1e-12).log()-marginal[:,None].clamp_min(1e-12).log())).sum(-1)).sum(-1)
        r=torch.arange(len(i)); act=torch.multinomial(lp,1,generator=gen).squeeze(-1)
        raw.append(dict(sampled=(act==target).float().tolist(),expected=lp[r,target].tolist(),shuffled=shuffled[r,target].tolist(),zero=zero[r,target].tolist(),exact=(sp*all_probs[r,:,target]).sum(-1).tolist(),cic=cic.tolist()))
    metrics={k:float(np.mean([v for block in raw for v in block[k]])) for k in raw[0]}
    q=signal_mass/n; payoff=listener_payoff/n; prior=q.sum(-1)
    metrics['original_listener_power']=float(q.max(0).values.sum()-prior.max())
    metrics['original_speaker_power']=float((prior*(payoff.max(-1).values-payoff.mean(-1))).sum())
    metrics['uniform_speaker_expected']=float((prior*payoff.mean(-1)).sum())
    metrics['optimal_label_listener']=float(q.max(0).values.sum())
    metrics['pairs']=n; metrics['seed']=seed
    return metrics

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    c=yaml.safe_load(Path(a.config).read_text());torch.set_num_threads(c['threads']);torch.set_num_interop_threads(1)
    out=Path(c['output']+('_smoke' if a.smoke else ''));out.mkdir(parents=True,exist_ok=False)
    if a.smoke:c.update(seeds=[0],milestones=[5],evaluation_pairs=64,validation_pairs=64)
    (out/'registration.json').write_text(json.dumps(c,indent=2));start=time.monotonic()
    dataset,hashes=read_data(Path(c['data'])); (out/'data_hashes.json').write_text(json.dumps(hashes,indent=2))
    train,lab=dataset['train'];test,testlab=dataset['test']; summaries=[];previous=0;selected=0
    try:
      for milestone in c['milestones']:
        summaries=[]
        for seed in c['seeds']:
          for arm in c['arms']:
            torch.manual_seed(seed); gen=torch.Generator().manual_seed(100000+seed)
            speaker=Policy();listener=Policy(True)
            ospeaker=torch.optim.Adam(speaker.parameters(),lr=c['learning_rate']);olistener=torch.optim.Adam(listener.parameters(),lr=c['learning_rate'])
            checkpoint=out/f'seed{seed}_{arm}.pt'
            if previous:
              saved=torch.load(checkpoint,weights_only=True);speaker.load_state_dict(saved['speaker']);listener.load_state_dict(saved['listener']);ospeaker.load_state_dict(saved['ospeaker']);olistener.load_state_dict(saved['olistener']);torch.set_rng_state(saved['rng']);gen.set_state(saved['pair_rng'])
            with (out/f'seed{seed}_{arm}_training.jsonl').open('a') as log:
              for update in range(previous,milestone):
                if time.monotonic()-start>c['wall_seconds']:raise TimeoutError('Registered CPU wall budget exhausted')
                i,j=torch.randint(50000,(2,c['batch_size']),generator=gen)
                sl=speaker(image_batch(train,i));sd=Categorical(logits=sl);m=sd.sample()
                ll=listener(image_batch(train,j),torch.nn.functional.one_hot(m,20).float());ld=Categorical(logits=ll);act=ld.sample()
                reward=(act==lab[i]+lab[j]).float()
                ls=-(sd.log_prob(m)*reward).mean()-c['entropy_message_'+arm]*sd.entropy().mean()
                lloss=-(ld.log_prob(act)*reward).mean()-c['entropy_action']*ld.entropy().mean()
                if arm=='both':
                  zero=listener(image_batch(train,j),torch.zeros(len(j),20));ps,pl,ce=biases(sl,ll,zero,c)
                  ls=ls+c['ps_weight']*ps;lloss=lloss+c['pl_weight']*pl+c['ce_weight']*ce
                ospeaker.zero_grad();olistener.zero_grad();ls.backward();lloss.backward();ospeaker.step();olistener.step()
                if (update+1)%100==0 or update+1==milestone:
                  log.write(json.dumps(dict(update=update+1,batch_reward=float(reward.mean()),speaker_loss=float(ls.detach()),listener_loss=float(lloss.detach()),elapsed=time.monotonic()-start))+'\n');log.flush()
            saved=dict(speaker=speaker.state_dict(),listener=listener.state_dict(),ospeaker=ospeaker.state_dict(),olistener=olistener.state_dict(),rng=torch.get_rng_state(),pair_rng=gen.get_state(),updates=milestone)
            torch.save(saved,checkpoint)
            val=evaluate(speaker,listener,train,lab,torch.arange(50000,60000),c['validation_pairs'],c['evaluation_seed'])
            row=dict(seed=seed,arm=arm,updates=milestone,validation=val);summaries.append(row)
            print(json.dumps(row),flush=True)
        (out/f'validation_{milestone}.json').write_text(json.dumps(summaries,indent=2));selected=milestone;previous=milestone
        passes=all(r['validation']['sampled']>.2 and r['validation']['expected']>r['validation']['shuffled'] for r in summaries if r['arm']=='both')
        if a.smoke or (milestone>=3000 and passes):break
      final=[]
      for seed in c['seeds']:
        for arm in c['arms']:
          saved=torch.load(out/f'seed{seed}_{arm}.pt',weights_only=True);speaker=Policy();listener=Policy(True);speaker.load_state_dict(saved['speaker']);listener.load_state_dict(saved['listener'])
          met=evaluate(speaker,listener,test,testlab,torch.arange(10000),c['evaluation_pairs'],c['evaluation_seed']+1)
          final.append(dict(seed=seed,arm=arm,updates=selected,test=met))
      result=dict(status='completed',elapsed=time.monotonic()-start,selected_updates=selected,validation_gate_pass=passes,records=final,scope='Paper-based reconstruction; not author code; no deadline or cache manipulation; labels reward/evaluation only')
      (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    except Exception as exc:
      (out/'status.json').write_text(json.dumps(dict(stage='failed',error=repr(exc),elapsed=time.monotonic()-start),indent=2));raise
    (out/'status.json').write_text(json.dumps(dict(stage='completed',validation_gate_pass=passes)))
if __name__=='__main__':main()
