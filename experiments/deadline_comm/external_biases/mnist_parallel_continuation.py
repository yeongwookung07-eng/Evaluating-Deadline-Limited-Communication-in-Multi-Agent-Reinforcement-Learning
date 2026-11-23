"""Prospective parallel continuation of intact seed-specific optimizer/RNG states."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import torch,yaml
from torch.distributions import Categorical
from .mnist_sum_learning import Policy,biases,read_data,image_batch,evaluate

def worker(c,out,seed,arm,source,start_update,target):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    data,_=read_data(Path(c['data']));images,labels=data['train']
    saved=torch.load(source,weights_only=True);assert saved['updates']==start_update
    speaker=Policy();listener=Policy(True);speaker.load_state_dict(saved['speaker']);listener.load_state_dict(saved['listener'])
    ospeaker=torch.optim.Adam(speaker.parameters(),lr=c['learning_rate']);olistener=torch.optim.Adam(listener.parameters(),lr=c['learning_rate'])
    ospeaker.load_state_dict(saved['ospeaker']);olistener.load_state_dict(saved['olistener'])
    torch.set_rng_state(saved['rng']);gen=torch.Generator();gen.set_state(saved['pair_rng']);begin=time.monotonic()
    with (out/f'seed{seed}_{arm}_{target}_training.jsonl').open('x') as log:
      for update in range(start_update,target):
        i,j=torch.randint(50000,(2,c['batch_size']),generator=gen)
        sl=speaker(image_batch(images,i));sd=Categorical(logits=sl);m=sd.sample()
        ll=listener(image_batch(images,j),torch.nn.functional.one_hot(m,20).float());ld=Categorical(logits=ll);act=ld.sample()
        reward=(act==labels[i]+labels[j]).float()
        ls=-(sd.log_prob(m)*reward).mean()-c['entropy_message_'+arm]*sd.entropy().mean()
        lloss=-(ld.log_prob(act)*reward).mean()-c['entropy_action']*ld.entropy().mean()
        if arm=='both':
            zero=listener(image_batch(images,j),torch.zeros(len(j),20));ps,pl,ce=biases(sl,ll,zero,c)
            ls=ls+c['ps_weight']*ps;lloss=lloss+c['pl_weight']*pl+c['ce_weight']*ce
        ospeaker.zero_grad();olistener.zero_grad();ls.backward();lloss.backward();ospeaker.step();olistener.step()
        if (update+1)%100==0 or update+1==target:
            log.write(json.dumps(dict(update=update+1,batch_reward=float(reward.mean()),elapsed=time.monotonic()-begin))+'\n');log.flush()
    dest=out/f'seed{seed}_{arm}_{target}.pt'
    torch.save(dict(speaker=speaker.state_dict(),listener=listener.state_dict(),ospeaker=ospeaker.state_dict(),olistener=olistener.state_dict(),rng=torch.get_rng_state(),pair_rng=gen.get_state(),updates=target),dest)
    val=evaluate(speaker,listener,images,labels,torch.arange(50000,60000),c['validation_pairs'],c['evaluation_seed'])
    (out/f'seed{seed}_{arm}_{target}_validation.json').write_text(json.dumps(dict(seed=seed,arm=arm,updates=target,validation=val),indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--worker',nargs=5);p.add_argument('--handoff-pid',type=int);a=p.parse_args()
    extra=yaml.safe_load(Path(a.config).read_text());c=yaml.safe_load(Path(extra['base_config']).read_text());out=Path(extra['output'])
    if a.worker:
        seed,arm,source,initial,target=a.worker;worker(c,out,int(seed),arm,Path(source),int(initial),int(target));return
    out.mkdir(parents=True,exist_ok=False);(out/'registration.json').write_text(json.dumps(dict(base=c,amendment=extra),indent=2))
    source=Path(extra['source']);initial=extra['handoff_update'];waitstart=time.monotonic()
    expected=f'python -m experiments.deadline_comm.external_biases.mnist_sum_learning --config {extra["base_config"]}'
    while not (source/f'validation_{initial}.json').exists():
        if time.monotonic()-waitstart>extra['handoff_wait_seconds']:raise TimeoutError('Handoff common checkpoint wait')
        if (source/'status.json').exists():raise RuntimeError('Serial segment ended before handoff')
        time.sleep(3)
    if not a.handoff_pid:raise ValueError('Explicit owned PID required')
    actual=Path(f'/proc/{a.handoff_pid}/cmdline').read_bytes().replace(b'\0',b' ').decode().strip()
    if actual!=expected:raise ValueError(('Refusing unrelated process',actual,expected))
    os.kill(a.handoff_pid,signal.SIGSTOP)
    checkpoints={}
    try:
      for seed in c['seeds']:
        for arm in c['arms']:
          original=source/'milestones'/f'seed{seed}_{arm}_{initial}.pt'
          if not original.exists():original=source/f'seed{seed}_{arm}.pt'
          saved=torch.load(original,weights_only=True);assert saved['updates']==initial
          dest=out/f'seed{seed}_{arm}_{initial}.pt';torch.save(saved,dest);checkpoints[seed,arm]=dest
      (source/'status.json').write_text(json.dumps(dict(stage='handed_off',common_updates=initial,destination=str(out),reason='Prospective parallel execution amendment; objective/data/RNG retained; any uncheckpointed later serial work discarded'),indent=2))
      os.kill(a.handoff_pid,signal.SIGTERM);os.kill(a.handoff_pid,signal.SIGCONT)
    except BaseException:
      os.kill(a.handoff_pid,signal.SIGCONT);raise
    begin=time.monotonic();latest=initial;rows=json.loads((source/f'validation_{initial}.json').read_text());passed=False;status='completed'
    children=[];logs=[]
    try:
      for target in extra['milestones']:
        children=[];logs=[]
        for seed in c['seeds']:
          for arm in c['arms']:
            log=(out/f'seed{seed}_{arm}_{target}.log').open('x');logs.append(log)
            children.append(subprocess.Popen([sys.executable,'-m','experiments.deadline_comm.external_biases.mnist_parallel_continuation','--config',a.config,'--worker',str(seed),arm,str(checkpoints[seed,arm]),str(latest),str(target)],stdout=log,stderr=subprocess.STDOUT))
        for child in children:
            remain=extra['wall_seconds']-(time.monotonic()-begin)
            if remain<=0:raise TimeoutError('Parallel registered wall budget')
            code=child.wait(timeout=remain)
            if code:raise RuntimeError(f'Worker failed {code}')
        for log in logs:log.close()
        rows=[]
        for seed in c['seeds']:
          for arm in c['arms']:
            checkpoints[seed,arm]=out/f'seed{seed}_{arm}_{target}.pt'
            rows.append(json.loads((out/f'seed{seed}_{arm}_{target}_validation.json').read_text()))
        (out/f'validation_{target}.json').write_text(json.dumps(rows,indent=2));latest=target
        passed=all(r['validation']['sampled']>.2 and r['validation']['expected']>r['validation']['shuffled'] for r in rows if r['arm']=='both')
        print(json.dumps(dict(milestone=target,gate=passed,validation=rows,elapsed=time.monotonic()-begin)),flush=True)
        if passed:break
    except (TimeoutError,subprocess.TimeoutExpired) as exc:
        status='budget_exhausted';(out/'budget_stop.json').write_text(json.dumps(dict(error=repr(exc),last_common_update=latest)))
        for child in children:
            if child.poll() is None:child.terminate()
        for child in children:
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill()
    finally:
        for log in logs:log.close()
    # Final test is reserved until validation-only continuation finishes.
    torch.set_num_threads(1);data,_=read_data(Path(c['data']));images,labels=data['test'];final=[]
    for seed in c['seeds']:
      for arm in c['arms']:
        saved=torch.load(checkpoints[seed,arm],weights_only=True);speaker=Policy();listener=Policy(True);speaker.load_state_dict(saved['speaker']);listener.load_state_dict(saved['listener'])
        met=evaluate(speaker,listener,images,labels,torch.arange(10000),c['evaluation_pairs'],c['evaluation_seed']+1)
        final.append(dict(seed=seed,arm=arm,updates=latest,test=met,checkpoint=str(checkpoints[seed,arm])))
    result=dict(status=status,validation_gate_pass=passed,selected_updates=latest,elapsed=time.monotonic()-begin,records=final,scope='Independent paper reconstruction, no author code; original MNIST; no supervised policy loss; not paper-score replication')
    (out/'summary.json').write_text(json.dumps(result,indent=2));(out/'status.json').write_text(json.dumps(dict(stage=status,validation_gate_pass=passed)))
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
