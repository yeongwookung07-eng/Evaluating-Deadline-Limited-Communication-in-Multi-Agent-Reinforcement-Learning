"""Local vectorized equivalent of CommLearner; padded transitions excluded."""
import torch
from torch.nn import functional as F
from models.shared.buffers.episode import EpisodeBatch
from .learner import CommLearner
from .td_targets import td_target

class BatchedCommLearner(CommLearner):
 def train_batch(self,records):
  model=self.model;model.train();device=next(model.parameters()).device
  batch=EpisodeBatch.from_episodes([r.episode for r in records]).to(device)
  b,length,n,_=batch.observations.shape
  delivery=torch.zeros(b,length,n,dtype=torch.bool,device=device)
  for i,r in enumerate(records):delivery[i,:len(r.delivery_mask)]=torch.as_tensor(r.delivery_mask,device=device)
  def unroll(net):
   h=net.initial_hidden(b);previous=batch.observations.new_zeros(b,n,net.n_actions)
   cache=torch.stack([torch.as_tensor(r.initial_cache,device=device) for r in records])
   source=torch.stack([torch.as_tensor(r.initial_source,device=device) for r in records]);qs=[]
   for t in range(length):
    h=net.encode(batch.observations[:,t],previous,h);message=net.message(h)
    cache=torch.where(delivery[:,t,:,None],message,cache)
    source=torch.where(delivery[:,t],torch.full_like(source,t),source)
    qs.append(net.q(h,cache,source,t))
    if t<length-1:previous=F.one_hot(batch.actions[:,t],net.n_actions).float()
   return torch.stack(qs,1)
  online=unroll(model)
  with torch.no_grad():target=unroll(self.target)
  pred=model.joint(online[:,:-1].reshape(-1,n,model.n_actions),batch.actions.reshape(-1,n),batch.states[:,:-1].reshape(-1,batch.states.shape[-1])).reshape(b,-1)
  with torch.no_grad():
   chosen=online[:,1:].masked_fill(~batch.available_actions[:,1:],-torch.inf).argmax(-1)
   nxt=self.target.joint(target[:,1:].reshape(-1,n,model.n_actions),chosen.reshape(-1,n),batch.states[:,1:].reshape(-1,batch.states.shape[-1])).reshape(b,-1)
   expected=td_target(batch.rewards,nxt,batch.terminated,batch.truncated,self.config['gamma'],self.config.get('bootstrap_timeouts',True))
  assert pred.shape==expected.shape==batch.filled.shape
  loss=((pred-expected).square()*batch.filled).sum()/batch.filled.sum()
  if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
  self.optimizer.zero_grad(set_to_none=True);loss.backward()
  grad=torch.nn.utils.clip_grad_norm_(model.parameters(),self.config['grad_clip'],error_if_nonfinite=True)
  self.optimizer.step();self.updates+=1
  if self.updates%self.config['target_update_interval']==0:self.target.load_state_dict(model.state_dict())
  return dict(td_loss=float(loss.detach()),grad_norm=float(grad),updates=self.updates)
