"""Two diagnostic information tasks, explicitly not a standard benchmark port.

At each step fresh independent bits are drawn before decisions. Only one source
(or two for XOR) observes them. Followers receive neither source actions nor
rewards as observations. Expected follower accuracy without communication <=1/2
regardless of recurrent memory; an ideal bit-sharing oracle achieves 1.
"""
import numpy as np


class HiddenSource:
    def __init__(self,n_agents=5,task='broadcast',horizon=20,seed=0):
        if n_agents<3 or task not in ('broadcast','xor'):raise ValueError('Invalid task')
        self.n_agents=n_agents;self.n_actions=2;self.obs_dim=4;self.state_dim=2*n_agents
        self.task=task;self.episode_length=horizon;self.rng=np.random.default_rng(seed)
        self.n_sources=1 if task=='broadcast' else 2
    def sample(self):self.bits=self.rng.integers(0,2,size=self.n_sources)
    def obs(self):
        obs=np.zeros((self.n_agents,4),np.float32)
        for i in range(self.n_sources):obs[i,0]=2*self.bits[i]-1;obs[i,1]=1;obs[i,2]=i
        obs[:,3]=self.t/self.episode_length
        return obs
    def reset(self,seed=None):
        if seed is not None:self.rng=np.random.default_rng(seed)
        self.t=0;self.sample();return self.obs(),{}
    def get_state(self):
        state=np.zeros(self.state_dim,np.float32);state[:self.n_sources]=self.bits
        state[-1]=self.t/self.episode_length;return state
    def available_actions(self):return np.ones((self.n_agents,2),bool)
    def oracle_actions(self):return np.full(self.n_agents,int(np.bitwise_xor.reduce(self.bits)))
    def step(self,actions):
        actions=np.asarray(actions)
        if actions.shape!=(self.n_agents,) or not np.isin(actions,[0,1]).all():raise ValueError('Invalid action')
        correct=actions[self.n_sources:]==int(np.bitwise_xor.reduce(self.bits))
        reward=float(correct.mean());self.t+=1;done=self.t>=self.episode_length;self.sample()
        return self.obs(),reward,done,False,dict(success=bool(correct.all()),follower_accuracy=reward)
