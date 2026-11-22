import random
import numpy as np
import pytest
from ..smac_adapter import SMACAdapter, seed_distributions

class Fake:
    def __init__(self):
        self.env_key_to_distribution_map={};self.win_counted=False;self.defeat_counted=False;self.t=0;self.win=False
    def get_env_info(self):return dict(n_agents=2,n_actions=3,obs_shape=2,state_shape=2,episode_limit=2)
    def reset(self):self.t=0;random.random();np.random.rand()
    def get_obs(self):return np.full((2,2),self.t)
    def get_state(self):return np.full(2,self.t)
    def get_avail_actions(self):return [[1,1,0],[1,0,0]]
    def step(self,a):
        self.t+=1;self.win_counted=self.win and self.t==2
        return float(self.win_counted),self.t==2,dict(battle_won=self.win_counted)
    def close(self):pass

def test_masks_and_ownership():
    env=SMACAdapter(raw=Fake());o,_=env.reset(1);o[:]=99
    assert (env.observation()==0).all()
    with pytest.raises(ValueError):env.step(np.array([2,0]))
    with pytest.raises(ValueError):env.step(np.array([0,1]))
    env.step(np.array([1,0]));assert (env.get_state()==1).all()
def test_timeout_vs_true_win_at_limit():
    for win in [False,True]:
        raw=Fake();raw.win=win;env=SMACAdapter(raw=raw);env.reset(2)
        env.step(np.array([0,0]));_,r,term,trunc,_=env.step(np.array([0,0]))
        assert term==win and trunc!=win and r==float(win)
        with pytest.raises(RuntimeError):env.step(np.array([0,0]))
def test_rng_does_not_contaminate_caller():
    env=SMACAdapter(raw=Fake());random.seed(123);np.random.seed(123)
    py=random.getstate();state=np.random.get_state();env.reset(5)
    assert random.getstate()==py;np.testing.assert_array_equal(np.random.get_state()[1],state[1])
def test_all_nested_distribution_generators_seeded():
    Node=type('Node',(),{});Node.__module__='smacv2.env.starcraft2.distributions'
    p=Node();p.rng=np.random.default_rng();p.child=Node();p.child.rng=np.random.default_rng()
    raw=Fake();raw.env_key_to_distribution_map={'team':p}
    assert seed_distributions(raw,9)==2
    a=(p.rng.random(),p.child.rng.random());seed_distributions(raw,9)
    assert a==(p.rng.random(),p.child.rng.random()) and a[0]!=a[1]
