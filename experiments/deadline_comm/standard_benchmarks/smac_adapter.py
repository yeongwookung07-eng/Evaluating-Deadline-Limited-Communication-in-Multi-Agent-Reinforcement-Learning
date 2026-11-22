"""SMACv2 API adapter preserving upstream dynamics and exposing local RNG control."""
from contextlib import contextmanager
from copy import deepcopy
import random
import numpy as np

def scenario_config(race):
    if race not in ('terran','protoss'):raise ValueError(race)
    units=['marine','marauder','medivac'] if race=='terran' else ['stalker','zealot','colossus']
    team=dict(dist_type='weighted_teams',unit_types=units,weights=[.45,.45,.1],observe=True)
    if race=='terran':team['exception_unit_types']=['medivac']
    return dict(map_name='10gen_'+race,capability_config=dict(n_units=5,n_enemies=5,team_gen=team,
        start_positions=dict(dist_type='surrounded_and_reflect',p=.5,n_enemies=5,map_x=32,map_y=32)),
        difficulty='7',conic_fov=False,use_unit_ranges=True,min_attack_range=2,obs_own_pos=True,
        continuing_episode=False,debug=False,step_mul=8,reward_scale=True,reward_scale_rate=20)

def seed_distributions(wrapper,seed):
    """Upstream default_rng() objects are NOT seeded by np.random.seed()."""
    seen=set();seq=np.random.SeedSequence(seed);nodes=[]
    def visit(obj):
        if id(obj) in seen:return
        seen.add(id(obj))
        if hasattr(obj,'rng'):nodes.append(obj)
        for _,value in sorted(vars(obj).items()):
            if type(value).__module__.startswith('smacv2.env.starcraft2.distributions'):visit(value)
    for _,obj in sorted(wrapper.env_key_to_distribution_map.items()):visit(obj)
    for obj,child in zip(nodes,seq.spawn(len(nodes))):obj.rng=np.random.default_rng(child)
    return len(nodes)

class SMACAdapter:
    def __init__(self,race='terran',engine_seed=0,raw=None):
        self.config=scenario_config(race);self.race=race;self.engine_seed=int(engine_seed)
        if raw is None:
            from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper
            raw=StarCraftCapabilityEnvWrapper(**deepcopy(self.config),seed=self.engine_seed)
        self.env=raw;info=raw.get_env_info()
        self.n_agents=info['n_agents'];self.n_actions=info['n_actions']
        self.obs_dim=info['obs_shape'];self.state_dim=info['state_shape'];self.episode_length=info['episode_limit']
        self.py_rng=random.Random(engine_seed);self.np_rng=np.random.RandomState(engine_seed);self.steps=0;self.done=True
    @contextmanager
    def local_rng(self):
        py,npstate=random.getstate(),np.random.get_state()
        random.setstate(self.py_rng.getstate());np.random.set_state(self.np_rng.get_state())
        try:yield
        finally:
            self.py_rng.setstate(random.getstate());self.np_rng.set_state(np.random.get_state())
            random.setstate(py);np.random.set_state(npstate)
    def reset(self,seed=None):
        if seed is not None:
            self.py_rng.seed(int(seed));self.np_rng.seed(int(seed));seed_distributions(self.env,int(seed))
        with self.local_rng():self.env.reset()
        self.steps=0;self.done=False
        return self.observation(),dict(engine_seed=self.engine_seed,scenario_seed=seed)
    def observation(self):
        obs=np.array(self.env.get_obs(),dtype=np.float32,copy=True)
        assert obs.shape==(self.n_agents,self.obs_dim) and np.isfinite(obs).all()
        return obs
    def get_state(self):
        state=np.array(self.env.get_state(),dtype=np.float32,copy=True)
        assert state.shape==(self.state_dim,) and np.isfinite(state).all()
        return state
    def available_actions(self):
        mask=np.array(self.env.get_avail_actions(),dtype=bool,copy=True)
        assert mask.shape==(self.n_agents,self.n_actions) and mask.any(axis=-1).all()
        return mask
    def step(self,actions):
        if self.done:raise RuntimeError('reset required after terminal transition')
        a=np.asarray(actions)
        if a.shape!=(self.n_agents,) or a.dtype.kind not in 'iu':raise ValueError('joint integer action required')
        if (a<0).any() or (a>=self.n_actions).any() or not self.available_actions()[np.arange(self.n_agents),a].all():raise ValueError('unavailable action')
        with self.local_rng():reward,done,info=self.env.step(a.tolist())
        self.steps+=1;self.done=bool(done);info=dict(info)
        # Native continuing_episode=False omits episode_limit from info. Preserve
        # native reward/done, distinguish true battle end using upstream counters.
        raw=getattr(self.env,'env',self.env)
        natural=bool(raw.win_counted or raw.defeat_counted)
        truncated=bool(done and not natural and (info.get('episode_limit',False) or self.steps>=self.episode_length))
        info.update(success=bool(info.get('battle_won',False)),native_done=bool(done),time_limit=truncated)
        return self.observation(),float(reward),bool(done and not truncated),truncated,info
    def close(self):self.env.close()
