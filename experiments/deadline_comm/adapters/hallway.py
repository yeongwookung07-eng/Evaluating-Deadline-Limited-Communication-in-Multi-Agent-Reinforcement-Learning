"""Execute the upstream Hallway class unchanged with module-local dependencies.

Only its class AST is loaded: avoids importing SMAC/SC2 or patching numpy globally.
The upstream reward/state transition semantics, including aliasing, are preserved.
"""
import ast
import hashlib
import random
from pathlib import Path
import numpy as np

SOURCE = Path(__file__).resolve().parents[3] / "experiments/deadline_comm/environments/upstream/CASEC-MACO-benchmark/src/envs/hallway.py"

class _NumpyProxy:
    int = int
    def __init__(self, seed):
        self.random = np.random.RandomState(seed)
    def __getattr__(self, key):
        return getattr(np, key)

class Hallway:
    def __init__(self, n_agents=5, state_numbers=None, group_ids=None, seed=0, reward_win=10):
        state_numbers = state_numbers or [4] * n_agents
        group_ids = group_ids or [i % 2 for i in range(n_agents)]
        if len(state_numbers) != n_agents or len(group_ids) != n_agents or min(state_numbers) < 1:
            raise ValueError("Invalid Hallway dimensions")
        if sorted(set(group_ids)) != list(range(max(group_ids)+1)):
            raise ValueError("Group IDs must be contiguous")
        source = SOURCE.read_text()
        self.source_hash = hashlib.sha256(source.encode()).hexdigest()
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "HallwayEnv")
        proxy = _NumpyProxy(seed)
        namespace = {"MultiAgentEnv": object, "np": proxy, "random": random.Random(seed)}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
        self.env = namespace["HallwayEnv"](n_agents=n_agents, n_groups=max(group_ids)+1,
            state_numbers=state_numbers, group_ids=group_ids, seed=seed, reward_win=reward_win)
        self.rng = proxy.random
        self.n_agents, self.n_actions, self.obs_dim = n_agents, 3, 2
        self.state_dim, self.episode_length = n_agents*2, self.env.episode_limit
    def reset(self, seed=None):
        if seed is not None:
            self.rng.seed(seed)
        obs, _ = self.env.reset()
        return np.asarray(obs, dtype=np.float32), {}
    def step(self, actions):
        actions = np.asarray(actions)
        if actions.shape != (self.n_agents,) or not np.isin(actions, [0,1,2]).all():
            raise ValueError("Invalid joint action")
        reward, done, info = self.env.step(actions)
        natural = self.env._win_group == self.env.n_groups or self.env._fail_group == self.env.n_groups
        truncated = bool(done and not natural and self.env._episode_steps >= self.episode_length)
        info["success"] = bool(info["battle_won"])
        return np.asarray(self.env.get_obs(), dtype=np.float32), float(reward), bool(done and not truncated), truncated, info
    def get_state(self):
        return np.asarray(self.env.get_state(), dtype=np.float32)
    def available_actions(self):
        return np.asarray(self.env.get_avail_actions(), dtype=bool)
