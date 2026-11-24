"""Whole-episode replay: recurrent states are recomputed from episode start."""
from dataclasses import dataclass, fields
from collections import deque
import numpy as np
import torch


@dataclass
class Episode:
    observations: np.ndarray  # [T+1,N,O]
    states: np.ndarray        # [T+1,S], central training data only
    available_actions: np.ndarray  # [T+1,N,A]
    actions: np.ndarray       # [T,N]
    rewards: np.ndarray       # [T]
    terminated: np.ndarray    # [T], excludes time-limit truncation
    truncated: np.ndarray     # [T]

    def __post_init__(self):
        t = len(self.actions)
        if t < 1 or any(len(getattr(self, k)) != t + 1 for k in
                        ("observations", "states", "available_actions")):
            raise ValueError("Episode requires T transitions and T+1 observations/states/masks")
        if any(getattr(self, k).shape != (t,) for k in ("rewards", "terminated", "truncated")):
            raise ValueError("Rewards and terminal flags must have shape [T]")
        done = self.terminated | self.truncated
        if done[:-1].any() or not done[-1]:
            raise ValueError("Replay stores complete episodes, ending on the last transition")


@dataclass
class EpisodeBatch:
    observations: torch.Tensor  # [B,T+1,N,O]
    states: torch.Tensor        # [B,T+1,S]
    available_actions: torch.Tensor  # [B,T+1,N,A]
    actions: torch.Tensor       # [B,T,N]
    rewards: torch.Tensor       # [B,T]
    terminated: torch.Tensor    # [B,T]
    truncated: torch.Tensor     # [B,T]
    filled: torch.Tensor        # [B,T], 1 for real transitions, 0 for padding

    def to(self, device: str | torch.device) -> "EpisodeBatch":
        return EpisodeBatch(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})

    @classmethod
    def from_episodes(cls, episodes: list[Episode]) -> "EpisodeBatch":
        if not episodes:
            raise ValueError("Cannot batch zero episodes")
        b, t = len(episodes), max(len(e.actions) for e in episodes)
        result = {}
        for field in fields(Episode):
            key = field.name
            example = getattr(episodes[0], key)
            extra = int(key in ("observations", "states", "available_actions"))
            # Padding masks permit actions so Max-Sum can process padded samples;
            # filled removes their losses. Real masks are copied without alteration.
            fill = 1 if key == "available_actions" else 0
            value = np.full((b, t + extra, *example.shape[1:]), fill, dtype=example.dtype)
            for i, episode in enumerate(episodes):
                item = getattr(episode, key)
                if item.shape[1:] != example.shape[1:]:
                    raise ValueError("Episode feature dimensions must agree")
                value[i, :len(item)] = item
            result[key] = torch.from_numpy(value)
        result["filled"] = torch.tensor([[float(j < len(e.actions)) for j in range(t)]
                                         for e in episodes])
        return cls(**result)


class EpisodeReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0):
        if capacity < 1:
            raise ValueError("Replay capacity must be positive")
        self.episodes: deque[Episode] = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.episodes)

    def insert(self, episode: Episode) -> None:
        # Own the arrays: callers cannot retroactively modify experience.
        self.episodes.append(Episode(**{f.name: getattr(episode, f.name).copy() for f in fields(episode)}))

    def sample(self, batch_size: int) -> EpisodeBatch:
        if not 1 <= batch_size <= len(self):
            raise ValueError("Not enough complete episodes for a sample without replacement")
        indices = self.rng.choice(len(self), batch_size, replace=False)
        return EpisodeBatch.from_episodes([self.episodes[int(i)] for i in indices])
