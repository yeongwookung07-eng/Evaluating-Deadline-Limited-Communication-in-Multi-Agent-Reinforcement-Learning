"""Algorithm-independent recurrent double Q-learning with complete-episode BPTT."""
from copy import deepcopy
from pathlib import Path
import torch
from torch.nn import functional as F
from models.base import RecurrentJointQ
from models.shared.buffers.episode import EpisodeBatch


def make_optimizer(parameters, name: str, learning_rate: float):
    if name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate)
    if name == "rmsprop":
        return torch.optim.RMSprop(parameters, lr=learning_rate)
    raise ValueError(f"Unsupported optimizer: {name}")


class TDLearner:
    def __init__(self, model: RecurrentJointQ, config: dict):
        self.model, self.config = model, dict(config)
        self.target = deepcopy(model).eval()
        self.target.requires_grad_(False)
        self.optimizer = make_optimizer(model.parameters(), config["optimizer"], config["learning_rate"])
        self.updates = 0

    @staticmethod
    def _unroll(model: RecurrentJointQ, batch: EpisodeBatch) -> list[torch.Tensor]:
        b, length, n, _ = batch.observations.shape
        hidden = model.initial_hidden(b)
        previous = batch.observations.new_zeros(b, n, model.n_actions)
        representations = []
        for t in range(length):
            hidden = model.encode(batch.observations[:, t], previous, hidden)
            representations.append(hidden)
            if t < length - 1:
                previous = F.one_hot(batch.actions[:, t], model.n_actions).to(batch.observations.dtype)
        return representations

    def train_batch(self, batch: EpisodeBatch) -> dict[str, float | int]:
        self.model.train()
        batch = batch.to(next(self.model.parameters()).device)
        online = self._unroll(self.model, batch)
        with torch.no_grad():
            target = self._unroll(self.target, batch)
        q_values, targets = [], []
        for t in range(batch.actions.shape[1]):
            q_values.append(self.model.joint_q(online[t], batch.actions[:, t], batch.states[:, t]))
            with torch.no_grad():
                # Double Q: online network selects; target network evaluates.
                actions = self.model.greedy_actions(online[t + 1].detach(), batch.available_actions[:, t + 1])
                next_q = self.target.joint_q(target[t + 1], actions, batch.states[:, t + 1])
                targets.append(batch.rewards[:, t] + self.config["gamma"]
                               * (~batch.terminated[:, t]).float() * next_q)
        prediction, expected = torch.stack(q_values, 1), torch.stack(targets, 1)
        loss = ((prediction - expected).square() * batch.filled).sum() / batch.filled.sum()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite TD loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["grad_clip"], error_if_nonfinite=True)
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.config["target_update_interval"] == 0:
            self.target.load_state_dict(self.model.state_dict())
        return {"td_loss": float(loss.detach()), "grad_norm": float(grad_norm), "learner_updates": self.updates}

    def save_checkpoint(self, path: str | Path, metadata: dict | None = None) -> None:
        """Save learning state. Replay/environment state is not an exact run resume."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save({"format_version": 1, "model": self.model.state_dict(),
                    "target": self.target.state_dict(), "optimizer": self.optimizer.state_dict(),
                    "updates": self.updates, "training_config": self.config,
                    "metadata": metadata or {}}, temporary)
        temporary.replace(path)

    def load_checkpoint(self, path: str | Path) -> dict:
        saved = torch.load(path, map_location=next(self.model.parameters()).device, weights_only=True)
        if saved["format_version"] != 1 or saved["training_config"] != self.config:
            raise ValueError("Checkpoint format/training configuration mismatch")
        self.model.load_state_dict(saved["model"])
        self.target.load_state_dict(saved["target"])
        self.optimizer.load_state_dict(saved["optimizer"])
        self.updates = saved["updates"]
        return saved["metadata"]
