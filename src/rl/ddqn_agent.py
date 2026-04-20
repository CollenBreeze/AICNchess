from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .action_space import ACTION_DIM, choose_best_legal_action
from .ddqn_model import DuelingDDQN
from .state_encoder import NUM_CHANNELS, batch_encode_snapshots


@dataclass(slots=True)
class DDQNConfig:
    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = 64
    min_buffer_size: int = 2_000
    target_sync_interval: int = 1_000
    grad_clip_norm: float = 5.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 200_000
    channels: int = 64
    num_blocks: int = 2
    hidden_dim: int = 256


class DDQNAgent:
    """
    单网络自对弈版本。

    关键点：
    - state 已经包含 side-to-move
    - 下一状态轮到对手走，所以 bootstrap 需要“取负号”
      target = reward - gamma * Q_target(s', argmax_a Q_online(s', a))
    """

    def __init__(self, config: Optional[DDQNConfig] = None, device: Optional[str] = None):
        self.config = DDQNConfig() if config is None else config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.online_net = DuelingDDQN(
            in_channels=NUM_CHANNELS,
            action_dim=ACTION_DIM,
            channels=self.config.channels,
            num_blocks=self.config.num_blocks,
            hidden_dim=self.config.hidden_dim,
        ).to(self.device)
        self.target_net = DuelingDDQN(
            in_channels=NUM_CHANNELS,
            action_dim=ACTION_DIM,
            channels=self.config.channels,
            num_blocks=self.config.num_blocks,
            hidden_dim=self.config.hidden_dim,
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.config.lr)
        self.training_steps = 0

    def epsilon(self, step: Optional[int] = None) -> float:
        if step is None:
            step = self.training_steps
        if step >= self.config.epsilon_decay_steps:
            return self.config.epsilon_end
        ratio = step / max(1, self.config.epsilon_decay_steps)
        return self.config.epsilon_start + ratio * (self.config.epsilon_end - self.config.epsilon_start)

    def select_action(
        self,
        encoded_state: np.ndarray,
        legal_actions,
        explore: bool = True,
        force_epsilon: Optional[float] = None,
    ) -> Optional[int]:
        legal_actions = list(legal_actions)
        if not legal_actions:
            return None

        epsilon = self.epsilon() if force_epsilon is None else float(force_epsilon)
        if explore and random.random() < epsilon:
            return random.choice(legal_actions)

        state_tensor = torch.from_numpy(encoded_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(state_tensor)[0].detach().cpu().numpy()

        return choose_best_legal_action(q_values, legal_actions)

    def sync_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def train_step(self, replay_buffer):
        if len(replay_buffer) < max(self.config.batch_size, self.config.min_buffer_size):
            return None

        batch = replay_buffer.sample(self.config.batch_size)

        state_snapshots = [(t.state_board, t.state_side, t.state_repeat) for t in batch]
        next_snapshots = [(t.next_board, t.next_side, t.next_repeat) for t in batch]

        states = torch.from_numpy(batch_encode_snapshots(state_snapshots)).to(self.device)
        next_states = torch.from_numpy(batch_encode_snapshots(next_snapshots)).to(self.device)

        actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device)

        q_values = self.online_net(states)
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_online_q = self.online_net(next_states)
            next_target_q = self.target_net(next_states)

            next_q_list = []
            for i, transition in enumerate(batch):
                legal_actions = transition.next_legal_actions
                if transition.done or not legal_actions:
                    next_q_list.append(next_target_q.new_tensor(0.0))
                    continue

                legal_idx = torch.tensor(legal_actions, dtype=torch.long, device=self.device)
                best_pos = torch.argmax(next_online_q[i, legal_idx])
                best_action = legal_idx[best_pos]
                next_q_list.append(next_target_q[i, best_action])

            next_q = torch.stack(next_q_list)

            # 零和自对弈：下一状态是“对手视角”，因此 bootstrap 取负
            targets = rewards - (1.0 - dones) * self.config.gamma * next_q

        loss = F.smooth_l1_loss(current_q, targets)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.config.grad_clip_norm)
        self.optimizer.step()

        self.training_steps += 1
        if self.training_steps % self.config.target_sync_interval == 0:
            self.sync_target()

        return {
            "loss": float(loss.item()),
            "epsilon": float(self.epsilon()),
            "q_mean": float(current_q.mean().item()),
            "target_mean": float(targets.mean().item()),
            "steps": int(self.training_steps),
        }

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.config),
                "model_state": self.online_net.state_dict(),
                "target_state": self.target_net.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "training_steps": self.training_steps,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: Optional[str] = None):
        payload = torch.load(path, map_location=device or ("cuda" if torch.cuda.is_available() else "cpu"))
        config = DDQNConfig(**payload["config"])
        agent = cls(config=config, device=device)
        agent.online_net.load_state_dict(payload["model_state"])
        agent.target_net.load_state_dict(payload.get("target_state", payload["model_state"]))
        if "optimizer_state" in payload:
            agent.optimizer.load_state_dict(payload["optimizer_state"])
        agent.training_steps = int(payload.get("training_steps", 0))
        agent.online_net.eval()
        agent.target_net.eval()
        return agent
