from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(slots=True)
class Transition:
    state_board: np.ndarray
    state_side: int
    state_repeat: int
    action: int
    reward: float
    next_board: np.ndarray
    next_side: int
    next_repeat: int
    done: bool
    next_legal_actions: tuple[int, ...]


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.capacity = int(capacity)
        self.buffer: Deque[Transition] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(
        self,
        state_board,
        state_side: int,
        state_repeat: int,
        action: int,
        reward: float,
        next_board,
        next_side: int,
        next_repeat: int,
        done: bool,
        next_legal_actions,
    ) -> None:
        transition = Transition(
            state_board=np.asarray(state_board, dtype=np.int8).copy(),
            state_side=int(state_side),
            state_repeat=int(state_repeat),
            action=int(action),
            reward=float(reward),
            next_board=np.asarray(next_board, dtype=np.int8).copy(),
            next_side=int(next_side),
            next_repeat=int(next_repeat),
            done=bool(done),
            next_legal_actions=tuple(int(a) for a in next_legal_actions),
        )
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)
