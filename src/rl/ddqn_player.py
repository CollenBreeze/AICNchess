from __future__ import annotations

from pathlib import Path
from typing import Optional

from .action_space import action_to_move, legal_action_ids
from .ddqn_agent import DDQNAgent
from .state_encoder import encode_board

_AGENT_CACHE = {}


def load_agent(weights_path: str, device: Optional[str] = None) -> DDQNAgent:
    key = (str(Path(weights_path).resolve()), device)
    agent = _AGENT_CACHE.get(key)
    if agent is None:
        agent = DDQNAgent.load(weights_path, device=device)
        _AGENT_CACHE[key] = agent
    return agent


def choose_ddqn_move(board, weights_path: str, device: Optional[str] = None):
    legal_actions = legal_action_ids(board)
    if not legal_actions:
        return None

    agent = load_agent(weights_path, device=device)
    state = encode_board(board)
    action_id = agent.select_action(state, legal_actions, explore=False, force_epsilon=0.0)
    if action_id is None:
        return None
    return action_to_move(action_id)
