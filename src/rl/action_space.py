from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from move import Move
from legal_moves import generate_legal_moves

ACTION_DIM = 90 * 90


def move_to_action(move: Move) -> int:
    return move.from_pos * 90 + move.to_pos


def action_to_move(action_id: int) -> Move:
    if not 0 <= action_id < ACTION_DIM:
        raise ValueError(f"action_id out of range: {action_id}")
    from_pos, to_pos = divmod(action_id, 90)
    return Move(from_pos, to_pos)


def legal_moves(board) -> List[Move]:
    return generate_legal_moves(board)


def legal_action_ids(board) -> List[int]:
    return [move_to_action(move) for move in generate_legal_moves(board)]


def find_matching_legal_move(board, action_id: int) -> Optional[Move]:
    candidate = action_to_move(action_id)
    for move in generate_legal_moves(board):
        if move.from_pos == candidate.from_pos and move.to_pos == candidate.to_pos:
            return move
    return None


def choose_best_legal_action(q_values, legal_actions: Sequence[int]) -> Optional[int]:
    if not legal_actions:
        return None

    best_action = legal_actions[0]
    best_score = float(q_values[best_action])

    for action_id in legal_actions[1:]:
        score = float(q_values[action_id])
        if score > best_score:
            best_score = score
            best_action = action_id

    return best_action


def legal_action_mask(legal_actions: Iterable[int]):
    mask = [False] * ACTION_DIM
    for action_id in legal_actions:
        mask[action_id] = True
    return mask
