from __future__ import annotations

import copy
from typing import Dict, Optional, Tuple

from board import Board, DRAW_REPEAT_COUNT
from legal_moves import generate_legal_moves

from .action_space import find_matching_legal_move, legal_action_ids
from .state_encoder import snapshot_board


class XiangqiEnv:
    """
    轻量环境。

    设计重点：
    1. 复用现有 Board / legal_moves / 长将过滤逻辑
    2. 奖励从“当前落子方”的视角返回
    3. 三次重复作为 AI / 训练内部和棋规则处理
    """

    def __init__(self, max_plies: int = 300, draw_repeat_count: int = DRAW_REPEAT_COUNT):
        self.max_plies = int(max_plies)
        self.draw_repeat_count = int(draw_repeat_count)
        self.board = None
        self.plies = 0
        self.reset()

    def clone_board(self, board) -> Board:
        return copy.deepcopy(board)

    def reset(self, board: Optional[Board] = None):
        if board is None:
            self.board = Board()
            self.board.init_startpos()
        else:
            self.board = self.clone_board(board)

        self.plies = len(self.board.history)
        return snapshot_board(self.board)

    def legal_moves(self):
        return generate_legal_moves(self.board)

    def legal_action_ids(self):
        return legal_action_ids(self.board)

    def current_snapshot(self):
        return snapshot_board(self.board)

    def step(self, action_id: int):
        move = find_matching_legal_move(self.board, action_id)
        if move is None:
            raise ValueError(f"illegal action_id for current state: {action_id}")

        moving_side = self.board.side
        self.board.make_move(move)
        self.plies += 1

        done = False
        reward = 0.0
        info: Dict[str, object] = {
            "winner": None,
            "draw": False,
            "reason": None,
            "moved_side": moving_side,
        }

        if self.board.is_draw_by_repetition(self.draw_repeat_count):
            done = True
            info["draw"] = True
            info["reason"] = "threefold_repetition"
        else:
            opponent_moves = generate_legal_moves(self.board)
            if not opponent_moves:
                done = True
                reward = 1.0
                info["winner"] = moving_side
                info["reason"] = "mate_or_no_legal_moves"
            elif self.plies >= self.max_plies:
                done = True
                info["draw"] = True
                info["reason"] = "max_plies"

        return snapshot_board(self.board), reward, done, info
