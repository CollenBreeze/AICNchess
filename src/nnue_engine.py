# nnue_engine.py

from __future__ import annotations

from engine import INF, MATE_SCORE, TT_EXACT, TT_LOWER, TT_UPPER, XiangqiEngine, params_from_preset
from legal_moves import generate_legal_captures, generate_legal_moves
from nnue_model import XiangqiNNUE
from rules import is_in_check


class NNUEEngine(XiangqiEngine):
    def __init__(self, nnue: XiangqiNNUE, params=None, tt_size=200_000, eval_cache_size=200_000):
        super().__init__(params=params, tt_size=tt_size)
        self.nnue = nnue
        self.eval_cache = {}
        self.eval_cache_size = int(eval_cache_size)
        self.acc_stack = []

    def reset_search_state(self, board=None):
        super().reset_search_state()
        self.eval_cache = {}
        self.acc_stack = []
        if board is not None:
            self.acc_stack.append(self.nnue.build_accumulator(board))

    def _current_accumulator(self):
        if not self.acc_stack:
            return None
        return self.acc_stack[-1]

    def _push_accumulator_after_move(self, board):
        if not self.acc_stack:
            self.acc_stack.append(self.nnue.build_accumulator(board))
            return
        self.acc_stack.append(self.nnue.update_accumulator(self.acc_stack[-1], board))

    def _pop_accumulator(self):
        if self.acc_stack:
            self.acc_stack.pop()

    def evaluate(self, board):
        key = board.zhash
        cached = self.eval_cache.get(key)
        if cached is not None:
            return cached

        base_score = super().evaluate(board)
        acc = self._current_accumulator()
        if acc is None or acc.key != board.zhash:
            acc = self.nnue.build_accumulator(board)
        correction = self.nnue.evaluate_accumulator(acc, board.side)
        score = int(base_score + correction)

        if len(self.eval_cache) >= self.eval_cache_size:
            self.eval_cache.clear()
        self.eval_cache[key] = score
        return score

    def quiescence(self, board, alpha, beta, ply, qdepth):
        self.nodes += 1

        if board.is_draw_by_repetition():
            return 0

        if is_in_check(board, board.side):
            moves = generate_legal_moves(board)
            if not moves:
                return -MATE_SCORE + ply

            self._ordered_moves(board, moves, None, ply)

            for move in moves:
                board.make_move(move)
                self._push_accumulator_after_move(board)
                score = -self.quiescence(board, -beta, -alpha, ply + 1, qdepth + 1)
                self._pop_accumulator()
                board.undo_move()

                if score >= beta:
                    self.cutoffs += 1
                    return beta
                if score > alpha:
                    alpha = score

            return alpha

        stand_pat = self.evaluate(board)
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        if qdepth >= self.params.max_quiescence_depth:
            return alpha

        moves = generate_legal_captures(board)
        if not moves:
            return alpha

        self._ordered_moves(board, moves, None, ply)

        for move in moves:
            captured = board.board[move.to_pos]
            if captured != 0 and stand_pat + self.piece_value(captured) + 32 < alpha:
                continue

            board.make_move(move)
            self._push_accumulator_after_move(board)
            score = -self.quiescence(board, -beta, -alpha, ply + 1, qdepth + 1)
            self._pop_accumulator()
            board.undo_move()

            if score >= beta:
                self.cutoffs += 1
                return beta
            if score > alpha:
                alpha = score

        return alpha

    def negamax(self, board, depth, alpha, beta, ply):
        self.nodes += 1
        alpha_orig = alpha

        if board.is_draw_by_repetition():
            return 0

        tt_score, tt_move = self._tt_lookup(board, depth, alpha, beta)
        if tt_score is not None:
            return tt_score

        if depth <= 0:
            if is_in_check(board, board.side):
                depth = 1
            else:
                return self.quiescence(board, alpha, beta, ply, 0)

        moves = generate_legal_moves(board)
        if not moves:
            return -MATE_SCORE + ply

        best_move = None
        best_score = -INF
        self._ordered_moves(board, moves, tt_move, ply)

        for move in moves:
            board.make_move(move)
            self._push_accumulator_after_move(board)
            score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            self._pop_accumulator()
            board.undo_move()

            if score > best_score:
                best_score = score
                best_move = move

            if score > alpha:
                alpha = score

            if alpha >= beta:
                self.cutoffs += 1
                self._record_cutoff(board, move, depth, ply)
                break

        if best_score <= alpha_orig:
            flag = TT_UPPER
        elif best_score >= beta:
            flag = TT_LOWER
        else:
            flag = TT_EXACT
        self._tt_store(board, depth, best_score, flag, best_move)
        return best_score

    def _root_search(self, board, depth):
        best_move = None
        best_score = -INF
        alpha = -INF
        beta = INF

        tt_entry = self.tt.get(self._tt_key(board))
        tt_move = None
        if tt_entry is not None and tt_entry.best_from >= 0:
            tt_move = (tt_entry.best_from, tt_entry.best_to)

        moves = generate_legal_moves(board)
        if not moves:
            return None, -MATE_SCORE

        self._ordered_moves(board, moves, tt_move, 0)

        for move in moves:
            board.make_move(move)
            self._push_accumulator_after_move(board)
            score = -self.negamax(board, depth - 1, -beta, -alpha, 1)
            self._pop_accumulator()
            board.undo_move()

            if score > best_score:
                best_score = score
                best_move = move

            if score > alpha:
                alpha = score

        self._tt_store(board, depth, best_score, TT_EXACT, best_move)
        return best_move, best_score

    def choose_move(self, board, depth=4):
        self.reset_search_state(board)

        best_move = None
        best_score = -INF

        for current_depth in range(1, depth + 1):
            move, score = self._root_search(board, current_depth)
            if move is not None:
                best_move = move
                best_score = score
            self.last_depth = current_depth
            self.last_score = best_score

        return best_move



def create_nnue_engine(weights_path, preset="balanced", tt_size=200_000, eval_cache_size=200_000):
    nnue = XiangqiNNUE.load(weights_path)
    return NNUEEngine(
        nnue=nnue,
        params=params_from_preset(preset),
        tt_size=tt_size,
        eval_cache_size=eval_cache_size,
    )
