# engine.py

import json
from dataclasses import asdict, dataclass

from bitboards import FILE_OF, RANK_OF
from board import DRAW_REPEAT_COUNT
from legal_moves import generate_legal_captures, generate_legal_moves
from rules import is_in_check

INF = 10**9
MATE_SCORE = 900_000
TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2
MAX_PLY = 128


@dataclass(slots=True)
class EngineParams:
    name: str = "balanced"
    king_value: int = 10000
    rook_value: int = 900
    knight_value: int = 450
    cannon_value: int = 500
    elephant_value: int = 220
    advisor_value: int = 220
    pawn_value: int = 120
    center_weight: int = 6
    major_advance_weight: int = 2
    pawn_advance_weight: int = 4
    pawn_cross_bonus: int = 50
    king_guard_weight: int = 18
    check_penalty: int = 120
    tempo_bonus: int = 12
    max_quiescence_depth: int = 8

    @classmethod
    def from_dict(cls, data):
        valid = {field.name for field in cls.__dataclass_fields__.values()}
        clean = {k: v for k, v in data.items() if k in valid}
        return cls(**clean)

    def to_dict(self):
        return asdict(self)


@dataclass(slots=True)
class TTEntry:
    depth: int
    score: int
    flag: int
    best_from: int
    best_to: int


PIECE_VALUE_ATTR = {
    1: "king_value",
    2: "rook_value",
    3: "knight_value",
    4: "cannon_value",
    5: "elephant_value",
    6: "advisor_value",
    7: "pawn_value",
}


PRESET_PARAMS = {
    "balanced": EngineParams(name="balanced"),
    "aggressive": EngineParams(
        name="aggressive",
        rook_value=920,
        knight_value=470,
        cannon_value=520,
        pawn_value=130,
        center_weight=8,
        major_advance_weight=3,
        pawn_advance_weight=5,
        pawn_cross_bonus=60,
        king_guard_weight=14,
        check_penalty=95,
        tempo_bonus=16,
        max_quiescence_depth=10,
    ),
    "solid": EngineParams(
        name="solid",
        rook_value=890,
        knight_value=440,
        cannon_value=490,
        pawn_value=118,
        center_weight=5,
        major_advance_weight=1,
        pawn_advance_weight=3,
        pawn_cross_bonus=42,
        king_guard_weight=24,
        check_penalty=140,
        tempo_bonus=8,
        max_quiescence_depth=6,
    ),
}


def params_from_preset(name):
    preset = PRESET_PARAMS.get(name)
    if preset is None:
        raise ValueError(f"未知引擎预设: {name}")
    return EngineParams.from_dict(preset.to_dict())


class XiangqiEngine:
    def __init__(self, params=None, tt_size=200_000):
        self.params = EngineParams() if params is None else params
        self.tt_size = tt_size
        self.tt = {}
        self.killers = [[None, None] for _ in range(MAX_PLY)]
        self.history = [[[0 for _ in range(90)] for _ in range(90)] for _ in range(2)]
        self.nodes = 0
        self.tt_hits = 0
        self.cutoffs = 0
        self.last_score = 0
        self.last_depth = 0

    # ======================
    # 参数管理
    # ======================
    def set_params(self, params):
        self.params = params

    def save_params(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.params.to_dict(), f, ensure_ascii=False, indent=2)

    def load_params(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.params = EngineParams.from_dict(data)

    # ======================
    # 估值
    # ======================
    def piece_value(self, piece):
        return getattr(self.params, PIECE_VALUE_ATTR[abs(piece)])

    def positional_bonus(self, piece, pos):
        x = FILE_OF[pos]
        y = RANK_OF[pos]
        side = 1 if piece > 0 else -1
        piece_type = abs(piece)

        score = max(0, 4 - abs(x - 4)) * self.params.center_weight

        if piece_type == 7:
            crossed = (y < 5) if side > 0 else (y > 4)
            if crossed:
                score += self.params.pawn_cross_bonus
            score += ((9 - y) if side > 0 else y) * self.params.pawn_advance_weight
        elif piece_type in (2, 3, 4):
            score += ((9 - y) if side > 0 else y) * self.params.major_advance_weight
        elif piece_type in (5, 6):
            score += self.params.king_guard_weight

        return score

    def side_score(self, board, side):
        total = 0
        b = board.board
        for pos in board.piece_positions[side]:
            piece = b[pos]
            total += self.piece_value(piece)
            total += self.positional_bonus(piece, pos)
        return total

    def evaluate(self, board):
        red_score = self.side_score(board, 1)
        black_score = self.side_score(board, -1)
        score = red_score - black_score

        if board.side < 0:
            score = -score

        score += self.params.tempo_bonus

        if is_in_check(board, board.side):
            score -= self.params.check_penalty

        return score

    # ======================
    # 搜索辅助
    # ======================
    @staticmethod
    def _side_index(side):
        return 0 if side > 0 else 1

    @staticmethod
    def _move_equals(move, route):
        return route is not None and move.from_pos == route[0] and move.to_pos == route[1]

    @staticmethod
    def _route_from_move(move):
        return move.from_pos, move.to_pos

    def _tt_key(self, board):
        return board.zhash, min(board.get_repeat_count(), DRAW_REPEAT_COUNT)

    def _tt_lookup(self, board, depth, alpha, beta):
        entry = self.tt.get(self._tt_key(board))
        if entry is None:
            return None, None

        tt_move = (entry.best_from, entry.best_to) if entry.best_from >= 0 else None
        if entry.depth < depth:
            return None, tt_move

        self.tt_hits += 1

        if entry.flag == TT_EXACT:
            return entry.score, tt_move
        if entry.flag == TT_LOWER and entry.score >= beta:
            return entry.score, tt_move
        if entry.flag == TT_UPPER and entry.score <= alpha:
            return entry.score, tt_move
        return None, tt_move

    def _tt_store(self, board, depth, score, flag, best_move):
        if len(self.tt) >= self.tt_size:
            self.tt.clear()

        if best_move is None:
            best_from = -1
            best_to = -1
        else:
            best_from = best_move.from_pos
            best_to = best_move.to_pos

        self.tt[self._tt_key(board)] = TTEntry(
            depth=depth,
            score=score,
            flag=flag,
            best_from=best_from,
            best_to=best_to,
        )

    def _record_cutoff(self, board, move, depth, ply):
        if board.board[move.to_pos] != 0:
            return

        if ply < MAX_PLY:
            killers = self.killers[ply]
            route = self._route_from_move(move)
            if killers[0] != route:
                killers[1] = killers[0]
                killers[0] = route

        history_bucket = self.history[self._side_index(board.side)]
        history_bucket[move.from_pos][move.to_pos] += depth * depth

    def _move_order_score(self, board, move, tt_move, ply):
        if self._move_equals(move, tt_move):
            return 10_000_000

        moving_piece = board.board[move.from_pos]
        captured = board.board[move.to_pos]
        score = 0

        if captured != 0:
            score += 1_000_000
            score += self.piece_value(captured) * 16
            score -= self.piece_value(moving_piece)
        else:
            killers = self.killers[ply] if ply < MAX_PLY else (None, None)
            route = self._route_from_move(move)
            if route == killers[0]:
                score += 900_000
            elif route == killers[1]:
                score += 800_000

        history_bucket = self.history[self._side_index(board.side)]
        score += history_bucket[move.from_pos][move.to_pos]
        score += self.positional_bonus(moving_piece, move.to_pos) // 2
        return score

    def _ordered_moves(self, board, moves, tt_move, ply):
        moves.sort(
            key=lambda mv: self._move_order_score(board, mv, tt_move, ply),
            reverse=True,
        )
        return moves

    # ======================
    # 搜索
    # ======================
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
                score = -self.quiescence(board, -beta, -alpha, ply + 1, qdepth + 1)
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
            score = -self.quiescence(board, -beta, -alpha, ply + 1, qdepth + 1)
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
            score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
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
            score = -self.negamax(board, depth - 1, -beta, -alpha, 1)
            board.undo_move()

            if score > best_score:
                best_score = score
                best_move = move

            if score > alpha:
                alpha = score

        self._tt_store(board, depth, best_score, TT_EXACT, best_move)
        return best_move, best_score

    def reset_search_state(self):
        self.killers = [[None, None] for _ in range(MAX_PLY)]
        self.history = [[[0 for _ in range(90)] for _ in range(90)] for _ in range(2)]
        self.nodes = 0
        self.tt_hits = 0
        self.cutoffs = 0
        self.last_score = 0
        self.last_depth = 0

    def choose_move(self, board, depth=4):
        self.reset_search_state()

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

    def analyze(self, board, depth=4):
        move = self.choose_move(board, depth=depth)
        return {
            "move": move,
            "score": self.last_score,
            "depth": self.last_depth,
            "nodes": self.nodes,
            "tt_hits": self.tt_hits,
            "cutoffs": self.cutoffs,
            "params": self.params.to_dict(),
        }


def create_engine(preset="balanced", tt_size=200_000):
    return XiangqiEngine(params=params_from_preset(preset), tt_size=tt_size)
