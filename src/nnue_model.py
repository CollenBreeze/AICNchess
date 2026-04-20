# nnue_model.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

NUM_SQUARES = 90
ACC_SIZE_DEFAULT = 64
HIDDEN_SIZE_DEFAULT = 32

NON_KING_PIECES = (2, 3, 4, 5, 6, 7, -2, -3, -4, -5, -6, -7)
PIECE_TO_FEATURE = {piece: idx for idx, piece in enumerate(NON_KING_PIECES)}
NUM_PIECE_FEATURES = len(NON_KING_PIECES)

# 在“己方在下方”的统一朝向里，将/帅所在九宫的 9 个格子
PALACE_BUCKET_SQUARES = (66, 67, 68, 75, 76, 77, 84, 85, 86)
PALACE_BUCKET_INDEX = {sq: idx for idx, sq in enumerate(PALACE_BUCKET_SQUARES)}
NUM_KING_BUCKETS = len(PALACE_BUCKET_SQUARES)
NUM_FEATURES = NUM_KING_BUCKETS * NUM_PIECE_FEATURES * NUM_SQUARES

EPS = 1e-9


def rotate180_square(pos: int) -> int:
    x = pos % 9
    y = pos // 9
    return (9 - y) * 9 + (8 - x)


def orient_square(pos: int, perspective: int) -> int:
    if perspective > 0:
        return int(pos)
    return rotate180_square(int(pos))


def palace_bucket_from_king(king_pos: int, perspective: int) -> int:
    oriented = orient_square(king_pos, perspective)
    bucket = PALACE_BUCKET_INDEX.get(oriented)
    if bucket is not None:
        return bucket

    # 理论上不会发生（将帅应在九宫内），但为了健壮性给一个就近桶
    x = oriented % 9
    y = oriented // 9
    x = min(5, max(3, x))
    y = min(9, max(7, y))
    return (y - 7) * 3 + (x - 3)



def find_king_positions(board_array: Iterable[int]) -> tuple[int, int]:
    red_king = -1
    black_king = -1
    for pos, piece in enumerate(board_array):
        if piece == 1:
            red_king = pos
        elif piece == -1:
            black_king = pos
    return red_king, black_king



def feature_index(king_bucket: int, piece: int, piece_pos: int, perspective: int) -> int:
    piece_idx = PIECE_TO_FEATURE[piece]
    oriented_piece_sq = orient_square(piece_pos, perspective)
    return ((king_bucket * NUM_PIECE_FEATURES + piece_idx) * NUM_SQUARES) + oriented_piece_sq



def active_feature_indices_from_array(board_array, red_king: int, black_king: int, perspective: int) -> np.ndarray:
    king_pos = red_king if perspective > 0 else black_king
    if king_pos < 0:
        return np.empty(0, dtype=np.int32)

    king_bucket = palace_bucket_from_king(king_pos, perspective)
    indices = []
    for pos, piece in enumerate(board_array):
        if piece == 0 or abs(piece) == 1:
            continue
        indices.append(feature_index(king_bucket, int(piece), pos, perspective))
    return np.asarray(indices, dtype=np.int32)



def feature_pair_from_board(board) -> tuple[np.ndarray, np.ndarray]:
    red_king = board.king_pos.get(1, -1)
    black_king = board.king_pos.get(-1, -1)
    board_array = board.board
    return (
        active_feature_indices_from_array(board_array, red_king, black_king, 1),
        active_feature_indices_from_array(board_array, red_king, black_king, -1),
    )



def dense_pair_from_board(board) -> tuple[np.ndarray, np.ndarray]:
    red_idx, black_idx = feature_pair_from_board(board)
    red = np.zeros(NUM_FEATURES, dtype=np.float32)
    black = np.zeros(NUM_FEATURES, dtype=np.float32)
    red[red_idx] = 1.0
    black[black_idx] = 1.0
    return red, black


@dataclass(slots=True)
class NNUEConfig:
    acc_size: int = ACC_SIZE_DEFAULT
    hidden_size: int = HIDDEN_SIZE_DEFAULT
    version: str = "xiangqi-nnue-acc-v3"


@dataclass(slots=True)
class AccumulatorState:
    red_bucket: int
    black_bucket: int
    red_raw: np.ndarray
    black_raw: np.ndarray
    red_psqt: float
    black_psqt: float
    key: int | None = None


class XiangqiNNUE:
    def __init__(
        self,
        *,
        ft_weight: np.ndarray,
        ft_bias: np.ndarray,
        psqt_weight: np.ndarray,
        psqt_bias: float,
        l1_weight: np.ndarray,
        l1_bias: np.ndarray,
        l2_weight: np.ndarray,
        l2_bias: float,
        version: str = "xiangqi-nnue-acc-v3",
    ):
        self.ft_weight = np.asarray(ft_weight, dtype=np.float32)
        self.ft_bias = np.asarray(ft_bias, dtype=np.float32)
        self.psqt_weight = np.asarray(psqt_weight, dtype=np.float32).reshape(-1)
        self.psqt_bias = float(psqt_bias)
        self.l1_weight = np.asarray(l1_weight, dtype=np.float32)
        self.l1_bias = np.asarray(l1_bias, dtype=np.float32)
        self.l2_weight = np.asarray(l2_weight, dtype=np.float32).reshape(1, -1)
        self.l2_bias = float(l2_bias)
        self.version = str(version)

        self.acc_size = int(self.ft_bias.shape[0])
        self.hidden_size = int(self.l1_bias.shape[0])

    @classmethod
    def zeros(cls, acc_size: int = ACC_SIZE_DEFAULT, hidden_size: int = HIDDEN_SIZE_DEFAULT):
        return cls(
            ft_weight=np.zeros((acc_size, NUM_FEATURES), dtype=np.float32),
            ft_bias=np.zeros(acc_size, dtype=np.float32),
            psqt_weight=np.zeros(NUM_FEATURES, dtype=np.float32),
            psqt_bias=0.0,
            l1_weight=np.zeros((hidden_size, acc_size * 2), dtype=np.float32),
            l1_bias=np.zeros(hidden_size, dtype=np.float32),
            l2_weight=np.zeros((1, hidden_size), dtype=np.float32),
            l2_bias=0.0,
        )

    @classmethod
    def load(cls, path: str | Path):
        data = np.load(path, allow_pickle=False)
        return cls(
            ft_weight=data["ft_weight"],
            ft_bias=data["ft_bias"],
            psqt_weight=data["psqt_weight"],
            psqt_bias=float(data["psqt_bias"]),
            l1_weight=data["l1_weight"],
            l1_bias=data["l1_bias"],
            l2_weight=data["l2_weight"],
            l2_bias=float(data["l2_bias"]),
            version=str(data.get("version", "xiangqi-nnue-acc-v3")),
        )

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            ft_weight=self.ft_weight,
            ft_bias=self.ft_bias,
            psqt_weight=self.psqt_weight,
            psqt_bias=np.asarray(self.psqt_bias, dtype=np.float32),
            l1_weight=self.l1_weight,
            l1_bias=self.l1_bias,
            l2_weight=self.l2_weight,
            l2_bias=np.asarray(self.l2_bias, dtype=np.float32),
            version=np.asarray(self.version),
        )

    def _empty_raw_accumulator(self) -> tuple[np.ndarray, float]:
        return self.ft_bias.copy(), float(self.psqt_bias)

    def _accumulate_raw(self, feature_indices: np.ndarray) -> tuple[np.ndarray, float]:
        acc = self.ft_bias.copy()
        psqt = float(self.psqt_bias)
        if feature_indices.size:
            acc += self.ft_weight[:, feature_indices].sum(axis=1)
            psqt += float(self.psqt_weight[feature_indices].sum())
        return acc, psqt

    def _build_side_raw(self, board, perspective: int) -> tuple[int, np.ndarray, float]:
        king_pos = board.king_pos.get(perspective, -1)
        if king_pos < 0:
            raw, psqt = self._empty_raw_accumulator()
            return -1, raw, psqt

        red_king = board.king_pos.get(1, -1)
        black_king = board.king_pos.get(-1, -1)
        indices = active_feature_indices_from_array(board.board, red_king, black_king, perspective)
        raw, psqt = self._accumulate_raw(indices)
        return palace_bucket_from_king(king_pos, perspective), raw, psqt

    def build_accumulator(self, board) -> AccumulatorState:
        red_bucket, red_raw, red_psqt = self._build_side_raw(board, 1)
        black_bucket, black_raw, black_psqt = self._build_side_raw(board, -1)
        return AccumulatorState(
            red_bucket=red_bucket,
            black_bucket=black_bucket,
            red_raw=red_raw,
            black_raw=black_raw,
            red_psqt=red_psqt,
            black_psqt=black_psqt,
            key=board.zhash,
        )

    def _apply_feature_delta(self, raw: np.ndarray, psqt: float, feature_idx: int, sign: float) -> tuple[np.ndarray, float]:
        raw += self.ft_weight[:, feature_idx] * sign
        psqt += float(self.psqt_weight[feature_idx]) * sign
        return raw, psqt

    def _update_side_after_move(
        self,
        raw: np.ndarray,
        psqt: float,
        bucket: int,
        *,
        perspective: int,
        board,
        move,
        piece: int,
        captured: int,
    ) -> tuple[int, np.ndarray, float]:
        current_king = board.king_pos.get(perspective, -1)
        if current_king < 0:
            empty_raw, empty_psqt = self._empty_raw_accumulator()
            return -1, empty_raw, empty_psqt

        # 该视角自己的王移动/被吃时，整个桶会变化，直接重建这一侧的 accumulator。
        if piece == perspective or captured == perspective or bucket < 0:
            return self._build_side_raw(board, perspective)

        updated_raw = raw.copy()
        updated_psqt = float(psqt)

        # 任意非王棋子移动，都会影响双方视角的特征集合。
        if piece != 0 and abs(piece) != 1:
            old_idx = feature_index(bucket, int(piece), int(move.from_pos), perspective)
            new_idx = feature_index(bucket, int(piece), int(move.to_pos), perspective)
            updated_raw, updated_psqt = self._apply_feature_delta(updated_raw, updated_psqt, old_idx, -1.0)
            updated_raw, updated_psqt = self._apply_feature_delta(updated_raw, updated_psqt, new_idx, +1.0)

        # 被吃掉的非王棋子要从特征里移除。
        if captured != 0 and abs(captured) != 1:
            cap_idx = feature_index(bucket, int(captured), int(move.to_pos), perspective)
            updated_raw, updated_psqt = self._apply_feature_delta(updated_raw, updated_psqt, cap_idx, -1.0)

        return bucket, updated_raw, updated_psqt

    def update_accumulator(self, parent_state: AccumulatorState, board) -> AccumulatorState:
        if not board.history:
            return self.build_accumulator(board)

        move, piece, captured, _prev_side, _red_king, _black_king, prev_hash, _post_hash = board.history[-1]
        if parent_state.key is not None and parent_state.key != prev_hash:
            return self.build_accumulator(board)

        red_bucket, red_raw, red_psqt = self._update_side_after_move(
            parent_state.red_raw,
            parent_state.red_psqt,
            parent_state.red_bucket,
            perspective=1,
            board=board,
            move=move,
            piece=piece,
            captured=captured,
        )
        black_bucket, black_raw, black_psqt = self._update_side_after_move(
            parent_state.black_raw,
            parent_state.black_psqt,
            parent_state.black_bucket,
            perspective=-1,
            board=board,
            move=move,
            piece=piece,
            captured=captured,
        )
        return AccumulatorState(
            red_bucket=red_bucket,
            black_bucket=black_bucket,
            red_raw=red_raw,
            black_raw=black_raw,
            red_psqt=red_psqt,
            black_psqt=black_psqt,
            key=board.zhash,
        )

    @staticmethod
    def _clip_acc(raw: np.ndarray) -> np.ndarray:
        return np.clip(raw, 0.0, 1.0)

    def forward_raw(
        self,
        red_raw: np.ndarray,
        black_raw: np.ndarray,
        red_psqt: float,
        black_psqt: float,
        side_to_move: int,
    ) -> float:
        red_acc = self._clip_acc(red_raw)
        black_acc = self._clip_acc(black_raw)

        if side_to_move > 0:
            hidden_input = np.concatenate((red_acc, black_acc)).astype(np.float32, copy=False)
            psqt = red_psqt - black_psqt
        else:
            hidden_input = np.concatenate((black_acc, red_acc)).astype(np.float32, copy=False)
            psqt = black_psqt - red_psqt

        hidden = np.clip(self.l1_weight @ hidden_input + self.l1_bias, 0.0, 1.0)
        out = float(self.l2_weight.reshape(-1) @ hidden) + self.l2_bias
        return float(psqt + out)

    def forward_indices(self, red_features: np.ndarray, black_features: np.ndarray, side_to_move: int) -> float:
        red_raw, red_psqt = self._accumulate_raw(red_features)
        black_raw, black_psqt = self._accumulate_raw(black_features)
        return self.forward_raw(red_raw, black_raw, red_psqt, black_psqt, side_to_move)

    def forward_accumulator(self, state: AccumulatorState, side_to_move: int) -> float:
        return self.forward_raw(
            state.red_raw,
            state.black_raw,
            state.red_psqt,
            state.black_psqt,
            side_to_move,
        )

    def evaluate(self, board) -> int:
        state = self.build_accumulator(board)
        return int(round(self.forward_accumulator(state, board.side)))

    def evaluate_accumulator(self, state: AccumulatorState, side_to_move: int) -> int:
        return int(round(self.forward_accumulator(state, side_to_move)))
