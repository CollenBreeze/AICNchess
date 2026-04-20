from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np

BOARD_HEIGHT = 10
BOARD_WIDTH = 9
NUM_PIECE_PLANES = 14
SIDE_TO_MOVE_CHANNEL = 14
REPETITION_CHANNEL = 15
NUM_CHANNELS = 16
MAX_REPEAT_BUCKET = 3


def _piece_plane(piece: int) -> int:
    piece_type = abs(piece) - 1
    if piece > 0:
        return piece_type
    return 7 + piece_type


def repeat_bucket(repeat_count: int) -> int:
    return max(1, min(MAX_REPEAT_BUCKET, int(repeat_count)))


def snapshot_board(board) -> Tuple[np.ndarray, int, int]:
    board_array = np.asarray(board.board, dtype=np.int8).copy()
    side = int(board.side)
    repeat_count = repeat_bucket(board.get_repeat_count())
    return board_array, side, repeat_count


def encode_snapshot(board_array: Sequence[int], side: int, repeat_count: int = 1) -> np.ndarray:
    planes = np.zeros((NUM_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float32)

    for pos, piece in enumerate(board_array):
        piece = int(piece)
        if piece == 0:
            continue
        x = pos % BOARD_WIDTH
        y = pos // BOARD_WIDTH
        planes[_piece_plane(piece), y, x] = 1.0

    if side > 0:
        planes[SIDE_TO_MOVE_CHANNEL, :, :] = 1.0

    bucket = repeat_bucket(repeat_count)
    if bucket > 1:
        planes[REPETITION_CHANNEL, :, :] = (bucket - 1) / (MAX_REPEAT_BUCKET - 1)

    return planes


def encode_board(board) -> np.ndarray:
    return encode_snapshot(*snapshot_board(board))


def batch_encode_snapshots(snapshots: Iterable[Tuple[np.ndarray, int, int]]) -> np.ndarray:
    encoded = [encode_snapshot(board_array, side, repeat_count) for board_array, side, repeat_count in snapshots]
    return np.stack(encoded, axis=0).astype(np.float32, copy=False)
