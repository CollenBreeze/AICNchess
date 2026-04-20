# zobrist.py

import random

BOARD_SIZE = 90

PIECES = (-7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7)

_rng = random.Random(20260417)
PIECE_KEYS = {
    piece: tuple(_rng.getrandbits(64) for _ in range(BOARD_SIZE))
    for piece in PIECES
}
SIDE_KEY = _rng.getrandbits(64)


def piece_key(piece, pos):
    return PIECE_KEYS[piece][pos]


def compute_hash(board_array, side):
    h = 0
    for pos, piece in enumerate(board_array):
        if piece != 0:
            h ^= PIECE_KEYS[piece][pos]
    if side < 0:
        h ^= SIDE_KEY
    return h
