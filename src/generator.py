# generator.py

from move import Move
from bitboards import (
    ensure_precomputed,
    ROOK_RAYS,
    KNIGHT_STEPS,
    ELEPHANT_STEPS_RED,
    ELEPHANT_STEPS_BLACK,
    ADVISOR_MOVES_RED,
    ADVISOR_MOVES_BLACK,
    KING_MOVES_RED,
    KING_MOVES_BLACK,
    PAWN_MOVES_RED,
    PAWN_MOVES_BLACK,
)

ORTHO_DIRECTIONS = ("up", "down", "left", "right")
VERTICAL_DIRECTIONS = ("up", "down")


# ======================
# 主入口
# ======================

def generate_all_moves(board, captures_only=False):
    ensure_precomputed()

    moves = []
    b = board.board
    side = board.side

    for pos in board.piece_positions[side]:
        piece = b[pos]
        abs_piece = abs(piece)

        if abs_piece == 2:
            gen_rook(board, pos, moves, captures_only)
        elif abs_piece == 3:
            gen_knight(board, pos, moves, captures_only)
        elif abs_piece == 4:
            gen_cannon(board, pos, moves, captures_only)
        elif abs_piece == 5:
            gen_elephant(board, pos, moves, captures_only)
        elif abs_piece == 6:
            gen_advisor(board, pos, moves, captures_only)
        elif abs_piece == 1:
            gen_king(board, pos, moves, captures_only)
        elif abs_piece == 7:
            gen_pawn(board, pos, moves, captures_only)

    return moves


def generate_moves(board):
    return generate_all_moves(board)


# ======================
# 车
# ======================

def gen_rook(board, pos, moves, captures_only=False):
    b = board.board
    side = b[pos] > 0

    for direction in ORTHO_DIRECTIONS:
        for sq in ROOK_RAYS[direction][pos]:
            piece = b[sq]

            if piece == 0:
                if not captures_only:
                    moves.append(Move(pos, sq))
                continue

            if (piece > 0) != side:
                moves.append(Move(pos, sq))
            break


# ======================
# 炮
# ======================

def gen_cannon(board, pos, moves, captures_only=False):
    b = board.board
    side = b[pos] > 0

    for direction in ORTHO_DIRECTIONS:
        jumped = False

        for sq in ROOK_RAYS[direction][pos]:
            piece = b[sq]

            if not jumped:
                if piece == 0:
                    if not captures_only:
                        moves.append(Move(pos, sq))
                else:
                    jumped = True
            else:
                if piece != 0:
                    if (piece > 0) != side:
                        moves.append(Move(pos, sq))
                    break


# ======================
# 马
# ======================

def gen_knight(board, pos, moves, captures_only=False):
    b = board.board
    side = b[pos] > 0

    for to_sq, leg_sq in KNIGHT_STEPS[pos]:
        if b[leg_sq] != 0:
            continue

        piece = b[to_sq]

        if piece == 0:
            if not captures_only:
                moves.append(Move(pos, to_sq))
        elif (piece > 0) != side:
            moves.append(Move(pos, to_sq))


# ======================
# 象
# ======================

def gen_elephant(board, pos, moves, captures_only=False):
    b = board.board
    piece = b[pos]
    side = piece > 0
    steps = ELEPHANT_STEPS_RED[pos] if side else ELEPHANT_STEPS_BLACK[pos]

    for to_sq, eye_sq in steps:
        if b[eye_sq] != 0:
            continue

        target = b[to_sq]
        if target == 0:
            if not captures_only:
                moves.append(Move(pos, to_sq))
        elif (target > 0) != side:
            moves.append(Move(pos, to_sq))


# ======================
# 士
# ======================

def gen_advisor(board, pos, moves, captures_only=False):
    b = board.board
    side = b[pos] > 0
    targets = ADVISOR_MOVES_RED[pos] if side else ADVISOR_MOVES_BLACK[pos]

    for to_sq in targets:
        piece = b[to_sq]

        if piece == 0:
            if not captures_only:
                moves.append(Move(pos, to_sq))
        elif (piece > 0) != side:
            moves.append(Move(pos, to_sq))


# ======================
# 将 / 帅
# ======================

def gen_king(board, pos, moves, captures_only=False):
    b = board.board
    side = b[pos] > 0
    targets = KING_MOVES_RED[pos] if side else KING_MOVES_BLACK[pos]

    for to_sq in targets:
        piece = b[to_sq]

        if piece == 0:
            if not captures_only:
                moves.append(Move(pos, to_sq))
        elif (piece > 0) != side:
            moves.append(Move(pos, to_sq))

    # 飞将：同一路无阻隔时可直接吃对方将/帅
    for direction in VERTICAL_DIRECTIONS:
        for sq in ROOK_RAYS[direction][pos]:
            piece = b[sq]
            if piece == 0:
                continue

            if abs(piece) == 1 and (piece > 0) != side:
                moves.append(Move(pos, sq))
            break


# ======================
# 兵 / 卒
# ======================

def gen_pawn(board, pos, moves, captures_only=False):
    b = board.board
    piece = b[pos]
    side = piece > 0
    targets = PAWN_MOVES_RED[pos] if side else PAWN_MOVES_BLACK[pos]

    for to_sq in targets:
        target_piece = b[to_sq]

        if target_piece == 0:
            if not captures_only:
                moves.append(Move(pos, to_sq))
        elif (target_piece > 0) != side:
            moves.append(Move(pos, to_sq))
