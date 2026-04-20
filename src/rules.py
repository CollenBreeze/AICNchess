# rules.py

from bitboards import (
    ROOK_RAYS,
    KNIGHT_ATTACKERS,
    PAWN_ATTACKERS_RED,
    PAWN_ATTACKERS_BLACK,
)

ORTHO_DIRECTIONS = ("up", "down", "left", "right")


def _line_attacked(board_array, target_pos, enemy_side):
    for direction in ORTHO_DIRECTIONS:
        first_blocker_seen = False

        for sq in ROOK_RAYS[direction][target_pos]:
            piece = board_array[sq]
            if piece == 0:
                continue

            if not first_blocker_seen:
                if (piece > 0) == (enemy_side > 0):
                    abs_piece = abs(piece)
                    if abs_piece == 2 or abs_piece == 1:
                        return True
                first_blocker_seen = True
            else:
                if (piece > 0) == (enemy_side > 0) and abs(piece) == 4:
                    return True
                break

    return False


def _knight_attacked(board_array, target_pos, enemy_side):
    for from_sq, leg_sq in KNIGHT_ATTACKERS[target_pos]:
        piece = board_array[from_sq]
        if piece == 0 or (piece > 0) != (enemy_side > 0):
            continue

        if abs(piece) == 3 and board_array[leg_sq] == 0:
            return True

    return False


def _pawn_attacked(board_array, target_pos, enemy_side):
    attackers = PAWN_ATTACKERS_RED[target_pos] if enemy_side > 0 else PAWN_ATTACKERS_BLACK[target_pos]
    enemy_pawn = 7 if enemy_side > 0 else -7

    for from_sq in attackers:
        if board_array[from_sq] == enemy_pawn:
            return True

    return False


def is_square_attacked(board, target_pos, by_side):
    b = board.board
    return (
        _line_attacked(b, target_pos, by_side)
        or _knight_attacked(b, target_pos, by_side)
        or _pawn_attacked(b, target_pos, by_side)
    )


def is_in_check(board, side):
    king_pos = board.king_pos.get(side, -1)
    if king_pos == -1:
        return False
    return is_square_attacked(board, king_pos, -side)
