# legal_moves.py

from board import DRAW_REPEAT_COUNT
from generator import generate_all_moves
from rules import is_in_check


LONG_CHECK_REPEAT = DRAW_REPEAT_COUNT


def is_checkmate(board):
    if not is_in_check(board, board.side):
        return False

    moves = generate_legal_moves(board)
    return len(moves) == 0


def is_stalemate(board):
    if is_in_check(board, board.side):
        return False

    moves = generate_legal_moves(board)
    return len(moves) == 0


def generate_legal_moves(board, captures_only=False):
    if board.king_pos.get(board.side, -1) == -1:
        return []

    pseudo_moves = generate_all_moves(board, captures_only=captures_only)
    legal_moves = []
    moving_side = board.side

    for move in pseudo_moves:
        board.make_move(move)

        legal = False

        # 1. 不能让自己被将
        if not is_in_check(board, moving_side):
            repeat_count = board.get_repeat_count()

            # 2. 没有重复到阈值，直接合法
            if repeat_count < LONG_CHECK_REPEAT:
                legal = True
            else:
                # 3. 重复局面里，如果这步形成“对对方将军”，判作长将违例
                # board.side 此时已经切到对方，因此判断对方是否被将
                gives_check = is_in_check(board, board.side)
                if not gives_check:
                    legal = True

        if legal:
            legal_moves.append(move)

        board.undo_move()

    return legal_moves


def generate_legal_captures(board):
    return generate_legal_moves(board, captures_only=True)
