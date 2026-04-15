# legal_moves.py

from generator import generate_all_moves
from rules import is_in_check

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

def generate_legal_moves(board):
    moves = generate_all_moves(board)

    legal_moves = []

    side = board.side

    for move in moves:
        board.make_move(move)

        # 判断自己是否被将军
        if not is_in_check(board, side):
            legal_moves.append(move)

        board.undo_move()

    return legal_moves