# ai_heuristic.py

import random

from legal_moves import generate_legal_moves
from rules import is_in_check

PIECE_VALUE = {
    1: 10000,  # 帅/将
    2: 900,    # 车
    3: 450,    # 马
    4: 500,    # 炮
    5: 220,    # 相/象
    6: 220,    # 仕/士
    7: 120,    # 兵/卒
}

INF = 10**9


def piece_value(piece):
    return PIECE_VALUE.get(abs(piece), 0)


def material_score(board):
    score = 0
    for piece in board.board:
        if piece == 0:
            continue

        value = piece_value(piece)
        if (piece > 0) == (board.side > 0):
            score += value
        else:
            score -= value

    return score


def pos_bonus(piece, pos):
    x = pos % 9
    y = pos // 9
    side = 1 if piece > 0 else -1
    p = abs(piece)

    score = 0

    center_dist = abs(x - 4)
    score += max(0, 4 - center_dist) * 6

    if p == 7:
        crossed = (y < 5) if side > 0 else (y > 4)
        if crossed:
            score += 50

        if side > 0:
            score += (9 - y) * 4
        else:
            score += y * 4

    elif p in (2, 3, 4):
        if side > 0:
            score += (9 - y) * 2
        else:
            score += y * 2

    return score


def positional_score(board):
    score = 0
    for pos, piece in enumerate(board.board):
        if piece == 0:
            continue

        bonus = pos_bonus(piece, pos)
        if (piece > 0) == (board.side > 0):
            score += bonus
        else:
            score -= bonus

    return score


def mobility_score(board):
    my_moves = generate_legal_moves(board)
    return len(my_moves) * 2


def legal_moves_for_side(board, side):
    original_side = board.side
    board.side = side
    moves = generate_legal_moves(board)
    board.side = original_side
    return moves


def count_attackers(board, target_pos, side):
    moves = legal_moves_for_side(board, side)
    count = 0
    for mv in moves:
        if mv.to_pos == target_pos:
            count += 1
    return count


def moved_piece_in_danger(board, moved_to_pos):
    enemy_moves = generate_legal_moves(board)
    for mv in enemy_moves:
        if mv.to_pos == moved_to_pos:
            return True
    return False


def rooted_capture_adjustment(board, move):
    """
    专门处理“吃有根子”的修正分。
    这里假设还没 make_move。
    """
    piece = board.board[move.from_pos]
    captured = board.board[move.to_pos]

    if captured == 0:
        return 0

    my_side = board.side
    enemy_side = -my_side

    attacker_value = piece_value(piece)
    captured_value = piece_value(captured)

    # 吃之前先看目标子有没有根
    enemy_defenders_before = count_attackers(board, move.to_pos, enemy_side)

    board.make_move(move)

    # 吃过去之后，这个点敌方有多少人能反吃，我方有多少人能保护
    enemy_attackers_after = count_attackers(board, move.to_pos, board.side)
    my_defenders_after = count_attackers(board, move.to_pos, -board.side)

    board.undo_move()

    score = 0

    # 无根子，鼓励吃
    if enemy_defenders_before == 0 and enemy_attackers_after == 0:
        score += captured_value * 3
        return score

    # 有根，但我方也有保护：谨慎处理
    if enemy_attackers_after > 0 and my_defenders_after > 0:
        # 低价值子换高价值子通常还能接受
        if attacker_value <= captured_value:
            score += (captured_value - attacker_value) * 2 + 20
        else:
            score -= (attacker_value - captured_value) * 2
        return score

    # 对方能反吃，而我没保护：重罚
    if enemy_attackers_after > 0 and my_defenders_after == 0:
        # 如果是小吃大，还是允许一点
        if attacker_value < captured_value:
            score += (captured_value - attacker_value) - 20
        else:
            score -= (attacker_value - captured_value) * 5 + captured_value * 2
        return score

    # 默认轻微奖励
    score += captured_value
    return score


def static_eval(board):
    score = 0
    score += material_score(board)
    score += positional_score(board)
    score += mobility_score(board)

    if is_in_check(board, board.side):
        score -= 120

    return score


def move_order_score(board, move):
    piece = board.board[move.from_pos]
    captured = board.board[move.to_pos]

    score = 0

    if captured != 0:
        score += piece_value(captured) * 20
        score -= piece_value(piece)

    score += rooted_capture_adjustment(board, move)

    board.make_move(move)

    if is_in_check(board, board.side):
        score += 150

    moved_piece = board.board[move.to_pos]
    if moved_piece_in_danger(board, move.to_pos):
        score -= piece_value(moved_piece) * 3

    board.undo_move()

    score += pos_bonus(piece, move.to_pos)
    return score


def ordered_moves(board):
    moves = generate_legal_moves(board)
    moves.sort(key=lambda mv: move_order_score(board, mv), reverse=True)
    return moves


def negamax(board, depth, alpha, beta):
    moves = generate_legal_moves(board)

    if not moves:
        return -INF + 1

    if depth == 0:
        return static_eval(board)

    best = -INF

    for move in ordered_moves(board):
        board.make_move(move)
        score = -negamax(board, depth - 1, -beta, -alpha)
        board.undo_move()

        if score > best:
            best = score

        if best > alpha:
            alpha = best

        if alpha >= beta:
            break

    return best


def choose_heuristic_move(board, depth=2):
    moves = generate_legal_moves(board)
    if not moves:
        return None

    best_score = -INF
    best_moves = []

    for move in ordered_moves(board):
        board.make_move(move)
        score = -negamax(board, depth - 1, -INF, INF)
        board.undo_move()

        if score > best_score:
            best_score = score
            best_moves = [move]
        elif score == best_score:
            best_moves.append(move)

    return random.choice(best_moves)