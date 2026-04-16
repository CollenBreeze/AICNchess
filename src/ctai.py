# ctai.py

import os
from datetime import datetime

import pygame

from board import Board
from renderer import Renderer
from legal_moves import generate_legal_moves
from rules import is_in_check
from notation import move_to_notation
from ai_heuristic import choose_heuristic_move

AI_SIDE = -1   # 黑方 AI
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
QIPU_DIR = os.path.join(BASE_DIR, "Qipu")


def ensure_qipu_dir():
    os.makedirs(QIPU_DIR, exist_ok=True)


def build_qipu_text(notations):
    lines = []

    for i in range(0, len(notations), 2):
        step_no = i // 2 + 1
        red_move = notations[i]
        black_move = notations[i + 1] if i + 1 < len(notations) else ""
        lines.append(f"{step_no}. {red_move} {black_move}".rstrip())

    return "\n".join(lines)


def save_notations(notations):
    ensure_qipu_dir()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"qipu_{timestamp}.txt"
    file_path = os.path.join(QIPU_DIR, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(build_qipu_text(notations))

    return file_path


def get_turn_text(side):
    return "红方走棋" if side > 0 else "黑方走棋"


def get_winner_text(side):
    return "红方胜！" if side > 0 else "黑方胜！"


def undo_last_round(board, notations):
    """
    人机模式下一次悔棋：同时悔玩家和 AI 的一步。
    最多撤销两步；如果当前只有一步可撤销，就撤销一步。
    """
    undone = 0

    for _ in range(2):
        if board.undo_move():
            undone += 1
            if notations:
                notations.pop()
        else:
            break

    if undone == 0:
        return False, "没有可悔的棋。"
    if undone == 1:
        return True, "已悔 1 步。"
    return True, "已悔一个回合。"


def apply_move(board, move, notations, mover_name=None):
    """
    执行一步棋，返回：
    status_text, game_over
    """
    notation = move_to_notation(board, move)
    if mover_name:
        print(f"{mover_name}: {notation}")
    else:
        print(notation)

    notations.append(notation)

    moved_side = board.side
    board.make_move(move)

    if is_in_check(board, board.side):
        print("将军！")
        status_text = "将军！"
    else:
        status_text = get_turn_text(board.side)

    opponent_moves = generate_legal_moves(board)
    if len(opponent_moves) == 0:
        status_text = get_winner_text(moved_side)
        print(status_text)
        saved_path = save_notations(notations)
        print(f"棋谱已保存到：{saved_path}")
        return status_text, True

    return status_text, False


def ctai():
    board = Board()
    board.init_startpos()

    renderer = Renderer()

    selected = None
    possible_moves = []
    running = True
    game_over = False
    notations = []
    status_text = get_turn_text(board.side)

    # 用来保证：玩家走完后先渲染一帧，再轮到 AI 算
    ai_pending_frames = 0

    clock = pygame.time.Clock()

    while running:
        renderer.draw(board, selected, possible_moves, status_text, game_over)

        moves = [] if game_over else generate_legal_moves(board)

        # 无棋可走判负
        if not game_over and len(moves) == 0:
            status_text = get_winner_text(-board.side)
            print(status_text)
            saved_path = save_notations(notations)
            print(f"棋谱已保存到：{saved_path}")
            game_over = True

        # ======================
        # 统一处理事件
        # ======================
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos

                # 悔棋按钮
                if renderer.is_undo_button_clicked(mouse_pos):
                    success, message = undo_last_round(board, notations)
                    status_text = message
                    if success:
                        game_over = False
                        ai_pending_frames = 0
                    selected = None
                    possible_moves = []
                    print(message)
                    continue

                # 保存棋谱按钮
                if hasattr(renderer, "is_save_button_clicked") and renderer.is_save_button_clicked(mouse_pos):
                    saved_path = save_notations(notations)
                    status_text = f"已保存：{os.path.basename(saved_path)}"
                    selected = None
                    possible_moves = []
                    print(f"棋谱已保存到：{saved_path}")
                    continue

                # 终局后不再响应棋盘点击
                if game_over:
                    continue

                # AI 回合不允许玩家点棋盘
                if board.side == AI_SIDE:
                    continue

                pos = renderer.get_click_pos(mouse_pos)
                if pos is None:
                    continue

                piece = board.board[pos]

                if selected is None:
                    if piece != 0 and (piece > 0) == (board.side > 0):
                        selected = pos
                        possible_moves = [m.to_pos for m in moves if m.from_pos == pos]
                else:
                    move_found = None

                    for m in moves:
                        if m.from_pos == selected and m.to_pos == pos:
                            move_found = m
                            break

                    if move_found:
                        status_text, game_over = apply_move(board, move_found, notations)

                        selected = None
                        possible_moves = []

                        # 玩家走完后，先显示当前局面一帧，再让 AI 走
                        if not game_over and board.side == AI_SIDE:
                            ai_pending_frames = 1

                    else:
                        if piece != 0 and (piece > 0) == (board.side > 0):
                            selected = pos
                            possible_moves = [m.to_pos for m in moves if m.from_pos == pos]
                            continue

                        selected = None
                        possible_moves = []

        if not running:
            break

        # ======================
        # AI 回合
        # 先等一帧，确保玩家刚落子的画面先显示出来
        # ======================
        if not game_over and board.side == AI_SIDE:
            if ai_pending_frames > 0:
                ai_pending_frames -= 1
            else:
                ai_move = choose_heuristic_move(board, depth=3)

                if ai_move is None:
                    status_text = get_winner_text(-AI_SIDE)
                    print(status_text)
                    saved_path = save_notations(notations)
                    print(f"棋谱已保存到：{saved_path}")
                    game_over = True
                else:
                    status_text, game_over = apply_move(
                        board, ai_move, notations, mover_name="AI"
                    )
                    selected = None
                    possible_moves = []

        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    ctai()