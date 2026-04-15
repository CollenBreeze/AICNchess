# play.py

import os
from datetime import datetime

import pygame

from board import Board
from renderer import Renderer
from legal_moves import generate_legal_moves
from rules import is_in_check
from notation import move_to_notation

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
QIPU_DIR = os.path.join(BASE_DIR, "Qipu")


def ensure_qipu_dir():
    os.makedirs(QIPU_DIR, exist_ok=True)


def get_turn_text(side):
    return "红方走棋" if side > 0 else "黑方走棋"


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


def undo_last_move(board, notations):
    if not board.undo_move():
        return False, "没有可悔的棋。"

    removed_notation = notations.pop() if notations else ""
    if removed_notation:
        return True, f"已悔棋：{removed_notation}"
    return True, "已悔棋。"


def play():
    board = Board()
    board.init_startpos()

    renderer = Renderer()

    selected = None
    possible_moves = []
    running = True
    game_over = False
    notations = []
    status_text = get_turn_text(board.side)

    while running:
        renderer.draw(board, selected, possible_moves, status_text, game_over)

        moves = [] if game_over else generate_legal_moves(board)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos

                if renderer.is_undo_button_clicked(mouse_pos):
                    success, message = undo_last_move(board, notations)
                    status_text = message
                    if success:
                        game_over = False
                    selected = None
                    possible_moves = []
                    print(message)
                    continue

                if renderer.is_save_button_clicked(mouse_pos):
                    saved_path = save_notations(notations)
                    status_text = f"已保存：{os.path.basename(saved_path)}"
                    selected = None
                    possible_moves = []
                    print(f"棋谱已保存到：{saved_path}")
                    continue

                if game_over:
                    continue

                pos = renderer.get_click_pos(mouse_pos)
                if pos is None:
                    continue

                piece = board.board[pos]

                if selected is None:
                    if piece != 0 and (piece > 0) == (board.side > 0):
                        selected = pos
                        possible_moves = [
                            m.to_pos for m in moves if m.from_pos == pos
                        ]
                else:
                    move_found = None

                    for m in moves:
                        if m.from_pos == selected and m.to_pos == pos:
                            move_found = m
                            break

                    if move_found:
                        notation = move_to_notation(board, move_found)
                        print(notation)
                        notations.append(notation)

                        moved_side = board.side
                        board.make_move(move_found)

                        if is_in_check(board, board.side):
                            status_text = "将军！"
                            print("将军！")
                        else:
                            status_text = get_turn_text(board.side)

                        opponent_moves = generate_legal_moves(board)
                        if len(opponent_moves) == 0:
                            winner = "红方" if moved_side > 0 else "黑方"
                            status_text = f"{winner}胜！"
                            print(status_text)
                            saved_path = save_notations(notations)
                            print(f"棋谱已保存到：{saved_path}")
                            game_over = True

                    else:
                        if piece != 0 and (piece > 0) == (board.side > 0):
                            selected = pos
                            possible_moves = [
                                m.to_pos for m in moves if m.from_pos == pos
                            ]
                            continue

                    selected = None
                    possible_moves = []

    pygame.quit()


if __name__ == "__main__":
    play()