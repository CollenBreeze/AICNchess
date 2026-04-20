# play.py

import argparse
import os
import subprocess
import sys
from datetime import datetime

import pygame

from ai_selector import (
    AI_MODE_DDQN,
    AI_MODE_NNUE,
    AI_MODE_SEARCH,
    ai_mode_label,
    choose_ai_move,
    cycle_ai_mode,
    ddqn_unavailable_reason,
    nnue_unavailable_reason,
    normalize_ai_mode,
)
from board import Board
from legal_moves import generate_legal_moves
from notation import move_to_notation
from renderer import Renderer
from rules import is_in_check

AI_PRESET = "balanced"
AI_PARAMS_FILE = None  # 例如："/path/to/params_A_xxx.json"
AI_SEARCH_DEPTH = 4
MIN_SEARCH_DEPTH = 1
MAX_SEARCH_DEPTH = 8
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.dirname(__file__)
QIPU_DIR = os.path.join(BASE_DIR, "Qipu")
DEFAULT_NNUE_WEIGHTS = os.path.join(BASE_DIR, "checkpoints_nnue", "xiangqi_nnue_v2.npz")
DEFAULT_DDQN_WEIGHTS = os.path.join(BASE_DIR, "checkpoints_ddqn", "ddqn_final.pt")
NNUE_TRAIN_GUI_SCRIPT = os.path.join(SRC_DIR, "train_nnue_gui.py")



def clamp_search_depth(depth):
    try:
        value = int(depth)
    except (TypeError, ValueError):
        value = AI_SEARCH_DEPTH
    return max(MIN_SEARCH_DEPTH, min(MAX_SEARCH_DEPTH, value))



def depth_status_text(depth):
    return f"AI 搜索深度：{depth}"



def ai_mode_status_text(ai_mode):
    return f"AI 模式：{ai_mode_label(ai_mode)}"



def unavailable_reason_for_mode(ai_mode, nnue_weights_path, ddqn_weights_path):
    ai_mode = normalize_ai_mode(ai_mode)
    if ai_mode == AI_MODE_NNUE:
        return nnue_unavailable_reason(nnue_weights_path)
    if ai_mode == AI_MODE_DDQN:
        return ddqn_unavailable_reason(ddqn_weights_path)
    return None



def ensure_qipu_dir():
    os.makedirs(QIPU_DIR, exist_ok=True)



def get_turn_text(side):
    return "红方走棋" if side > 0 else "黑方走棋"



def get_winner_text(side):
    return "红方胜！" if side > 0 else "黑方胜！"



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



def side_is_ai(side, red_ai_enabled, black_ai_enabled):
    if side > 0:
        return red_ai_enabled
    return black_ai_enabled



def _update_depth_from_key(current_depth, event):
    numeric_map = {
        pygame.K_1: 1,
        pygame.K_2: 2,
        pygame.K_3: 3,
        pygame.K_4: 4,
        pygame.K_5: 5,
        pygame.K_6: 6,
        pygame.K_7: 7,
        pygame.K_8: 8,
        pygame.K_KP1: 1,
        pygame.K_KP2: 2,
        pygame.K_KP3: 3,
        pygame.K_KP4: 4,
        pygame.K_KP5: 5,
        pygame.K_KP6: 6,
        pygame.K_KP7: 7,
        pygame.K_KP8: 8,
    }

    if event.key in numeric_map:
        return numeric_map[event.key]

    if event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_RIGHTBRACKET):
        return clamp_search_depth(current_depth + 1)

    if event.key in (pygame.K_MINUS, pygame.K_KP_MINUS, pygame.K_LEFTBRACKET):
        return clamp_search_depth(current_depth - 1)

    return current_depth



def apply_move(board, move, notations, renderer=None, mover_name=None):
    notation = move_to_notation(board, move)
    if mover_name:
        print(f"{mover_name}: {notation}")
    else:
        print(notation)

    notations.append(notation)

    moved_side = board.side
    captured_piece = board.board[move.to_pos]
    board.make_move(move)

    if renderer is not None:
        if captured_piece != 0:
            renderer.play_sound("capture")
        else:
            renderer.play_sound("place")

    in_check = is_in_check(board, board.side)
    if in_check:
        status_text = "将军！"
    else:
        status_text = get_turn_text(board.side)

    opponent_moves = generate_legal_moves(board)
    if len(opponent_moves) == 0:
        status_text = get_winner_text(moved_side)
        print(status_text)
        if renderer is not None:
            renderer.play_sound("checkmate")
        saved_path = save_notations(notations)
        print(f"棋谱已保存到：{saved_path}")
        return status_text, True

    if in_check:
        print("将军！")
        if renderer is not None:
            renderer.play_sound("check")

    return status_text, False



def choose_current_ai_move(board, ai_mode, search_depth, nnue_weights_path, ddqn_weights_path, ddqn_device):
    return choose_ai_move(
        board,
        ai_mode=ai_mode,
        search_depth=search_depth,
        search_preset=AI_PRESET,
        search_params_path=AI_PARAMS_FILE,
        nnue_weights_path=nnue_weights_path,
        ddqn_weights_path=ddqn_weights_path,
        ddqn_device=ddqn_device,
    )



def try_switch_ai_mode(current_mode, nnue_weights_path, ddqn_weights_path):
    next_mode = cycle_ai_mode(
        current_mode,
        nnue_weights_path=nnue_weights_path,
        ddqn_weights_path=ddqn_weights_path,
    )
    return next_mode, ai_mode_status_text(next_mode)



def launch_nnue_training_gui(nnue_weights_path):
    if not os.path.isfile(NNUE_TRAIN_GUI_SCRIPT):
        return False, f"未找到训练界面脚本：{os.path.basename(NNUE_TRAIN_GUI_SCRIPT)}"

    output_path = nnue_weights_path or DEFAULT_NNUE_WEIGHTS
    cmd = [sys.executable, NNUE_TRAIN_GUI_SCRIPT, "--output", output_path]
    if nnue_weights_path and os.path.isfile(nnue_weights_path):
        cmd.extend(["--init-weights", nnue_weights_path])

    try:
        subprocess.Popen(cmd, cwd=BASE_DIR)
    except Exception as exc:
        return False, f"打开 NNUE 训练界面失败：{exc}"

    return True, "已打开 NNUE 训练界面"



def play(
    search_depth=AI_SEARCH_DEPTH,
    allow_depth_hotkeys=True,
    ai_mode=AI_MODE_SEARCH,
    nnue_weights_path=DEFAULT_NNUE_WEIGHTS,
    ddqn_weights_path=DEFAULT_DDQN_WEIGHTS,
    ddqn_device=None,
):
    board = Board()
    board.init_startpos()

    renderer = Renderer()

    selected = None
    possible_moves = []
    running = True
    game_over = False
    notations = []
    status_text = get_turn_text(board.side)
    search_depth = clamp_search_depth(search_depth)
    ai_mode = normalize_ai_mode(ai_mode)

    initial_reason = unavailable_reason_for_mode(ai_mode, nnue_weights_path, ddqn_weights_path)
    if initial_reason is not None:
        status_text = f"{initial_reason}，已切回搜索AI"
        print(status_text)
        ai_mode = AI_MODE_SEARCH

    red_ai_enabled = False
    black_ai_enabled = False
    ai_once_pending = False
    ai_pending_frames = 0

    clock = pygame.time.Clock()

    while running:
        renderer.draw(
            board,
            selected,
            possible_moves,
            status_text,
            game_over,
            search_depth=search_depth,
            red_ai_enabled=red_ai_enabled,
            black_ai_enabled=black_ai_enabled,
            ai_mode=ai_mode,
            show_depth_controls=True,
            show_ai_controls=True,
            show_ai_mode_button=True,
            show_training_button=True,
        )

        moves = [] if game_over else generate_legal_moves(board)

        if not game_over and len(moves) == 0:
            status_text = get_winner_text(-board.side)
            print(status_text)
            renderer.play_sound("checkmate")
            saved_path = save_notations(notations)
            print(f"棋谱已保存到：{saved_path}")
            game_over = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN and allow_depth_hotkeys:
                new_depth = _update_depth_from_key(search_depth, event)
                if new_depth != search_depth:
                    search_depth = new_depth
                    status_text = depth_status_text(search_depth)
                    renderer.play_sound("pickup")
                    print(status_text)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos

                if renderer.is_undo_button_clicked(mouse_pos):
                    success, message = undo_last_move(board, notations)
                    status_text = message
                    if success:
                        game_over = False
                        ai_once_pending = False
                        ai_pending_frames = 0
                        renderer.play_sound("undo")
                    selected = None
                    possible_moves = []
                    print(message)
                    continue

                if renderer.is_save_button_clicked(mouse_pos):
                    saved_path = save_notations(notations)
                    status_text = f"已保存：{os.path.basename(saved_path)}"
                    selected = None
                    possible_moves = []
                    renderer.play_sound("place")
                    print(f"棋谱已保存到：{saved_path}")
                    continue

                if renderer.is_depth_minus_clicked(mouse_pos):
                    new_depth = clamp_search_depth(search_depth - 1)
                    if new_depth != search_depth:
                        search_depth = new_depth
                        status_text = depth_status_text(search_depth)
                        renderer.play_sound("pickup")
                        print(status_text)
                    continue

                if renderer.is_depth_plus_clicked(mouse_pos):
                    new_depth = clamp_search_depth(search_depth + 1)
                    if new_depth != search_depth:
                        search_depth = new_depth
                        status_text = depth_status_text(search_depth)
                        renderer.play_sound("pickup")
                        print(status_text)
                    continue

                if renderer.is_ai_mode_button_clicked(mouse_pos, compact=False):
                    ai_mode, status_text = try_switch_ai_mode(ai_mode, nnue_weights_path, ddqn_weights_path)
                    selected = None
                    possible_moves = []
                    ai_once_pending = False
                    ai_pending_frames = 0
                    renderer.play_sound("pickup")
                    print(status_text)
                    continue

                if renderer.is_train_nnue_button_clicked(mouse_pos):
                    success, message = launch_nnue_training_gui(nnue_weights_path)
                    status_text = message
                    selected = None
                    possible_moves = []
                    if success:
                        renderer.play_sound("place")
                    else:
                        renderer.play_sound("surrender")
                    print(message)
                    continue

                if renderer.is_red_ai_button_clicked(mouse_pos):
                    red_ai_enabled = not red_ai_enabled
                    status_text = f"红方AI：{'开启' if red_ai_enabled else '关闭'}（{ai_mode_label(ai_mode)}）"
                    selected = None
                    possible_moves = []
                    ai_once_pending = False
                    renderer.play_sound("pickup")
                    print(status_text)
                    if not game_over and board.side > 0 and red_ai_enabled:
                        ai_pending_frames = 1
                    continue

                if renderer.is_black_ai_button_clicked(mouse_pos):
                    black_ai_enabled = not black_ai_enabled
                    status_text = f"黑方AI：{'开启' if black_ai_enabled else '关闭'}（{ai_mode_label(ai_mode)}）"
                    selected = None
                    possible_moves = []
                    ai_once_pending = False
                    renderer.play_sound("pickup")
                    print(status_text)
                    if not game_over and board.side < 0 and black_ai_enabled:
                        ai_pending_frames = 1
                    continue

                if renderer.is_ai_once_button_clicked(mouse_pos):
                    if not game_over:
                        ai_once_pending = True
                        ai_pending_frames = 1
                        selected = None
                        possible_moves = []
                        status_text = f"{ai_mode_label(ai_mode)}准备走一步"
                        renderer.play_sound("pickup")
                        print(status_text)
                    continue

                if game_over:
                    continue

                if side_is_ai(board.side, red_ai_enabled, black_ai_enabled) or ai_once_pending:
                    continue

                pos = renderer.get_click_pos(mouse_pos)
                if pos is None:
                    continue

                piece = board.board[pos]

                if selected is None:
                    if piece != 0 and (piece > 0) == (board.side > 0):
                        selected = pos
                        possible_moves = [m.to_pos for m in moves if m.from_pos == pos]
                        renderer.play_sound("pickup")
                else:
                    move_found = None

                    for m in moves:
                        if m.from_pos == selected and m.to_pos == pos:
                            move_found = m
                            break

                    if move_found:
                        status_text, game_over = apply_move(
                            board,
                            move_found,
                            notations,
                            renderer=renderer,
                        )
                        selected = None
                        possible_moves = []
                    else:
                        if piece != 0 and (piece > 0) == (board.side > 0):
                            selected = pos
                            possible_moves = [m.to_pos for m in moves if m.from_pos == pos]
                            renderer.play_sound("pickup")
                            continue

                        selected = None
                        possible_moves = []

        if not running:
            break

        auto_ai_turn = side_is_ai(board.side, red_ai_enabled, black_ai_enabled)
        run_ai_now = not game_over and (auto_ai_turn or ai_once_pending)

        if run_ai_now:
            if ai_pending_frames > 0:
                ai_pending_frames -= 1
            else:
                ai_move, actual_mode, fallback_reason = choose_current_ai_move(
                    board,
                    ai_mode,
                    search_depth,
                    nnue_weights_path,
                    ddqn_weights_path,
                    ddqn_device,
                )

                if fallback_reason:
                    ai_mode = actual_mode
                    status_text = f"{fallback_reason}，已切回{ai_mode_label(ai_mode)}"
                    print(status_text)

                if ai_move is None:
                    status_text = get_winner_text(-board.side)
                    print(status_text)
                    renderer.play_sound("checkmate")
                    saved_path = save_notations(notations)
                    print(f"棋谱已保存到：{saved_path}")
                    game_over = True
                else:
                    status_text, game_over = apply_move(
                        board,
                        ai_move,
                        notations,
                        renderer=renderer,
                        mover_name=ai_mode_label(actual_mode),
                    )
                    selected = None
                    possible_moves = []

                ai_once_pending = False

        clock.tick(60)

    pygame.quit()



def parse_args():
    parser = argparse.ArgumentParser(description="中国象棋对战 / AI托管模式")
    parser.add_argument(
        "--depth",
        type=int,
        default=AI_SEARCH_DEPTH,
        help=f"AI 搜索深度，范围 {MIN_SEARCH_DEPTH}-{MAX_SEARCH_DEPTH}，默认 {AI_SEARCH_DEPTH}",
    )
    parser.add_argument(
        "--disable-depth-hotkeys",
        action="store_true",
        help="禁用运行中通过键盘调整搜索深度",
    )
    parser.add_argument(
        "--ai-mode",
        choices=[AI_MODE_SEARCH, AI_MODE_NNUE, AI_MODE_DDQN],
        default=AI_MODE_SEARCH,
        help="AI 模式：search / nnue / ddqn",
    )
    parser.add_argument(
        "--nnue-weights",
        default=DEFAULT_NNUE_WEIGHTS,
        help="NNUE 权重文件路径",
    )
    parser.add_argument(
        "--ddqn-weights",
        default=DEFAULT_DDQN_WEIGHTS,
        help="DDQN 权重文件路径",
    )
    parser.add_argument(
        "--ddqn-device",
        default=None,
        help="DDQN 推理设备，例如 cpu / cuda",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    play(
        search_depth=args.depth,
        allow_depth_hotkeys=not args.disable_depth_hotkeys,
        ai_mode=args.ai_mode,
        nnue_weights_path=args.nnue_weights,
        ddqn_weights_path=args.ddqn_weights,
        ddqn_device=args.ddqn_device,
    )
