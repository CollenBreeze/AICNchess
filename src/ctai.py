# ctai.py

import argparse
import os
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

AI_SIDE = -1   # 黑方 AI
AI_PRESET = "balanced"
AI_PARAMS_FILE = None  # 例如："/path/to/params_A_xxx.json"
AI_SEARCH_DEPTH = 4
MIN_SEARCH_DEPTH = 1
MAX_SEARCH_DEPTH = 8
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
QIPU_DIR = os.path.join(BASE_DIR, "Qipu")
DEFAULT_NNUE_WEIGHTS = os.path.join(BASE_DIR, "checkpoints_nnue", "xiangqi_nnue_v2.npz")
DEFAULT_DDQN_WEIGHTS = os.path.join(BASE_DIR, "checkpoints_ddqn", "ddqn_final.pt")



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



def apply_move(board, move, notations, renderer=None, mover_name=None):
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



def ctai(
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

    # 用来保证：玩家走完后先渲染一帧，再轮到 AI 算
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
            red_ai_enabled=(AI_SIDE > 0),
            black_ai_enabled=(AI_SIDE < 0),
            ai_mode=ai_mode,
            show_depth_controls=True,
            show_ai_controls=False,
            show_ai_mode_button=True,
        )

        moves = [] if game_over else generate_legal_moves(board)

        # 无棋可走判负
        if not game_over and len(moves) == 0:
            status_text = get_winner_text(-board.side)
            print(status_text)
            renderer.play_sound("checkmate")
            saved_path = save_notations(notations)
            print(f"棋谱已保存到：{saved_path}")
            game_over = True

        # ======================
        # 统一处理事件
        # ======================
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

                # 悔棋按钮
                if renderer.is_undo_button_clicked(mouse_pos):
                    success, message = undo_last_round(board, notations)
                    status_text = message
                    if success:
                        game_over = False
                        ai_pending_frames = 0
                        renderer.play_sound("undo")
                    selected = None
                    possible_moves = []
                    print(message)
                    continue

                # 保存棋谱按钮
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

                if renderer.is_ai_mode_button_clicked(mouse_pos, compact=True):
                    ai_mode, status_text = try_switch_ai_mode(ai_mode, nnue_weights_path, ddqn_weights_path)
                    selected = None
                    possible_moves = []
                    ai_pending_frames = 0
                    renderer.play_sound("pickup")
                    print(status_text)
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
                        renderer.play_sound("pickup")
                else:
                    move_found = None

                    for m in moves:
                        if m.from_pos == selected and m.to_pos == pos:
                            move_found = m
                            break

                    if move_found:
                        status_text, game_over = apply_move(board, move_found, notations, renderer=renderer)

                        selected = None
                        possible_moves = []

                        # 玩家走完后，先显示当前局面一帧，再让 AI 走
                        if not game_over and board.side == AI_SIDE:
                            ai_pending_frames = 1

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

        # ======================
        # AI 回合
        # 先等一帧，确保玩家刚落子的画面先显示出来
        # ======================
        if not game_over and board.side == AI_SIDE:
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
                    status_text = get_winner_text(-AI_SIDE)
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

        clock.tick(60)

    pygame.quit()



def parse_args():
    parser = argparse.ArgumentParser(description="中国象棋人机对战")
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
    ctai(
        search_depth=args.depth,
        allow_depth_hotkeys=not args.disable_depth_hotkeys,
        ai_mode=args.ai_mode,
        nnue_weights_path=args.nnue_weights,
        ddqn_weights_path=args.ddqn_weights,
        ddqn_device=args.ddqn_device,
    )
