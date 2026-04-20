# ai_selector.py

import os

from ai_heuristic import choose_heuristic_move

AI_MODE_SEARCH = "search"
AI_MODE_NNUE = "nnue"
AI_MODE_DDQN = "ddqn"

AI_MODE_LABELS = {
    AI_MODE_SEARCH: "搜索AI",
    AI_MODE_NNUE: "NNUE AI",
    AI_MODE_DDQN: "DDQN AI",
}

try:
    from nnue_player import choose_nnue_move
    _NNUE_IMPORT_ERROR = None
except Exception as exc:
    choose_nnue_move = None
    _NNUE_IMPORT_ERROR = exc

try:
    from rl.ddqn_player import choose_ddqn_move
    _DDQN_IMPORT_ERROR = None
except Exception as exc:
    choose_ddqn_move = None
    _DDQN_IMPORT_ERROR = exc


ALL_AI_MODES = (AI_MODE_SEARCH, AI_MODE_NNUE, AI_MODE_DDQN)



def normalize_ai_mode(ai_mode):
    mode = str(ai_mode).strip().lower()
    if mode in AI_MODE_LABELS:
        return mode
    return AI_MODE_SEARCH



def ai_mode_label(ai_mode):
    return AI_MODE_LABELS[normalize_ai_mode(ai_mode)]



def ai_mode_button_text(ai_mode):
    return f"AI模式：{ai_mode_label(ai_mode)}"



def nnue_unavailable_reason(weights_path):
    if choose_nnue_move is None:
        if _NNUE_IMPORT_ERROR is None:
            return "NNUE 模块不可用"
        return f"NNUE 模块加载失败：{_NNUE_IMPORT_ERROR}"

    if not weights_path:
        return "未设置 NNUE 权重路径"

    if not os.path.isfile(weights_path):
        return f"未找到 NNUE 权重：{weights_path}"

    return None



def ddqn_unavailable_reason(weights_path):
    if choose_ddqn_move is None:
        if _DDQN_IMPORT_ERROR is None:
            return "DDQN 模块不可用"
        return f"DDQN 模块加载失败：{_DDQN_IMPORT_ERROR}"

    if not weights_path:
        return "未设置 DDQN 权重路径"

    if not os.path.isfile(weights_path):
        return f"未找到 DDQN 权重：{weights_path}"

    return None



def is_nnue_available(weights_path):
    return nnue_unavailable_reason(weights_path) is None



def is_ddqn_available(weights_path):
    return ddqn_unavailable_reason(weights_path) is None



def cycle_ai_mode(current_mode, *, nnue_weights_path=None, ddqn_weights_path=None):
    current_mode = normalize_ai_mode(current_mode)
    idx = ALL_AI_MODES.index(current_mode)

    for offset in range(1, len(ALL_AI_MODES) + 1):
        mode = ALL_AI_MODES[(idx + offset) % len(ALL_AI_MODES)]
        if mode == AI_MODE_SEARCH:
            return mode
        if mode == AI_MODE_NNUE and is_nnue_available(nnue_weights_path):
            return mode
        if mode == AI_MODE_DDQN and is_ddqn_available(ddqn_weights_path):
            return mode

    return AI_MODE_SEARCH



def choose_ai_move(
    board,
    *,
    ai_mode=AI_MODE_SEARCH,
    search_depth=3,
    search_preset="balanced",
    search_params_path=None,
    nnue_weights_path=None,
    ddqn_weights_path=None,
    ddqn_device=None,
):
    ai_mode = normalize_ai_mode(ai_mode)

    if ai_mode == AI_MODE_NNUE:
        reason = nnue_unavailable_reason(nnue_weights_path)
        if reason is None:
            try:
                move = choose_nnue_move(
                    board,
                    depth=search_depth,
                    weights_path=nnue_weights_path,
                    preset=search_preset,
                )
                return move, AI_MODE_NNUE, None
            except Exception as exc:
                reason = f"NNUE 推理失败：{exc}"

        fallback_move = choose_heuristic_move(
            board,
            depth=search_depth,
            preset=search_preset,
            params_path=search_params_path,
        )
        return fallback_move, AI_MODE_SEARCH, reason

    if ai_mode == AI_MODE_DDQN:
        reason = ddqn_unavailable_reason(ddqn_weights_path)
        if reason is None:
            try:
                move = choose_ddqn_move(
                    board,
                    weights_path=ddqn_weights_path,
                    device=ddqn_device,
                )
                return move, AI_MODE_DDQN, None
            except Exception as exc:
                reason = f"DDQN 推理失败：{exc}"

        fallback_move = choose_heuristic_move(
            board,
            depth=search_depth,
            preset=search_preset,
            params_path=search_params_path,
        )
        return fallback_move, AI_MODE_SEARCH, reason

    move = choose_heuristic_move(
        board,
        depth=search_depth,
        preset=search_preset,
        params_path=search_params_path,
    )
    return move, AI_MODE_SEARCH, None
