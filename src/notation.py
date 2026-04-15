# notation.py

PIECE_NAME = {
    1: "帅", 2: "车", 3: "马", 4: "炮", 5: "相", 6: "仕", 7: "兵",
    -1: "将", -2: "车", -3: "马", -4: "炮", -5: "象", -6: "士", -7: "卒",
}

NUM_MAP = {
    1: "一", 2: "二", 3: "三", 4: "四", 5: "五",
    6: "六", 7: "七", 8: "八", 9: "九"
}


def format_num(n, side):
    """红方使用中文数字，黑方使用阿拉伯数字。"""
    return NUM_MAP[n] if side > 0 else str(n)


def file_to_col(x, side):
    return 9 - x if side > 0 else x + 1


def sort_positions_front_to_back(positions, side):
    """按“前 -> 后”的顺序排序。"""
    return sorted(positions, key=lambda p: p // 9, reverse=(side < 0))


def find_same_file_pieces(board, piece, x):
    b = board.board
    return [pos for pos in range(90) if b[pos] == piece and pos % 9 == x]


def get_front_back(piece, positions):
    side = 1 if piece > 0 else -1
    ordered = sort_positions_front_to_back(positions, side)

    if len(ordered) < 2:
        return {}

    return {ordered[0]: "前", ordered[-1]: "后"}


# ===== 兵相关 =====

def get_pawn_groups(board, piece):
    b = board.board
    groups = {}

    for pos in range(90):
        if b[pos] == piece:
            x = pos % 9
            groups.setdefault(x, []).append(pos)

    return groups


def get_pawn_rank(side, positions):
    """
    兵卒同列时的称谓：
    2个：前 / 后
    3个：前 / 中 / 后
    4个：前 / 二 / 三 / 四
    5个：前 / 二 / 三 / 四 / 五
    """
    ordered = sort_positions_front_to_back(positions, side)
    n = len(ordered)

    if n < 2:
        return {}

    if n == 2:
        labels = ["前", "后"]
    elif n == 3:
        labels = ["前", "中", "后"]
    else:
        # 标准规则定义到5个；这里顺手做成可扩展写法。
        labels = ["前"] + [NUM_MAP.get(i, str(i)) for i in range(2, n + 1)]

    return {pos: label for pos, label in zip(ordered, labels)}


def get_action_and_target(piece, x1, y1, x2, y2, side):
    """
    生成“进 / 退 / 平”和目标数字。
    - 车、炮、帅/将、兵/卒：横走记“平”，直走记“进/退”
    - 马、相/象、仕/士：记“进/退”，目标为到达的纵线
    """
    forward = (side > 0 and y2 < y1) or (side < 0 and y2 > y1)

    if x1 == x2:
        action = "进" if forward else "退"
        target = format_num(abs(y2 - y1), side)
    elif abs(piece) in (3, 5, 6):
        action = "进" if forward else "退"
        target = format_num(file_to_col(x2, side), side)
    else:
        action = "平"
        target = format_num(file_to_col(x2, side), side)

    return action, target


# ======================
# 主函数
# ======================
def move_to_notation(board, move):
    b = board.board

    piece = b[move.from_pos]

    x1, y1 = move.from_pos % 9, move.from_pos // 9
    x2, y2 = move.to_pos % 9, move.to_pos // 9

    side = 1 if piece > 0 else -1
    action, target = get_action_and_target(piece, x1, y1, x2, y2, side)

    # ======================
    # 兵 / 卒
    # ======================
    if abs(piece) == 7:
        groups = get_pawn_groups(board, piece)
        current_group = groups.get(x1, [])
        multi_file_count = sum(1 for g in groups.values() if len(g) >= 2)

        # 只有当前这一路本身存在重兵时，才需要使用前/后/中/二/三...来区分
        if len(current_group) >= 2:
            rank_map = get_pawn_rank(side, current_group)
            prefix = rank_map.get(move.from_pos, "")

            # 两个纵线都达到两个以上：按旧式写法，可省略“兵/卒”
            # 例如：前九平八、二7平6
            if multi_file_count >= 2:
                name = prefix
                col_str = format_num(file_to_col(x1, side), side)
            else:
                # 只有单一路重兵：前兵进一 / 中兵平五 / 二卒平6 ...
                name = prefix + PIECE_NAME[piece]
                col_str = ""
        else:
            # 普通兵卒
            name = PIECE_NAME[piece]
            col_str = format_num(file_to_col(x1, side), side)

    # ======================
    # 其他棋子
    # ======================
    else:
        same_file = find_same_file_pieces(board, piece, x1)

        if len(same_file) >= 2:
            fb_map = get_front_back(piece, same_file)
            prefix = fb_map.get(move.from_pos)

            if prefix:
                name = prefix + PIECE_NAME[piece]
                col_str = ""
            else:
                name = PIECE_NAME[piece]
                col_str = format_num(file_to_col(x1, side), side)
        else:
            name = PIECE_NAME[piece]
            col_str = format_num(file_to_col(x1, side), side)

    return f"{name}{col_str}{action}{target}"