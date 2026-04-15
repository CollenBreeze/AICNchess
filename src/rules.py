# rules.py

def is_in_check(board, side):
    b = board.board

    # ======================
    # 找将的位置
    # ======================
    king_pos = -1
    target = 1 if side > 0 else -1

    for i in range(90):
        if b[i] == target:
            king_pos = i
            break

    if king_pos == -1:
        return False  # 理论不应该发生

    x, y = king_pos % 9, king_pos // 9

    # ======================
    # 1️⃣ 车 / 将（直线攻击）
    # ======================
    for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
        nx, ny = x + dx, y + dy

        blocked = False

        while 0 <= nx < 9 and 0 <= ny < 10:
            idx = ny * 9 + nx
            piece = b[idx]

            if piece != 0:
                # 第一颗棋
                if not blocked:
                    # 敌方车 或 敌方将（对脸）
                    if (piece > 0) != (side > 0):
                        if abs(piece) == 2 or abs(piece) == 1:
                            return True
                    blocked = True
                else:
                    # 第二颗棋 → 炮
                    if (piece > 0) != (side > 0):
                        if abs(piece) == 4:
                            return True
                    break

            nx += dx
            ny += dy

    # ======================
    # 2️⃣ 马攻击（蹩马腿）
    # ======================
    knight_patterns = [
        (2,1,1,0), (2,-1,1,0),
        (-2,1,-1,0), (-2,-1,-1,0),
        (1,2,0,1), (1,-2,0,-1),
        (-1,2,0,1), (-1,-2,0,-1),
    ]

    for dx, dy, bx, by in knight_patterns:
        bx_, by_ = x + bx, y + by

        if not (0 <= bx_ < 9 and 0 <= by_ < 10):
            continue

        # 马腿被堵
        if b[by_ * 9 + bx_] != 0:
            continue

        nx, ny = x + dx, y + dy
        if 0 <= nx < 9 and 0 <= ny < 10:
            idx = ny * 9 + nx
            piece = b[idx]

            if piece != 0 and (piece > 0) != (side > 0):
                if abs(piece) == 3:
                    return True

    # ======================
    # 3️⃣ 兵攻击
    # ======================
    if side > 0:
        # 红将 → 被黑兵攻击（向下）
        directions = [(0,1), (-1,0), (1,0)]
        enemy_pawn = -7
    else:
        directions = [(0,-1), (-1,0), (1,0)]
        enemy_pawn = 7

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < 9 and 0 <= ny < 10:
            if b[ny * 9 + nx] == enemy_pawn:
                return True

    return False
