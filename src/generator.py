# generator.py

from move import Move


# ======================
# 工具函数
# ======================
def same_side(a, b):
    return (a > 0) == (b > 0)


# ======================
# 总入口
# ======================
def generate_all_moves(board):
    b = board.board
    side = board.side
    moves = []

    for pos in range(90):
        piece = b[pos]
        if piece == 0:
            continue

        if (piece > 0) != (side > 0):
            continue

        p = abs(piece)

        if p == 2:
            gen_car(b, pos, piece, moves)
        elif p == 3:
            gen_horse(b, pos, piece, moves)
        elif p == 4:
            gen_boom(b, pos, piece, moves)
        elif p == 7:
            gen_soldier(b, pos, piece, moves)
        elif p == 1:
            gen_king(b, pos, piece, moves)
        elif p == 6:
            gen_shi(b, pos, piece, moves)
        elif p == 5:
            gen_elephant(b, pos, piece, moves)

    return moves


# ======================
# 🚗 车（Car）
# ======================
def gen_car(b, pos, piece, moves):
    x, y = pos % 9, pos // 9

    for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
        nx, ny = x + dx, y + dy

        while 0 <= nx < 9 and 0 <= ny < 10:
            idx = ny * 9 + nx
            target = b[idx]

            if target == 0:
                moves.append(Move(pos, idx))
            else:
                if not same_side(target, piece):
                    moves.append(Move(pos, idx, target))
                break

            nx += dx
            ny += dy


# ======================
# 🐎 马（Horse）
# ======================
def gen_horse(b, pos, piece, moves):
    x, y = pos % 9, pos // 9

    patterns = [
        (2,1,1,0), (2,-1,1,0),
        (-2,1,-1,0), (-2,-1,-1,0),
        (1,2,0,1), (1,-2,0,-1),
        (-1,2,0,1), (-1,-2,0,-1),
    ]

    for dx, dy, bx, by in patterns:
        bx_, by_ = x + bx, y + by
        if not (0 <= bx_ < 9 and 0 <= by_ < 10):
            continue

        if b[by_ * 9 + bx_] != 0:
            continue

        nx, ny = x + dx, y + dy
        if 0 <= nx < 9 and 0 <= ny < 10:
            idx = ny * 9 + nx
            target = b[idx]

            if target == 0 or not same_side(target, piece):
                moves.append(Move(pos, idx, target))


# ======================
# 💣 炮（Boom）——引擎级实现
# ======================
def gen_boom(b, pos, piece, moves):
    x, y = pos % 9, pos // 9

    for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
        nx, ny = x + dx, y + dy

        # 第一阶段：走（无阻挡）
        while 0 <= nx < 9 and 0 <= ny < 10:
            idx = ny * 9 + nx
            if b[idx] == 0:
                moves.append(Move(pos, idx))
                nx += dx
                ny += dy
            else:
                break

        # 第二阶段：找“炮架”（第一个棋子）
        nx += dx
        ny += dy

        # 第三阶段：吃（跳过一个后找目标）
        while 0 <= nx < 9 and 0 <= ny < 10:
            idx = ny * 9 + nx
            target = b[idx]

            if target != 0:
                if not same_side(target, piece):
                    moves.append(Move(pos, idx, target))
                break

            nx += dx
            ny += dy


# ======================
# 🪖 兵（Soldier）
# ======================
def gen_soldier(b, pos, piece, moves):
    x, y = pos % 9, pos // 9
    forward = -1 if piece > 0 else 1

    # 前进
    ny = y + forward
    if 0 <= ny < 10:
        idx = ny * 9 + x
        if b[idx] == 0 or not same_side(b[idx], piece):
            moves.append(Move(pos, idx, b[idx]))

    # 过河判断
    crossed = (y < 5) if piece > 0 else (y > 4)

    if crossed:
        for dx in [-1, 1]:
            nx = x + dx
            if 0 <= nx < 9:
                idx = y * 9 + nx
                if b[idx] == 0 or not same_side(b[idx], piece):
                    moves.append(Move(pos, idx, b[idx]))


# ======================
# 👑 将 / 帅（King）
# ======================
def gen_king(b, pos, piece, moves):
    x, y = pos % 9, pos // 9

    palace_x = range(3, 6)
    palace_y = range(7, 10) if piece > 0 else range(0, 3)

    for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
        nx, ny = x + dx, y + dy
        if nx in palace_x and ny in palace_y:
            idx = ny * 9 + nx
            if b[idx] == 0 or not same_side(b[idx], piece):
                moves.append(Move(pos, idx, b[idx]))


# ======================
# 🛡 士（Shi）
# ======================
def gen_shi(b, pos, piece, moves):
    x, y = pos % 9, pos // 9

    palace_x = range(3, 6)
    palace_y = range(7, 10) if piece > 0 else range(0, 3)

    for dx, dy in [(1,1), (1,-1), (-1,1), (-1,-1)]:
        nx, ny = x + dx, y + dy
        if nx in palace_x and ny in palace_y:
            idx = ny * 9 + nx
            if b[idx] == 0 or not same_side(b[idx], piece):
                moves.append(Move(pos, idx, b[idx]))


# ======================
# 🐘 象（Elephant）
# ======================
def gen_elephant(b, pos, piece, moves):
    x, y = pos % 9, pos // 9

    for dx, dy in [(2,2), (2,-2), (-2,2), (-2,-2)]:
        eye_x, eye_y = x + dx // 2, y + dy // 2
        nx, ny = x + dx, y + dy

        if not (0 <= nx < 9 and 0 <= ny < 10):
            continue

        # 象眼
        if b[eye_y * 9 + eye_x] != 0:
            continue

        # 不过河
        if piece > 0 and ny < 5:
            continue
        if piece < 0 and ny > 4:
            continue

        idx = ny * 9 + nx
        if b[idx] == 0 or not same_side(b[idx], piece):
            moves.append(Move(pos, idx, b[idx]))