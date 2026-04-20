# bitboards.py

"""
预计算中国象棋的静态走法表。

这里并不维护“当前局面”的占用情况，只负责把：
- 车/炮的四个方向射线
- 马的落点与马腿
- 象的落点与象眼（并按红黑两侧拆分不过河规则）
- 士/将的九宫走法（按红黑拆分）
- 兵卒走法
- 反向攻击表（用于快速判将）

全部在启动时一次性算好，后续走子生成只做极少量局面判断。
"""

BOARD_SIZE = 90
BOARD_WIDTH = 9
BOARD_HEIGHT = 10

FILE_OF = tuple(pos % BOARD_WIDTH for pos in range(BOARD_SIZE))
RANK_OF = tuple(pos // BOARD_WIDTH for pos in range(BOARD_SIZE))

ROOK_RAYS = {
    "up": [tuple() for _ in range(BOARD_SIZE)],
    "down": [tuple() for _ in range(BOARD_SIZE)],
    "left": [tuple() for _ in range(BOARD_SIZE)],
    "right": [tuple() for _ in range(BOARD_SIZE)],
}

KNIGHT_STEPS = [tuple() for _ in range(BOARD_SIZE)]
KNIGHT_ATTACKERS = [tuple() for _ in range(BOARD_SIZE)]

ELEPHANT_STEPS_RED = [tuple() for _ in range(BOARD_SIZE)]
ELEPHANT_STEPS_BLACK = [tuple() for _ in range(BOARD_SIZE)]

ADVISOR_MOVES_RED = [tuple() for _ in range(BOARD_SIZE)]
ADVISOR_MOVES_BLACK = [tuple() for _ in range(BOARD_SIZE)]

KING_MOVES_RED = [tuple() for _ in range(BOARD_SIZE)]
KING_MOVES_BLACK = [tuple() for _ in range(BOARD_SIZE)]

PAWN_MOVES_RED = [tuple() for _ in range(BOARD_SIZE)]
PAWN_MOVES_BLACK = [tuple() for _ in range(BOARD_SIZE)]

PAWN_ATTACKERS_RED = [tuple() for _ in range(BOARD_SIZE)]
PAWN_ATTACKERS_BLACK = [tuple() for _ in range(BOARD_SIZE)]

_INITIALIZED = False


# ======================
# 工具函数
# ======================

def in_board(x, y):
    return 0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT


def in_palace(x, y, side):
    if side > 0:
        return 3 <= x <= 5 and 7 <= y <= 9
    return 3 <= x <= 5 and 0 <= y <= 2


# ======================
# 初始化细节
# ======================

def _freeze_table(table):
    return [tuple(items) for items in table]


def _init_rook_rays():
    for pos in range(BOARD_SIZE):
        x = FILE_OF[pos]
        y = RANK_OF[pos]

        up = []
        for ny in range(y - 1, -1, -1):
            up.append(ny * BOARD_WIDTH + x)
        ROOK_RAYS["up"][pos] = tuple(up)

        down = []
        for ny in range(y + 1, BOARD_HEIGHT):
            down.append(ny * BOARD_WIDTH + x)
        ROOK_RAYS["down"][pos] = tuple(down)

        left = []
        for nx in range(x - 1, -1, -1):
            left.append(y * BOARD_WIDTH + nx)
        ROOK_RAYS["left"][pos] = tuple(left)

        right = []
        for nx in range(x + 1, BOARD_WIDTH):
            right.append(y * BOARD_WIDTH + nx)
        ROOK_RAYS["right"][pos] = tuple(right)


def _init_knight():
    directions = (
        (-1, -2, 0, -1), (1, -2, 0, -1),
        (-2, -1, -1, 0), (-2, 1, -1, 0),
        (2, -1, 1, 0), (2, 1, 1, 0),
        (-1, 2, 0, 1), (1, 2, 0, 1),
    )

    steps = [[] for _ in range(BOARD_SIZE)]
    attackers = [[] for _ in range(BOARD_SIZE)]

    for pos in range(BOARD_SIZE):
        x = FILE_OF[pos]
        y = RANK_OF[pos]

        for dx, dy, lx, ly in directions:
            nx = x + dx
            ny = y + dy
            legx = x + lx
            legy = y + ly

            if not in_board(nx, ny):
                continue

            to_sq = ny * BOARD_WIDTH + nx
            leg_sq = legy * BOARD_WIDTH + legx
            steps[pos].append((to_sq, leg_sq))
            attackers[to_sq].append((pos, leg_sq))

    frozen_steps = _freeze_table(steps)
    frozen_attackers = _freeze_table(attackers)

    for pos in range(BOARD_SIZE):
        KNIGHT_STEPS[pos] = frozen_steps[pos]
        KNIGHT_ATTACKERS[pos] = frozen_attackers[pos]


def _init_elephant():
    directions = ((-2, -2), (2, -2), (-2, 2), (2, 2))

    red_steps = [[] for _ in range(BOARD_SIZE)]
    black_steps = [[] for _ in range(BOARD_SIZE)]

    for pos in range(BOARD_SIZE):
        x = FILE_OF[pos]
        y = RANK_OF[pos]

        for dx, dy in directions:
            nx = x + dx
            ny = y + dy
            ex = x + dx // 2
            ey = y + dy // 2

            if not in_board(nx, ny):
                continue

            to_sq = ny * BOARD_WIDTH + nx
            eye_sq = ey * BOARD_WIDTH + ex
            step = (to_sq, eye_sq)

            if ny >= 5:
                red_steps[pos].append(step)
            if ny <= 4:
                black_steps[pos].append(step)

    frozen_red = _freeze_table(red_steps)
    frozen_black = _freeze_table(black_steps)

    for pos in range(BOARD_SIZE):
        ELEPHANT_STEPS_RED[pos] = frozen_red[pos]
        ELEPHANT_STEPS_BLACK[pos] = frozen_black[pos]


def _init_advisor():
    directions = ((-1, -1), (1, -1), (-1, 1), (1, 1))

    red_moves = [[] for _ in range(BOARD_SIZE)]
    black_moves = [[] for _ in range(BOARD_SIZE)]

    for pos in range(BOARD_SIZE):
        x = FILE_OF[pos]
        y = RANK_OF[pos]

        for dx, dy in directions:
            nx = x + dx
            ny = y + dy

            if not in_board(nx, ny):
                continue

            to_sq = ny * BOARD_WIDTH + nx
            if in_palace(nx, ny, 1):
                red_moves[pos].append(to_sq)
            if in_palace(nx, ny, -1):
                black_moves[pos].append(to_sq)

    frozen_red = _freeze_table(red_moves)
    frozen_black = _freeze_table(black_moves)

    for pos in range(BOARD_SIZE):
        ADVISOR_MOVES_RED[pos] = frozen_red[pos]
        ADVISOR_MOVES_BLACK[pos] = frozen_black[pos]


def _init_king():
    directions = ((0, -1), (0, 1), (-1, 0), (1, 0))

    red_moves = [[] for _ in range(BOARD_SIZE)]
    black_moves = [[] for _ in range(BOARD_SIZE)]

    for pos in range(BOARD_SIZE):
        x = FILE_OF[pos]
        y = RANK_OF[pos]

        for dx, dy in directions:
            nx = x + dx
            ny = y + dy

            if not in_board(nx, ny):
                continue

            to_sq = ny * BOARD_WIDTH + nx
            if in_palace(nx, ny, 1):
                red_moves[pos].append(to_sq)
            if in_palace(nx, ny, -1):
                black_moves[pos].append(to_sq)

    frozen_red = _freeze_table(red_moves)
    frozen_black = _freeze_table(black_moves)

    for pos in range(BOARD_SIZE):
        KING_MOVES_RED[pos] = frozen_red[pos]
        KING_MOVES_BLACK[pos] = frozen_black[pos]


def _init_pawn():
    red_moves = [[] for _ in range(BOARD_SIZE)]
    black_moves = [[] for _ in range(BOARD_SIZE)]
    red_attackers = [[] for _ in range(BOARD_SIZE)]
    black_attackers = [[] for _ in range(BOARD_SIZE)]

    for pos in range(BOARD_SIZE):
        x = FILE_OF[pos]
        y = RANK_OF[pos]

        # 红兵
        if y > 0:
            to_sq = (y - 1) * BOARD_WIDTH + x
            red_moves[pos].append(to_sq)
            red_attackers[to_sq].append(pos)

        if y <= 4:
            if x > 0:
                to_sq = y * BOARD_WIDTH + (x - 1)
                red_moves[pos].append(to_sq)
                red_attackers[to_sq].append(pos)
            if x < 8:
                to_sq = y * BOARD_WIDTH + (x + 1)
                red_moves[pos].append(to_sq)
                red_attackers[to_sq].append(pos)

        # 黑卒
        if y < 9:
            to_sq = (y + 1) * BOARD_WIDTH + x
            black_moves[pos].append(to_sq)
            black_attackers[to_sq].append(pos)

        if y >= 5:
            if x > 0:
                to_sq = y * BOARD_WIDTH + (x - 1)
                black_moves[pos].append(to_sq)
                black_attackers[to_sq].append(pos)
            if x < 8:
                to_sq = y * BOARD_WIDTH + (x + 1)
                black_moves[pos].append(to_sq)
                black_attackers[to_sq].append(pos)

    frozen_red_moves = _freeze_table(red_moves)
    frozen_black_moves = _freeze_table(black_moves)
    frozen_red_attackers = _freeze_table(red_attackers)
    frozen_black_attackers = _freeze_table(black_attackers)

    for pos in range(BOARD_SIZE):
        PAWN_MOVES_RED[pos] = frozen_red_moves[pos]
        PAWN_MOVES_BLACK[pos] = frozen_black_moves[pos]
        PAWN_ATTACKERS_RED[pos] = frozen_red_attackers[pos]
        PAWN_ATTACKERS_BLACK[pos] = frozen_black_attackers[pos]


# ======================
# 对外入口
# ======================

def init_masks():
    global _INITIALIZED
    if _INITIALIZED:
        return

    _init_rook_rays()
    _init_knight()
    _init_elephant()
    _init_advisor()
    _init_king()
    _init_pawn()
    _INITIALIZED = True


def ensure_precomputed():
    init_masks()
