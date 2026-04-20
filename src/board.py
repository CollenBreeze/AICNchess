# board.py

from bitboards import ensure_precomputed
from zobrist import SIDE_KEY, compute_hash, piece_key


DRAW_REPEAT_COUNT = 3


class Board:
    def __init__(self):
        ensure_precomputed()

        # 9 x 10 棋盘 -> 一维数组
        self.board = [0] * 90
        self.history = []

        # 当前行棋方：1 = 红，-1 = 黑
        self.side = 1

        # 增量维护的缓存
        self.piece_positions = {1: set(), -1: set()}
        self.king_pos = {1: -1, -1: -1}
        self.zhash = 0

        # 当前对局 / 搜索路径上，局面（含 side）出现次数
        self.position_counts = {}

    # ======================
    # 坐标转换
    # ======================
    @staticmethod
    def xy_to_index(x, y):
        return y * 9 + x

    @staticmethod
    def index_to_xy(index):
        return index % 9, index // 9

    # ======================
    # 内部缓存维护
    # ======================
    def _rebuild_state(self):
        self.piece_positions = {1: set(), -1: set()}
        self.king_pos = {1: -1, -1: -1}

        for pos, piece in enumerate(self.board):
            if piece == 0:
                continue

            side = 1 if piece > 0 else -1
            self.piece_positions[side].add(pos)

            if abs(piece) == 1:
                self.king_pos[side] = pos

        self.zhash = compute_hash(self.board, self.side)
        self.position_counts = {self.zhash: 1}

    def get_repeat_count(self, zhash=None):
        if zhash is None:
            zhash = self.zhash
        return self.position_counts.get(zhash, 0)

    def is_draw_by_repetition(self, threshold=DRAW_REPEAT_COUNT):
        return self.get_repeat_count() >= threshold

    # ======================
    # 初始化棋盘
    # ======================
    def init_startpos(self):
        b = self.board

        # 清空
        for i in range(90):
            b[i] = 0

        # ===== 黑方 =====
        b[0] = -2  # 车
        b[1] = -3  # 马
        b[2] = -5  # 象
        b[3] = -6  # 士
        b[4] = -1  # 将
        b[5] = -6
        b[6] = -5
        b[7] = -3
        b[8] = -2

        b[19] = -4  # 炮
        b[25] = -4

        for i in [27, 29, 31, 33, 35]:
            b[i] = -7  # 卒

        # ===== 红方 =====
        b[81] = 2
        b[82] = 3
        b[83] = 5
        b[84] = 6
        b[85] = 1
        b[86] = 6
        b[87] = 5
        b[88] = 3
        b[89] = 2

        b[64] = 4
        b[70] = 4

        for i in [54, 56, 58, 60, 62]:
            b[i] = 7

        self.side = 1  # 红先
        self.history.clear()
        self._rebuild_state()

    # ======================
    # 获取棋子
    # ======================
    def get_piece(self, x, y):
        return self.board[self.xy_to_index(x, y)]

    def set_piece(self, x, y, piece):
        self.board[self.xy_to_index(x, y)] = piece
        self.history.clear()
        self._rebuild_state()

    # ======================
    # 判断阵营
    # ======================
    @staticmethod
    def is_red(piece):
        return piece > 0

    @staticmethod
    def is_black(piece):
        return piece < 0

    # ======================
    # 打印棋盘（调试）
    # ======================
    def print_board(self):
        piece_map = {
            0: ".",
            1: "J", 2: "C", 3: "M", 4: "P", 5: "X", 6: "S", 7: "B",
            -1: "j", -2: "c", -3: "m", -4: "p", -5: "x", -6: "s", -7: "z",
        }

        for y in range(10):
            row = []
            for x in range(9):
                piece = self.get_piece(x, y)
                row.append(piece_map[piece])
            print(" ".join(row))

        print("Side to move:", "Red" if self.side == 1 else "Black")

    def make_move(self, move):
        b = self.board

        piece = b[move.from_pos]
        captured = b[move.to_pos]
        moving_side = 1 if piece > 0 else -1

        prev_side = self.side
        prev_red_king = self.king_pos[1]
        prev_black_king = self.king_pos[-1]
        prev_hash = self.zhash

        # 记录吃子（用于 undo）
        move.captured = captured

        # Zobrist：移除旧位置、吃子、加入新位置、切换行棋方
        self.zhash ^= piece_key(piece, move.from_pos)
        if captured != 0:
            self.zhash ^= piece_key(captured, move.to_pos)
        self.zhash ^= piece_key(piece, move.to_pos)
        self.zhash ^= SIDE_KEY

        # 更新棋盘
        b[move.to_pos] = piece
        b[move.from_pos] = 0

        # 更新缓存
        self.piece_positions[moving_side].remove(move.from_pos)
        if captured != 0:
            self.piece_positions[-moving_side].remove(move.to_pos)
            if abs(captured) == 1:
                self.king_pos[-moving_side] = -1
        self.piece_positions[moving_side].add(move.to_pos)

        if abs(piece) == 1:
            self.king_pos[moving_side] = move.to_pos

        # 切换行棋方
        self.side = -self.side

        # 记录新局面重复次数
        post_hash = self.zhash
        self.position_counts[post_hash] = self.position_counts.get(post_hash, 0) + 1

        # 保存历史，供 undo 使用
        self.history.append(
            (
                move,
                piece,
                captured,
                prev_side,
                prev_red_king,
                prev_black_king,
                prev_hash,
                post_hash,
            )
        )

    def undo_move(self):
        if not self.history:
            return False

        move, piece, captured, prev_side, red_king, black_king, prev_hash, post_hash = self.history.pop()

        count = self.position_counts.get(post_hash, 0)
        if count <= 1:
            self.position_counts.pop(post_hash, None)
        else:
            self.position_counts[post_hash] = count - 1

        moving_side = 1 if piece > 0 else -1
        b = self.board

        # 还原棋盘
        b[move.from_pos] = piece
        b[move.to_pos] = captured

        # 还原缓存
        self.piece_positions[moving_side].remove(move.to_pos)
        self.piece_positions[moving_side].add(move.from_pos)
        if captured != 0:
            self.piece_positions[-moving_side].add(move.to_pos)

        self.king_pos[1] = red_king
        self.king_pos[-1] = black_king
        self.side = prev_side
        self.zhash = prev_hash
        return True


if __name__ == "__main__":
    board = Board()
    board.init_startpos()
    board.print_board()
