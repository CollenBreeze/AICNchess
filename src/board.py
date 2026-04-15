# board.py

class Board:
    def __init__(self):
        # 9 x 10 棋盘 -> 一维数组
        self.board = [0] * 90
        self.history = []
        # 当前行棋方：1 = 红，-1 = 黑
        self.side = 1

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

    # ======================
    # 获取棋子
    # ======================
    def get_piece(self, x, y):
        return self.board[self.xy_to_index(x, y)]

    def set_piece(self, x, y, piece):
        self.board[self.xy_to_index(x, y)] = piece

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

        # 记录吃子（用于 undo）
        move.captured = captured

        # 保存历史
        self.history.append(move)

        # 移动棋子
        b[move.to_pos] = piece
        b[move.from_pos] = 0

        # 切换行棋方
        self.side = -self.side

    def undo_move(self):
        if not self.history:
            return False

        move = self.history.pop()
        b = self.board

        piece = b[move.to_pos]

        # 还原棋盘
        b[move.from_pos] = piece
        b[move.to_pos] = move.captured

        # 切换回原方
        self.side = -self.side
        return True


if __name__ == "__main__":
    board = Board()
    board.init_startpos()
    board.print_board()