# renderer.py

import os
import pygame

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "pctr")
FONTS_DIR = os.path.join(BASE_DIR, "fonts")

CELL_SIZE = 64
BOARD_OFFSET_X = 50
BOARD_OFFSET_Y = 50
BOARD_WIDTH = CELL_SIZE * 9
BOARD_HEIGHT = CELL_SIZE * 10

PANEL_X = BOARD_OFFSET_X + BOARD_WIDTH + 24
PANEL_WIDTH = 160
SCREEN_WIDTH = PANEL_X + PANEL_WIDTH + 26
SCREEN_HEIGHT = BOARD_OFFSET_Y * 2 + BOARD_HEIGHT

BG_COLOR = (240, 220, 180)
PANEL_BG = (248, 237, 211)
PANEL_BORDER = (153, 102, 51)
BUTTON_COLOR = (188, 120, 67)
BUTTON_HOVER = (205, 136, 82)
BUTTON_TEXT = (255, 255, 255)
TEXT_COLOR = (60, 40, 20)
STATUS_GOOD = (160, 20, 20)
SELECT_COLOR = (255, 0, 0)
MOVE_HINT_COLOR = (0, 180, 0)


class Renderer:
    def __init__(self):
        pygame.init()
        pygame.font.init()

        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Chinese Chess")

        self.board_img = pygame.image.load(
            os.path.join(ASSETS_DIR, "board.png")
        )
        self.board_img = pygame.transform.scale(
            self.board_img,
            (BOARD_WIDTH, BOARD_HEIGHT)
        )

        self.piece_images = self.load_pieces()

        # 不再使用 pygame.font.SysFont，避免 Windows 系统字体枚举报错
        self.font = self.load_font(24)
        self.small_font = self.load_font(20)
        self.tiny_font = self.load_font(18)

        self.undo_button_rect = pygame.Rect(PANEL_X + 10, 170, 140, 46)
        self.save_button_rect = pygame.Rect(PANEL_X + 10, 230, 140, 46)

    def load_font(self, size):
        """
        优先加载项目内字体文件，避免 SysFont 在部分 Windows + pygame 环境下报错。
        可自行把中文字体放到项目根目录 fonts/ 下，例如：
        - fonts/simhei.ttf
        - fonts/msyh.ttc
        - fonts/simsun.ttc
        """
        candidates = [
            os.path.join(FONTS_DIR, "QINGNIAO.ttf"),

        ]

        for path in candidates:
            if os.path.exists(path):
                try:
                    return pygame.font.Font(path, size)
                except Exception:
                    pass

        # 最后回退到 pygame 默认字体
        return pygame.font.Font(None, size)

    def load_pieces(self):
        pieces = {}

        mapping = {
            1: "red_king.png", 2: "red_rook1.png", 3: "red_knight1.png", 4: "red_cannon1.png",
            5: "red_bishop1.png", 6: "red_advisor1.png", 7: "red_pawn1.png",
            -1: "black_king.png", -2: "black_rook1.png", -3: "black_knight1.png", -4: "black_cannon1.png",
            -5: "black_bishop1.png", -6: "black_advisor1.png", -7: "black_pawn1.png",
        }

        for k, name in mapping.items():
            path = os.path.join(ASSETS_DIR, name)

            if not os.path.exists(path):
                print(f"❌ 找不到图片: {path}")
                continue

            img = pygame.image.load(path)
            img = pygame.transform.scale(img, (CELL_SIZE, CELL_SIZE))
            pieces[k] = img

        return pieces

    def draw_text(self, text, x, y, font=None, color=TEXT_COLOR):
        if font is None:
            font = self.font
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, (x, y))

    def draw_button(self, rect, text):
        mouse_pos = pygame.mouse.get_pos()
        color = BUTTON_HOVER if rect.collidepoint(mouse_pos) else BUTTON_COLOR
        pygame.draw.rect(self.screen, color, rect, border_radius=8)
        pygame.draw.rect(self.screen, PANEL_BORDER, rect, width=2, border_radius=8)

        text_surface = self.font.render(text, True, BUTTON_TEXT)
        text_rect = text_surface.get_rect(center=rect.center)
        self.screen.blit(text_surface, text_rect)

    def draw_side_panel(self, board, status_text="", game_over=False):
        panel_rect = pygame.Rect(PANEL_X, BOARD_OFFSET_Y, PANEL_WIDTH, BOARD_HEIGHT)
        pygame.draw.rect(self.screen, PANEL_BG, panel_rect, border_radius=10)
        pygame.draw.rect(self.screen, PANEL_BORDER, panel_rect, width=2, border_radius=10)

        self.draw_text("对局信息", PANEL_X + 18, 72)

        turn_text = "红方走棋" if board.side > 0 else "黑方走棋"
        self.draw_text(turn_text, PANEL_X + 18, 115, self.small_font)

        self.draw_button(self.undo_button_rect, "悔棋")
        self.draw_button(self.save_button_rect, "保存棋谱")

        status_color = STATUS_GOOD if status_text else TEXT_COLOR
        if status_text:
            self.draw_text(status_text, PANEL_X + 18, 305, self.small_font, status_color)

        if game_over:
            self.draw_text("对局结束", PANEL_X + 18, 345, self.small_font, STATUS_GOOD)

        self.draw_text("操作说明", PANEL_X + 18, 405, self.small_font)
        self.draw_text("鼠标点击走棋", PANEL_X + 18, 443, self.tiny_font)
        self.draw_text("点击按钮悔棋", PANEL_X + 18, 473, self.tiny_font)
        self.draw_text("点击按钮保存棋谱", PANEL_X + 18, 503, self.tiny_font)
        self.draw_text("棋谱保存在", PANEL_X + 18, 543, self.tiny_font)
        self.draw_text("根目录/Qipu", PANEL_X + 18, 573, self.tiny_font)

    def draw(self, board, selected=None, possible_moves=None, status_text="", game_over=False):
        self.screen.fill(BG_COLOR)

        self.screen.blit(self.board_img, (BOARD_OFFSET_X, BOARD_OFFSET_Y))

        b = board.board

        for pos in range(90):
            piece = b[pos]
            if piece == 0:
                continue

            x = pos % 9
            y = pos // 9

            px = BOARD_OFFSET_X + x * CELL_SIZE
            py = BOARD_OFFSET_Y + y * CELL_SIZE

            if piece in self.piece_images:
                self.screen.blit(self.piece_images[piece], (px, py))
            else:
                print(f"⚠️ 未加载棋子图片: {piece}")

        if selected is not None:
            x = selected % 9
            y = selected // 9
            rect = pygame.Rect(
                BOARD_OFFSET_X + x * CELL_SIZE,
                BOARD_OFFSET_Y + y * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE,
            )
            pygame.draw.rect(self.screen, SELECT_COLOR, rect, 3)

        if possible_moves:
            for pos in possible_moves:
                x = pos % 9
                y = pos // 9

                center_x = BOARD_OFFSET_X + x * CELL_SIZE + CELL_SIZE // 2
                center_y = BOARD_OFFSET_Y + y * CELL_SIZE + CELL_SIZE // 2

                pygame.draw.circle(self.screen, MOVE_HINT_COLOR, (center_x, center_y), 8)

        self.draw_side_panel(board, status_text=status_text, game_over=game_over)
        pygame.display.flip()

    def get_click_pos(self, mouse_pos):
        mx, my = mouse_pos

        x = (mx - BOARD_OFFSET_X) // CELL_SIZE
        y = (my - BOARD_OFFSET_Y) // CELL_SIZE

        if 0 <= x < 9 and 0 <= y < 10:
            return y * 9 + x

        return None

    def is_undo_button_clicked(self, mouse_pos):
        return self.undo_button_rect.collidepoint(mouse_pos)

    def is_save_button_clicked(self, mouse_pos):
        return self.save_button_rect.collidepoint(mouse_pos)