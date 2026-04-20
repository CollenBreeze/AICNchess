# renderer.py

import os
import pygame

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "pctr")
FONTS_DIR = os.path.join(BASE_DIR, "fonts")
SOUNDS_DIR = os.path.join(BASE_DIR, "sounds")

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
VALUE_BG = (255, 248, 232)


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

        self.undo_button_rect = pygame.Rect(PANEL_X + 10, 160, 140, 42)
        self.save_button_rect = pygame.Rect(PANEL_X + 10, 212, 140, 42)

        self.depth_minus_rect = pygame.Rect(PANEL_X + 10, 294, 40, 38)
        self.depth_value_rect = pygame.Rect(PANEL_X + 54, 294, 52, 38)
        self.depth_plus_rect = pygame.Rect(PANEL_X + 110, 294, 40, 38)

        self.red_ai_button_rect = pygame.Rect(PANEL_X + 10, 352, 140, 40)
        self.black_ai_button_rect = pygame.Rect(PANEL_X + 10, 402, 140, 40)
        self.ai_once_button_rect = pygame.Rect(PANEL_X + 10, 452, 140, 40)
        self.ai_mode_compact_rect = pygame.Rect(PANEL_X + 10, 352, 140, 36)
        self.ai_mode_full_rect = pygame.Rect(PANEL_X + 10, 502, 140, 34)
        self.train_nnue_button_rect = pygame.Rect(PANEL_X + 10, 542, 140, 34)

        self.sounds = {}
        self.sound_enabled = False
        self.init_sounds()

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

    def init_sounds(self):
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
        except Exception as exc:
            print(f"⚠️ 音频初始化失败: {exc}")
            self.sound_enabled = False
            return

        mapping = {
            "pickup": "pickup.wav",
            "place": "place.wav",
            "capture": "capture.wav",
            "check": "check.wav",
            "checkmate": "checkmate.wav",
            "undo": "undo.wav",
            "surrender": "surrender.wav",
        }

        for key, filename in mapping.items():
            path = os.path.join(SOUNDS_DIR, filename)
            if not os.path.exists(path):
                continue
            try:
                self.sounds[key] = pygame.mixer.Sound(path)
            except Exception as exc:
                print(f"⚠️ 加载音效失败 {filename}: {exc}")

        self.sound_enabled = bool(self.sounds)

    def play_sound(self, name):
        if not self.sound_enabled:
            return
        sound = self.sounds.get(name)
        if sound is None:
            return
        try:
            sound.play()
        except Exception:
            pass

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

    def draw_button(self, rect, text, font=None):
        mouse_pos = pygame.mouse.get_pos()
        color = BUTTON_HOVER if rect.collidepoint(mouse_pos) else BUTTON_COLOR
        pygame.draw.rect(self.screen, color, rect, border_radius=8)
        pygame.draw.rect(self.screen, PANEL_BORDER, rect, width=2, border_radius=8)

        if font is None:
            font = self.font

        text_surface = font.render(text, True, BUTTON_TEXT)
        text_rect = text_surface.get_rect(center=rect.center)
        self.screen.blit(text_surface, text_rect)

    def draw_depth_controls(self, depth):
        self.draw_text("搜索深度", PANEL_X + 18, 268, self.small_font)

        self.draw_button(self.depth_minus_rect, "-", self.font)
        self.draw_button(self.depth_plus_rect, "+", self.font)

        pygame.draw.rect(self.screen, VALUE_BG, self.depth_value_rect, border_radius=8)
        pygame.draw.rect(self.screen, PANEL_BORDER, self.depth_value_rect, width=2, border_radius=8)

        value_surface = self.font.render(str(depth), True, TEXT_COLOR)
        value_rect = value_surface.get_rect(center=self.depth_value_rect.center)
        self.screen.blit(value_surface, value_rect)

    def draw_ai_controls(self, red_ai_enabled, black_ai_enabled):
        red_text = f"红方AI：{'开' if red_ai_enabled else '关'}"
        black_text = f"黑方AI：{'开' if black_ai_enabled else '关'}"

        self.draw_button(self.red_ai_button_rect, red_text, self.small_font)
        self.draw_button(self.black_ai_button_rect, black_text, self.small_font)
        self.draw_button(self.ai_once_button_rect, "AI走一步", self.small_font)

    def draw_ai_mode_button(self, ai_mode, compact=False):
        mode = str(ai_mode).lower()
        if mode == "ddqn":
            label = "AI模式：DDQN"
        elif mode == "nnue":
            label = "AI模式：NNUE"
        else:
            label = "AI模式：搜索"
        rect = self.ai_mode_compact_rect if compact else self.ai_mode_full_rect
        font = self.tiny_font if compact else self.small_font
        self.draw_button(rect, label, font)

    def draw_training_button(self):
        self.draw_button(self.train_nnue_button_rect, "训练NNUE", self.small_font)

    def draw_side_panel(
        self,
        board,
        status_text="",
        game_over=False,
        search_depth=4,
        red_ai_enabled=False,
        black_ai_enabled=False,
        ai_mode="search",
        show_depth_controls=False,
        show_ai_controls=False,
        show_ai_mode_button=False,
        show_training_button=False,
    ):
        panel_rect = pygame.Rect(PANEL_X, BOARD_OFFSET_Y, PANEL_WIDTH, BOARD_HEIGHT)
        pygame.draw.rect(self.screen, PANEL_BG, panel_rect, border_radius=10)
        pygame.draw.rect(self.screen, PANEL_BORDER, panel_rect, width=2, border_radius=10)

        self.draw_text("对局信息", PANEL_X + 18, 72)

        turn_text = "红方走棋" if board.side > 0 else "黑方走棋"
        self.draw_text(turn_text, PANEL_X + 18, 115, self.small_font)

        self.draw_button(self.undo_button_rect, "悔棋", self.small_font)
        self.draw_button(self.save_button_rect, "保存棋谱", self.small_font)

        if show_depth_controls:
            self.draw_depth_controls(search_depth)

        if show_ai_controls:
            self.draw_ai_controls(red_ai_enabled, black_ai_enabled)

        if show_ai_mode_button:
            self.draw_ai_mode_button(ai_mode, compact=not show_ai_controls)

        if show_training_button:
            self.draw_training_button()

        if show_ai_controls:
            if show_training_button:
                status_y = 592
                game_over_y = 624
                tips_title_y = 656
                tips = [
                    "鼠标点击走棋",
                    "按钮可切换AI托管/模式",
                    "训练NNUE可打开训练界面",
                ]
            else:
                status_y = 548
                game_over_y = 580
                tips_title_y = 615
                tips = [
                    "鼠标点击走棋",
                    "按钮可切换AI托管/模式",
                ]
        elif show_depth_controls and show_ai_mode_button:
            status_y = 404
            game_over_y = 440
            tips_title_y = 492
            tips = [
                "鼠标点击走棋",
                "点击按钮悔棋",
                "点击按钮保存棋谱",
                "+/- 或 1-8 调深度",
            ]
        elif show_depth_controls:
            status_y = 360
            game_over_y = 395
            tips_title_y = 455
            tips = [
                "鼠标点击走棋",
                "点击按钮悔棋",
                "点击按钮保存棋谱",
                "+/- 或 1-8 调深度",
            ]
        else:
            status_y = 305
            game_over_y = 345
            tips_title_y = 405
            tips = [
                "鼠标点击走棋",
                "点击按钮悔棋",
                "点击按钮保存棋谱",
                "棋谱保存在根目录/Qipu",
            ]

        status_color = STATUS_GOOD if status_text else TEXT_COLOR
        if status_text:
            self.draw_text(status_text, PANEL_X + 18, status_y, self.small_font, status_color)

        if game_over:
            self.draw_text("对局结束", PANEL_X + 18, game_over_y, self.small_font, STATUS_GOOD)

        self.draw_text("操作说明", PANEL_X + 18, tips_title_y, self.small_font)
        line_y = tips_title_y + 32
        for tip in tips:
            self.draw_text(tip, PANEL_X + 18, line_y, self.tiny_font)
            line_y += 22

    def draw(
        self,
        board,
        selected=None,
        possible_moves=None,
        status_text="",
        game_over=False,
        search_depth=4,
        red_ai_enabled=False,
        black_ai_enabled=False,
        ai_mode="search",
        show_depth_controls=False,
        show_ai_controls=False,
        show_ai_mode_button=False,
        show_training_button=False,
    ):
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

        self.draw_side_panel(
            board,
            status_text=status_text,
            game_over=game_over,
            search_depth=search_depth,
            red_ai_enabled=red_ai_enabled,
            black_ai_enabled=black_ai_enabled,
            ai_mode=ai_mode,
            show_depth_controls=show_depth_controls,
            show_ai_controls=show_ai_controls,
            show_ai_mode_button=show_ai_mode_button,
            show_training_button=show_training_button,
        )
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

    def is_depth_minus_clicked(self, mouse_pos):
        return self.depth_minus_rect.collidepoint(mouse_pos)

    def is_depth_plus_clicked(self, mouse_pos):
        return self.depth_plus_rect.collidepoint(mouse_pos)

    def is_red_ai_button_clicked(self, mouse_pos):
        return self.red_ai_button_rect.collidepoint(mouse_pos)

    def is_black_ai_button_clicked(self, mouse_pos):
        return self.black_ai_button_rect.collidepoint(mouse_pos)

    def is_ai_once_button_clicked(self, mouse_pos):
        return self.ai_once_button_rect.collidepoint(mouse_pos)

    def is_ai_mode_button_clicked(self, mouse_pos, compact=False):
        rect = self.ai_mode_compact_rect if compact else self.ai_mode_full_rect
        return rect.collidepoint(mouse_pos)

    def is_train_nnue_button_clicked(self, mouse_pos):
        return self.train_nnue_button_rect.collidepoint(mouse_pos)
