# -*- coding: utf-8 -*-
"""
GUI 共享模块

提取 main.py 和 main_enhanced.py 的重复代码:
- AIManager (模型加载和预测)
- resource_path (资源路径解析)
- 字体注册
- GameGrid (统一游戏网格，支持所有模式)
"""

import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from kivy.clock import Clock
from kivy.core.text import LabelBase
from kivy.graphics import Color, Rectangle
from kivy.properties import BooleanProperty, StringProperty
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button

from src.ai.train import NeuralNetwork
from src.ai.train import board_to_input as nn_board_to_input
from src.core.checkpoint import CheckpointManager, GameState
from src.core.common import GameBoard
from src.core.config import BEST_MODEL_PATH, DEFAULT_MODEL_PATH


# ==================== 资源路径 ====================


def resource_path(relative_path: str) -> str:
    """获取资源文件的绝对路径（兼容 PyInstaller）"""
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base_path, relative_path)


MODEL_PATH = BEST_MODEL_PATH if os.path.exists(BEST_MODEL_PATH) else DEFAULT_MODEL_PATH
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = resource_path("best_model.pkl")

FONT_PATH = resource_path("assets/fonts/SourceHanSansSC-Normal-2.otf")
FONT_NAME = "SourceHanSans"

try:
    LabelBase.register(name=FONT_NAME, fn_regular=FONT_PATH)
except Exception:
    FONT_NAME = "Roboto"


# ==================== AIManager ====================


class AIManager:
    """AI 管理器 — 加载模型并预测移动"""

    def __init__(self, model_path: str):
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str) -> NeuralNetwork:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
            if not hasattr(model, "get_best_move"):
                raise TypeError("模型缺少 get_best_move 方法")
            return model

    def predict_move(self, board: list, epsilon: float = 0.0) -> int | None:
        if not self.model:
            return None
        try:
            inputs = nn_board_to_input(board)
            self.model.forward(inputs)
            available = [
                i for i in range(9) if board[i // 3][i % 3] is None
            ]
            if epsilon > 0.0 and available and np.random.rand() < epsilon:
                return available[np.random.randint(len(available))]
            best_move = -1
            best_prob = -1.0
            for i in available:
                if self.model.outputs[i] > best_prob:
                    best_prob = self.model.outputs[i]
                    best_move = i
            return best_move
        except Exception as e:
            print(f"AI预测失败: {e}")
            return None


# ==================== GameGrid ====================


class GameGrid(GridLayout):
    """统一游戏棋盘网格 — 支持 pvp/pve/eve 模式"""

    game_mode = StringProperty("pve")
    ai_first = BooleanProperty(True)
    ai_difficulty = StringProperty("medium")
    enable_checkpoint = BooleanProperty(True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cols = 3
        self.padding = 5
        self.spacing = 5
        self.board = GameBoard()
        self.buttons = []
        self.game_over_flag = False
        self.move_history = []
        self.ai_manager = None
        self._ai_enabled = False

        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir="checkpoints/game", max_checkpoints=10, verbose=False
        )

        self.player_symbols = {0: "X", 1: "O"}
        self.ai_player = 0 if self.ai_first else 1
        self.human_player = 1 - self.ai_player

        for i in range(self.board.board_size):
            btn = Button(text="", font_size="40sp", font_name=FONT_NAME)
            btn.bind(on_press=self.on_button_press)
            self.add_widget(btn)
            self.buttons.append(btn)

    def initialize_game(
        self,
        game_mode: str = "pve",
        ai_first: bool = True,
        difficulty: str = "medium",
        model_path: str = None,
    ):
        self.game_mode = game_mode
        self.ai_first = ai_first
        self.ai_difficulty = difficulty
        self.ai_player = 0 if ai_first else 1
        self.human_player = 1 - self.ai_player

        self.reset_board()

        if game_mode in ["pve", "eve"]:
            try:
                path = model_path or MODEL_PATH
                self.ai_manager = AIManager(path)
                self._ai_enabled = True
            except Exception as e:
                print(f"AI加载失败: {e}")
                self._ai_enabled = False
                if game_mode != "pvp":
                    self.show_error("AI加载失败，切换到人人对战模式")
                    self.game_mode = "pvp"

        if self.game_mode == "pve" and self.ai_first and self._ai_enabled:
            self.board.current_player = self.ai_player
            Clock.schedule_once(self.ai_move_delayed, 0.5)
        elif self.game_mode == "eve" and self._ai_enabled:
            self.board.current_player = 0
            Clock.schedule_once(self.ai_vs_ai_move, 0.5)
        else:
            self.board.current_player = 0
            self.update_status("游戏开始！X先手")

    def reset_board(self):
        self.board.reset_board()
        self.game_over_flag = False
        self.move_history = []
        for btn in self.buttons:
            btn.text = ""
            btn.disabled = False

    def on_button_press(self, instance):
        if self.game_over_flag:
            return

        index = self.buttons.index(instance)
        row, col = index // 3, index % 3
        current_symbol = self.player_symbols[self.board.current_player]

        if self.game_mode == "pvp":
            self.make_move(row, col, current_symbol)
        elif self.game_mode == "pve":
            if self.board.current_player == self.human_player:
                self.make_move(row, col, current_symbol)

    def make_move(self, row: int, col: int, symbol: str):
        index = row * 3 + col

        if self.buttons[index].text != "" or self.board.board[row][col] is not None:
            return

        self.buttons[index].text = symbol
        self.board.board[row][col] = symbol

        self.move_history.append(
            {
                "player": self.board.current_player,
                "symbol": symbol,
                "position": (row, col),
            }
        )

        if self.enable_checkpoint:
            self.save_checkpoint()

        if self.check_game_status():
            return

        self.board.current_player = 1 - self.board.current_player
        next_symbol = self.player_symbols[self.board.current_player]

        if self.game_mode == "pve":
            if self.board.current_player == self.ai_player and self._ai_enabled:
                self.update_status(f"AI ({next_symbol}) 正在思考...")
                Clock.schedule_once(self.ai_move_delayed, 0.5)
            else:
                self.update_status(f"轮到您 ({next_symbol})")
        elif self.game_mode == "eve" and self._ai_enabled:
            Clock.schedule_once(self.ai_vs_ai_move, 0.5)
        else:
            self.update_status(f"轮到玩家 {next_symbol}")

    def ai_move_delayed(self, dt):
        self.ai_move()

    def _get_epsilon(self) -> float:
        if self.ai_difficulty == "easy":
            return 0.3
        elif self.ai_difficulty == "medium":
            return 0.1
        return 0.0

    def ai_move(self):
        if self.game_over_flag or not self._ai_enabled:
            return

        epsilon = self._get_epsilon()
        move_index = self.ai_manager.predict_move(self.board.board, epsilon=epsilon)

        if move_index is not None:
            row, col = move_index // 3, move_index % 3
            symbol = self.player_symbols[self.board.current_player]
            self.make_move(row, col, symbol)

    def ai_vs_ai_move(self, dt):
        if self.game_over_flag or not self._ai_enabled:
            return

        move_index = self.ai_manager.predict_move(self.board.board, epsilon=0.0)

        if move_index is not None:
            row, col = move_index // 3, move_index % 3
            symbol = self.player_symbols[self.board.current_player]
            self.make_move(row, col, symbol)

            if not self.game_over_flag:
                Clock.schedule_once(self.ai_vs_ai_move, 0.5)

    def check_game_status(self) -> bool:
        board = self.board.board
        winner = None

        for r in range(3):
            if board[r][0] is not None and board[r][0] == board[r][1] == board[r][2]:
                winner = board[r][0]
                break

        if winner is None:
            for c in range(3):
                if (
                    board[0][c] is not None
                    and board[0][c] == board[1][c] == board[2][c]
                ):
                    winner = board[0][c]
                    break

        if winner is None:
            center = board[1][1]
            if center is not None:
                if board[0][0] == center == board[2][2]:
                    winner = center
                elif board[0][2] == center == board[2][0]:
                    winner = center

        if winner:
            self.game_over(winner)
            return True

        is_full = all(cell is not None for row in board for cell in row)
        if is_full:
            self.game_over(None)
            return True

        return False

    def game_over(self, winner: str | None):
        self.game_over_flag = True
        for btn in self.buttons:
            btn.disabled = True

        if winner:
            status_text = f"游戏结束！{winner} 获胜！"
        else:
            status_text = "游戏结束！平局！"

        self.update_status(status_text)

    def save_checkpoint(self):
        try:
            state = GameState(
                board=self.board.board,
                current_player=self.player_symbols[self.board.current_player],
                move_history=self.move_history,
                game_mode=self.game_mode,
                ai_difficulty=self.ai_difficulty,
                is_game_over=self.game_over_flag,
                winner=None,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
            self.checkpoint_manager.save_game_checkpoint(
                state, filename="latest_game.pkl"
            )
        except Exception as e:
            print(f"保存检查点失败: {e}")

    def load_checkpoint(self) -> bool:
        try:
            checkpoints = self.checkpoint_manager.list_checkpoints("game")
            if not checkpoints.get("game"):
                return False

            state = self.checkpoint_manager.load_game_checkpoint()

            self.board.board = state.board
            self.board.current_player = 0 if state.current_player == "X" else 1
            self.move_history = state.move_history
            self.game_mode = state.game_mode
            self.ai_difficulty = state.ai_difficulty
            self.game_over_flag = state.is_game_over

            for i, btn in enumerate(self.buttons):
                row, col = i // 3, i % 3
                cell = self.board.board[row][col]
                btn.text = cell if cell else ""
                btn.disabled = self.game_over_flag

            return True
        except Exception as e:
            print(f"加载检查点失败: {e}")
            return False

    def update_status(self, message: str):
        if self.parent and hasattr(self.parent, "status_label"):
            self.parent.status_label.text = message

    def show_error(self, message: str):
        print(f"错误: {message}")
