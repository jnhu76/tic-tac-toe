# -*- coding: utf-8 -*-
"""
增强版井字棋游戏主程序 - GUI界面

本模块在原有功能基础上增加了：
1. 检查点功能：支持游戏中断恢复
2. 先手后手选择：玩家可以选择先手或后手
3. 游戏模式选择：人机对战、人人对战、机机对战
4. 难度选择：简单、中等、困难

技术栈：
- Kivy：跨平台GUI框架
- NumPy：数值计算
- Pickle：模型加载

作者：AI Assistant
版本：3.0
日期：2026-04-13
"""

import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kivy.app import App
from kivy.clock import Clock
from kivy.core.text import LabelBase
from kivy.graphics import Color, Rectangle
from kivy.properties import BooleanProperty, ObjectProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.spinner import Spinner

from src.ai.train import NeuralNetwork
from src.ai.train import board_to_input as nn_board_to_input
from src.core.checkpoint import CheckpointManager, GameState
from src.core.common import GameBoard
from src.core.config import BEST_MODEL_PATH, DEFAULT_MODEL_PATH


def resource_path(relative_path):
    """获取资源文件的绝对路径"""
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base_path, relative_path)


# ==================== 资源路径配置 ====================
# 优先使用统一配置中的模型路径，如果不存在则使用默认路径
MODEL_PATH = BEST_MODEL_PATH if os.path.exists(BEST_MODEL_PATH) else DEFAULT_MODEL_PATH
# 如果统一配置路径都不存在，使用本地路径作为后备
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = resource_path("best_model.pkl")

FONT_PATH = resource_path("assets/fonts/SourceHanSansSC-Normal-2.otf")
FONT_NAME = "SourceHanSans"

# ==================== 注册自定义字体 ====================
try:
    LabelBase.register(name=FONT_NAME, fn_regular=FONT_PATH)
except Exception as e:
    print(f"警告: 无法注册字体 {FONT_NAME}。错误: {e}")
    FONT_NAME = "Roboto"


class AIManager:
    """AI管理器类"""

    def __init__(self, model_path: str):
        self.model = self.load_model(model_path)

    def load_model(self, model_path: str):
        """加载神经网络模型"""
        try:
            print(f"加载模型: {model_path}")
            with open(model_path, "rb") as f:
                model = pickle.load(f)
                if not hasattr(model, "get_best_move"):
                    raise TypeError("模型缺少'get_best_move'方法")
                return model
        except Exception as e:
            print(f"加载模型失败: {e}")
            raise

    def predict_move(
        self, board: list, strategy: str = "greedy", epsilon: float = 0.0
    ) -> int:
        if not self.model:
            return None
        try:
            inputs = self.board_to_input(board)
            if strategy == "greedy" or epsilon == 0.0:
                return self.model.get_best_move(inputs, board)
            else:
                available = [
                    (i // 3, i % 3) for i in range(9) if board[i // 3][i % 3] is None
                ]
                if np.random.rand() < epsilon and available:
                    chosen = available[np.random.randint(len(available))]
                    return chosen[0] * 3 + chosen[1]
                return self.model.get_best_move(inputs, board)
        except Exception as e:
            print(f"AI预测失败: {e}")
            return None

    def board_to_input(
        self, board_state: list, col: int = 3, row: int = 3
    ) -> np.ndarray:
        return nn_board_to_input(board_state)


class GameGrid(GridLayout):
    """增强版游戏棋盘网格类"""

    # Kivy属性
    game_mode = StringProperty("pve")  # 'pvp', 'pve', 'eve'
    ai_first = BooleanProperty(True)  # AI是否先手
    ai_difficulty = StringProperty("medium")  # 'easy', 'medium', 'hard'
    enable_checkpoint = BooleanProperty(True)

    def __init__(self, **kwargs):
        super(GameGrid, self).__init__(**kwargs)
        self.cols = 3
        self.padding = 5
        self.spacing = 5
        self.board = GameBoard()
        self.buttons = []
        self.game_over_flag = False
        self.move_history = []

        # AI管理器
        self.ai_manager = None
        self._ai_enabled = False

        # 检查点管理器
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir="checkpoints/game", max_checkpoints=10, verbose=False
        )

        # 游戏配置
        self.player_symbols = {0: "X", 1: "O"}
        self.ai_player = 0 if self.ai_first else 1  # AI作为玩家0或1
        self.human_player = 1 - self.ai_player

        # 创建按钮
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
        """
        初始化游戏

        参数:
            game_mode (str): 游戏模式 ('pvp', 'pve', 'eve')
            ai_first (bool): AI是否先手
            difficulty (str): AI难度
            model_path (str): AI模型路径
        """
        self.game_mode = game_mode
        self.ai_first = ai_first
        self.ai_difficulty = difficulty
        self.ai_player = 0 if ai_first else 1
        self.human_player = 1 - self.ai_player

        # 重置游戏状态
        self.reset_board()

        # 加载AI（如果需要）
        if game_mode in ["pve", "eve"]:
            try:
                path = model_path or MODEL_PATH
                self.ai_manager = AIManager(path)
                self._ai_enabled = True
                print(f"AI加载成功，难度: {difficulty}, AI先手: {ai_first}")
            except Exception as e:
                print(f"AI加载失败: {e}")
                self._ai_enabled = False
                if game_mode != "pvp":
                    self.show_error("AI加载失败，切换到人人对战模式")
                    self.game_mode = "pvp"

        # 根据游戏模式开始游戏
        if self.game_mode == "pve" and self.ai_first and self._ai_enabled:
            # AI先手
            self.board.current_player = self.ai_player
            Clock.schedule_once(self.ai_move_delayed, 0.5)
        elif self.game_mode == "eve" and self._ai_enabled:
            # 机机对战
            self.board.current_player = 0
            Clock.schedule_once(self.ai_vs_ai_move, 0.5)
        else:
            # 人类先手或人人对战
            self.board.current_player = 0
            self.update_status("游戏开始！X先手")

    def reset_board(self):
        """重置棋盘"""
        self.board.reset_board()
        self.game_over_flag = False
        self.move_history = []
        for btn in self.buttons:
            btn.text = ""
            btn.disabled = False

    def on_button_press(self, instance):
        """处理按钮点击"""
        if self.game_over_flag:
            return

        index = self.buttons.index(instance)
        row, col = index // 3, index % 3

        # 检查是否是当前玩家的回合
        current_symbol = self.player_symbols[self.board.current_player]

        # 根据游戏模式处理
        if self.game_mode == "pvp":
            # 人人对战
            self.make_move(row, col, current_symbol)
        elif self.game_mode == "pve":
            # 人机对战
            if self.board.current_player == self.human_player:
                self.make_move(row, col, current_symbol)
            else:
                print("还没轮到您")

    def make_move(self, row: int, col: int, symbol: str):
        """
        执行移动

        参数:
            row (int): 行索引
            col (int): 列索引
            symbol (str): 玩家标记 ('X' 或 'O')
        """
        index = row * 3 + col

        # 检查位置是否为空
        if self.buttons[index].text != "" or self.board.board[row][col] is not None:
            return

        # 更新UI和棋盘状态
        self.buttons[index].text = symbol
        self.board.board[row][col] = symbol

        # 记录移动历史
        self.move_history.append(
            {
                "player": self.board.current_player,
                "symbol": symbol,
                "position": (row, col),
            }
        )

        # 保存检查点
        if self.enable_checkpoint:
            self.save_checkpoint()

        # 检查游戏状态
        if self.check_game_status():
            return

        # 切换玩家
        self.board.current_player = 1 - self.board.current_player
        next_symbol = self.player_symbols[self.board.current_player]

        # 根据游戏模式处理下一步
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
        """延迟执行AI移动（用于UI更新）"""
        self.ai_move()

    def ai_move(self):
        """执行AI移动"""
        if self.game_over_flag or not self._ai_enabled:
            return

        # 根据难度选择策略
        epsilon = 0.0
        if self.ai_difficulty == "easy":
            epsilon = 0.3
        elif self.ai_difficulty == "medium":
            epsilon = 0.1

        # 获取AI移动
        move_index = self.ai_manager.predict_move(
            self.board.board,
            strategy="exploration" if epsilon > 0 else "greedy",
            epsilon=epsilon,
        )

        if move_index is not None:
            row, col = move_index // 3, move_index % 3
            symbol = self.player_symbols[self.board.current_player]
            self.make_move(row, col, symbol)

    def ai_vs_ai_move(self, dt):
        """机机对战的移动"""
        if self.game_over_flag or not self._ai_enabled:
            return

        # 两个AI都使用贪心策略
        move_index = self.ai_manager.predict_move(
            self.board.board, strategy="greedy", epsilon=0.0
        )

        if move_index is not None:
            row, col = move_index // 3, move_index % 3
            symbol = self.player_symbols[self.board.current_player]
            self.make_move(row, col, symbol)

            # 继续下一步（如果游戏未结束）
            if not self.game_over_flag:
                Clock.schedule_once(self.ai_vs_ai_move, 0.5)

    def check_game_status(self) -> bool:
        """
        检查游戏状态

        返回:
            bool: 游戏是否结束
        """
        board = self.board.board
        winner = None

        # 检查行
        for r in range(3):
            if board[r][0] is not None and board[r][0] == board[r][1] == board[r][2]:
                winner = board[r][0]
                break

        # 检查列
        if winner is None:
            for c in range(3):
                if (
                    board[0][c] is not None
                    and board[0][c] == board[1][c] == board[2][c]
                ):
                    winner = board[0][c]
                    break

        # 检查对角线
        if winner is None:
            center = board[1][1]
            if center is not None:
                if board[0][0] == center == board[2][2]:
                    winner = center
                elif board[0][2] == center == board[2][0]:
                    winner = center

        # 处理获胜
        if winner:
            self.game_over(winner)
            return True

        # 检查平局
        is_full = all(cell is not None for row in board for cell in row)
        if is_full:
            self.game_over(None)
            return True

        return False

    def game_over(self, winner: str):
        """
        处理游戏结束

        参数:
            winner (str): 获胜者标记，None表示平局
        """
        self.game_over_flag = True

        for btn in self.buttons:
            btn.disabled = True

        if winner:
            status_text = f"游戏结束！{winner} 获胜！"
        else:
            status_text = "游戏结束！平局！"

        self.update_status(status_text)
        print(status_text)

    def save_checkpoint(self):
        """保存游戏检查点"""
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
        """
        加载游戏检查点

        返回:
            bool: 是否成功加载
        """
        try:
            checkpoints = self.checkpoint_manager.list_checkpoints("game")
            if not checkpoints.get("game"):
                print("没有可用的检查点")
                return False

            # 加载最新的检查点
            state = self.checkpoint_manager.load_game_checkpoint()

            # 恢复游戏状态
            self.board.board = state.board
            self.board.current_player = 0 if state.current_player == "X" else 1
            self.move_history = state.move_history
            self.game_mode = state.game_mode
            self.ai_difficulty = state.ai_difficulty
            self.game_over_flag = state.is_game_over

            # 更新UI
            for i, btn in enumerate(self.buttons):
                row, col = i // 3, i % 3
                cell = self.board.board[row][col]
                btn.text = cell if cell else ""
                btn.disabled = self.game_over_flag

            print(f"检查点已加载，当前玩家: {state.current_player}")
            return True

        except Exception as e:
            print(f"加载检查点失败: {e}")
            return False

    def update_status(self, message: str):
        """更新状态标签"""
        if self.parent and hasattr(self.parent, "status_label"):
            self.parent.status_label.text = message

    def show_error(self, message: str):
        """显示错误信息"""
        print(f"错误: {message}")


class GameControls(BoxLayout):
    """增强版游戏控制面板类"""

    def __init__(self, **kwargs):
        super(GameControls, self).__init__(**kwargs)
        self.orientation = "horizontal"
        self.padding = 10
        self.spacing = 10

        # 设置背景
        with self.canvas.before:
            Color(0.95, 0.95, 0.95, 1)
            self._background_rect = Rectangle(size=self.size, pos=self.pos)
        self.bind(size=self._update_background, pos=self._update_background)

        # 左侧：游戏棋盘
        self.game_grid = GameGrid(size_hint=(0.6, 1))
        self.add_widget(self.game_grid)

        # 右侧：控制面板
        control_panel = BoxLayout(
            orientation="vertical", size_hint=(0.4, 1), spacing=10, padding=10
        )

        # 标题
        control_panel.add_widget(
            Label(
                text="[b]井字棋 AI 对战[/b]",
                markup=True,
                font_size="24sp",
                font_name=FONT_NAME,
                color=[0.1, 0.1, 0.1, 1],
                size_hint_y=None,
                height=50,
            )
        )

        # 状态标签
        self.status_label = Label(
            font_size="18sp",
            font_name=FONT_NAME,
            color=[0.2, 0.2, 0.2, 1],
            size_hint=(1, 0.2),
        )
        control_panel.add_widget(self.status_label)

        # 游戏模式选择
        control_panel.add_widget(
            Label(
                text="游戏模式:",
                font_size="14sp",
                font_name=FONT_NAME,
                size_hint_y=None,
                height=30,
            )
        )

        self.mode_spinner = Spinner(
            text="人机对战",
            values=["人机对战", "人人对战", "机机对战"],
            font_name=FONT_NAME,
            size_hint_y=None,
            height=40,
        )
        control_panel.add_widget(self.mode_spinner)

        # 先手选择
        control_panel.add_widget(
            Label(
                text="先手选择:",
                font_size="14sp",
                font_name=FONT_NAME,
                size_hint_y=None,
                height=30,
            )
        )

        self.first_spinner = Spinner(
            text="玩家先手",
            values=["玩家先手", "AI先手"],
            font_name=FONT_NAME,
            size_hint_y=None,
            height=40,
        )
        control_panel.add_widget(self.first_spinner)

        # 难度选择
        control_panel.add_widget(
            Label(
                text="AI难度:",
                font_size="14sp",
                font_name=FONT_NAME,
                size_hint_y=None,
                height=30,
            )
        )

        self.difficulty_spinner = Spinner(
            text="中等",
            values=["简单", "中等", "困难"],
            font_name=FONT_NAME,
            size_hint_y=None,
            height=40,
        )
        control_panel.add_widget(self.difficulty_spinner)

        # 按钮区域
        button_layout = BoxLayout(
            orientation="vertical", spacing=5, size_hint_y=None, height=150
        )

        # 开始游戏按钮
        start_btn = Button(
            text="开始游戏",
            font_size="16sp",
            font_name=FONT_NAME,
            background_color=[0.2, 0.6, 0.8, 1],
            color=[1, 1, 1, 1],
        )
        start_btn.bind(on_press=self.start_game)
        button_layout.add_widget(start_btn)

        # 加载游戏按钮
        load_btn = Button(
            text="加载游戏",
            font_size="16sp",
            font_name=FONT_NAME,
            background_color=[0.4, 0.7, 0.4, 1],
            color=[1, 1, 1, 1],
        )
        load_btn.bind(on_press=self.load_game)
        button_layout.add_widget(load_btn)

        # 重置游戏按钮
        reset_btn = Button(
            text="重置游戏",
            font_size="16sp",
            font_name=FONT_NAME,
            background_color=[0.8, 0.4, 0.4, 1],
            color=[1, 1, 1, 1],
        )
        reset_btn.bind(on_press=self.reset_game)
        button_layout.add_widget(reset_btn)

        control_panel.add_widget(button_layout)

        # 添加弹性空间
        control_panel.add_widget(Label(size_hint_y=1))

        self.add_widget(control_panel)

        # 初始化状态
        self.status_label.text = "选择游戏模式并开始"

    def _update_background(self, instance, value):
        """更新背景"""
        self._background_rect.pos = instance.pos
        self._background_rect.size = instance.size

    def start_game(self, instance):
        """开始新游戏"""
        # 获取配置
        mode_map = {"人机对战": "pve", "人人对战": "pvp", "机机对战": "eve"}
        difficulty_map = {"简单": "easy", "中等": "medium", "困难": "hard"}

        game_mode = mode_map.get(self.mode_spinner.text, "pve")
        ai_first = self.first_spinner.text == "AI先手"
        difficulty = difficulty_map.get(self.difficulty_spinner.text, "medium")

        # 初始化游戏
        self.game_grid.initialize_game(
            game_mode=game_mode, ai_first=ai_first, difficulty=difficulty
        )

        self.status_label.text = "游戏开始！"

    def load_game(self, instance):
        """加载游戏"""
        if self.game_grid.load_checkpoint():
            self.status_label.text = "游戏已加载"
        else:
            self.status_label.text = "没有可加载的游戏"

    def reset_game(self, instance):
        """重置游戏"""
        self.game_grid.reset_board()
        self.status_label.text = "游戏已重置，请点击开始游戏"


class TicTacToeApp(App):
    """井字棋应用程序类"""

    def build(self):
        self.title = "井字棋 AI 对战 - 增强版"
        return GameControls()


if __name__ == "__main__":
    TicTacToeApp().run()
