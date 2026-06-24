# -*- coding: utf-8 -*-
"""
井字棋游戏主程序 - GUI界面

本模块实现了井字棋游戏的图形用户界面，基于Kivy框架开发，包含以下功能：
1. 游戏界面：3x3棋盘，支持鼠标点击交互
2. AI对战：集成神经网络AI，AI先手（X），玩家后手（O）
3. 游戏状态管理：实时检测胜负平，更新游戏状态
4. 界面美化：自定义字体、颜色和布局

技术栈：
- Kivy：跨平台GUI框架
- NumPy：数值计算
- Pickle：模型加载

作者：AI Assistant
版本：2.0
日期：2026-04-13
"""

import os
import pickle
import sys
from pathlib import Path

import numpy as np

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kivy.app import App
from kivy.clock import Clock
from kivy.core.text import LabelBase
from kivy.graphics import Color, Rectangle
from kivy.properties import ObjectProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label

from src.ai.train import NeuralNetwork
from src.ai.train import board_to_input as nn_board_to_input
from src.core.common import GameBoard
from src.core.config import BEST_MODEL_PATH, DEFAULT_MODEL_PATH


def resource_path(relative_path):
    """
    获取资源文件的绝对路径

    兼容开发环境和PyInstaller打包后的环境
    PyInstaller打包后，资源文件会被解压到临时目录_MEIPASS

    参数:
        relative_path (str): 相对路径

    返回:
        str: 绝对路径
    """
    try:
        # PyInstaller创建临时文件夹并存储路径在_MEIPASS中
        base_path = sys._MEIPASS
    except AttributeError:
        # 不在PyInstaller打包环境中运行
        base_path = os.path.abspath(os.path.dirname(__file__))

    return os.path.join(base_path, relative_path)


# ==================== 资源路径配置 ====================
# 优先使用统一配置中的模型路径，如果不存在则使用默认路径
MODEL_PATH = (
    DEFAULT_MODEL_PATH if os.path.exists(DEFAULT_MODEL_PATH) else BEST_MODEL_PATH
)
# 如果统一配置路径都不存在，使用本地路径作为后备
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = resource_path("tic_tac_toe_selfplay_nn_v1.pkl")

FONT_PATH = resource_path("assets/fonts/SourceHanSansSC-Normal-2.otf")  # 中文字体路径
FONT_NAME = "SourceHanSans"  # 字体注册名称

# ==================== 注册自定义字体 ====================
try:
    LabelBase.register(name=FONT_NAME, fn_regular=FONT_PATH)
except Exception as e:
    print(f"警告: 无法注册字体 {FONT_NAME} 从 {FONT_PATH}。错误: {e}")
    FONT_NAME = "Roboto"  # 回退到默认字体


class AIManager:
    """
    AI管理器类

    负责加载神经网络模型并进行移动预测
    """

    def __init__(self, model_path: str):
        """
        初始化AI管理器

        参数:
            model_path (str): 模型文件路径
        """
        self.model = self.load_model(model_path)

    def load_model(self, model_path: str):
        """
        加载神经网络模型

        参数:
            model_path (str): 模型文件路径

        返回:
            NeuralNetwork: 加载的神经网络模型

        异常:
            FileNotFoundError: 模型文件不存在
            TypeError: 加载的对象不是有效的模型
        """
        try:
            print(f"尝试从以下路径加载模型: {model_path}")
            with open(model_path, "rb") as f:
                model = pickle.load(f)
                print(f"成功加载模型，类型: {type(model)}")
                if not hasattr(model, "get_best_move"):
                    print(f"错误: 加载的对象没有'get_best_move'方法。")
                    raise TypeError("加载的模型对象缺少'get_best_move'方法。")
                return model
        except FileNotFoundError:
            print(f"错误: 模型文件未找到: {model_path}")
            raise
        except Exception as e:
            print(f"加载模型时出错: {e}")
            raise

    def predict_move(self, board: list[list[str | None]]) -> int | None:
        if not self.model:
            print("错误: AI模型未加载。")
            return None
        try:
            inputs = nn_board_to_input(board)
            move_index = self.model.get_best_move(inputs, board)
            return move_index
        except Exception as e:
            print(f"AI预测过程中出错: {e}")
            return None

    def board_to_input(
        self, GameState: list[list[str | None]], col: int = 3, row: int = 3
    ) -> np.ndarray:
        return nn_board_to_input(GameState)


class GameGrid(GridLayout):
    """
    游戏棋盘网格类

    实现3x3的井字棋棋盘，处理玩家点击和AI移动
    """

    def __init__(self, **kwargs):
        """
        初始化游戏棋盘

        创建9个按钮，初始化AI管理器，设置AI先手
        """
        super(GameGrid, self).__init__(**kwargs)
        self.cols = 3
        self.padding = 5
        self.spacing = 5
        self.board = GameBoard()
        self.buttons = []
        self._ai_enabled = False
        self.game_over_flag = False  # 游戏结束标志，防止结束后继续移动

        # --- 初始化AI管理器 ---
        try:
            self.ai_manager = AIManager(MODEL_PATH)
            self._ai_enabled = True
            print("AI管理器初始化成功。")
        except Exception as e:
            print(f"AI管理器初始化失败: {e}")
            self.ai_manager = None

        # --- 创建按钮 ---
        for i in range(self.board.board_size):
            btn = Button(text="", font_size="40sp")
            btn.bind(on_press=self.on_button_press)
            self.add_widget(btn)
            self.buttons.append(btn)

        # --- 设置初始游戏状态并触发AI第一步 ---
        # AI ('X') 是玩家0，人类 ('O') 是玩家1
        self.board.current_player = 0  # *** AI ('X') 先手 ***
        self.game_over_flag = False

        # 稍微延迟触发AI的第一步移动，避免UI构建过程中的问题
        # 只有在AI实际启用时才调度
        if self._ai_enabled:
            Clock.schedule_once(self.trigger_initial_ai_move, 0.1)  # 延迟0.1秒
        else:
            # 如果AI失败，游戏无法正常开始
            self.disable_buttons()

    def trigger_initial_ai_move(self, dt):
        """
        安全地触发AI的初始移动

        在初始化完成后调用，避免过早访问组件

        参数:
            dt: Clock调度的时间增量（未使用）
        """
        print("触发AI初始移动...")
        self.ai_move()
        # AI移动后更新状态
        if (
            not self.game_over_flag
            and self.parent
            and hasattr(self.parent, "status_label")
        ):
            self.parent.status_label.text = "轮到你了 ('O')"

    def on_button_press(self, instance):
        """
        处理按钮点击事件

        参数:
            instance: 被点击的按钮实例
        """
        if self.game_over_flag:
            print("游戏已结束。忽略点击。")
            return

        if not self._ai_enabled:
            print("AI不可用。")
            return

        # 检查是否轮到人类玩家（玩家1）
        if self.board.current_player == 1:
            index = self.buttons.index(instance)
            row, col = index // 3, index % 3

            if instance.text == "" and self.board.board[row][col] is None:
                print(f"玩家'O'点击了按钮{index} ([{row}][{col}])")
                instance.text = "O"
                self.board.board[row][col] = "O"

                if not self.check_game_status():  # 检查游戏是否结束
                    self.board.current_player = 0  # 切换到AI回合
                    self.update_status("AI ('X') 正在思考...")
                    # 调度AI移动，允许UI先更新状态
                    Clock.schedule_once(lambda dt: self.ai_move(), 0.1)

            elif instance.text != "":
                print(f"按钮{index}已被占用({instance.text})。")
            elif self.board.board[row][col] is not None:
                print(
                    f"按钮{index}处的棋盘/按钮不匹配。棋盘: {self.board.board[row][col]}, 按钮: {instance.text}"
                )
                instance.text = self.board.board[row][col]  # 同步按钮

        elif self.board.current_player == 0:
            print("还没轮到你（AI正在思考）。")
        else:
            print(f"意外的玩家索引: {self.board.current_player}")

    def ai_move(self):
        """
        执行AI移动

        使用神经网络预测最佳移动，更新棋盘状态
        """
        print("尝试AI移动...")
        if self.game_over_flag:
            print("AI移动跳过: 游戏已结束。")
            return

        # AI只在启用且轮到它时移动（玩家0）
        if self._ai_enabled and self.ai_manager and self.board.current_player == 0:
            move_index = self.ai_manager.predict_move(self.board.board)
            print(f"AI预测的移动索引: {move_index}")

            valid_move_made = False
            if move_index is not None and 0 <= move_index < len(self.buttons):
                # 检查选择的格子是否确实为空
                if self.buttons[move_index].text == "":
                    row, col = move_index // 3, move_index % 3
                    if self.board.board[row][col] is None:
                        print(f"AI在索引{move_index} ([{row}][{col}])处放置'X'")
                        self.buttons[move_index].text = "X"
                        self.board.board[row][col] = "X"
                        valid_move_made = True
                        if not self.check_game_status():  # 检查游戏是否结束
                            self.board.current_player = 1  # 切换回玩家回合
                            self.update_status("轮到你了 ('O')")
                    else:
                        print(
                            f"AI错误: 预测移动{move_index}但board[{row}][{col}]不为None ({self.board.board[row][col]})。"
                        )
                        self.handle_ai_prediction_error()
                else:
                    print(
                        f"AI错误: 预测移动索引{move_index}但按钮文本不为空('{self.buttons[move_index].text}')。"
                    )
                    self.handle_ai_prediction_error()

            elif move_index is None:
                print("AI预测失败（返回None）。")
                self.handle_ai_prediction_error()
            else:
                print(f"AI错误: 预测了无效的移动索引: {move_index}")
                self.handle_ai_prediction_error()

        elif not self._ai_enabled:
            print("AI移动跳过: AI已禁用。")
        elif self.board.current_player != 0:
            print(f"AI移动跳过: 不是AI的回合（玩家是{self.board.current_player}）。")

    def handle_ai_prediction_error(self):
        """
        处理AI预测失败的情况

        当AI无法做出有效移动时，尝试随机选择一个合法移动作为后备方案
        """
        print("AI无法做出有效移动。游戏可能卡住。")
        # 尝试随机合法移动作为后备方案
        available_moves = []
        for i in range(self.board.board_size):
            row, col = i // 3, i % 3
            if self.board.board[row][col] is None:
                available_moves.append(i)

        if available_moves:
            print("尝试随机后备移动...")
            random_index = np.random.choice(available_moves)
            row, col = random_index // 3, random_index % 3
            print(f"AI随机在索引{random_index} ([{row}][{col}])处放置'X'")
            self.buttons[random_index].text = "X"
            self.board.board[row][col] = "X"
            if not self.check_game_status():
                self.board.current_player = 1
                self.update_status("轮到你了 ('O')")
        else:
            # 没有剩余移动，应该已经被平局检查捕获？
            print("AI预测错误，但未找到可用移动。重新检查状态。")
            self.check_game_status()  # 重新检查平局/胜利

    def check_game_status(self):
        """
        检查游戏状态（胜利/失败/平局）

        检查所有可能的获胜线（行、列、对角线）和平局条件

        返回:
            bool: 游戏是否结束
        """
        if self.game_over_flag:  # 如果已经结束，不再重新检查
            return True

        print("检查游戏状态...")
        board_state = self.board.board
        winner = None

        # 检查行
        for r in range(3):
            if (
                board_state[r][0] is not None
                and board_state[r][0] == board_state[r][1] == board_state[r][2]
            ):
                winner = board_state[r][0]
                break

        # 检查列
        if winner is None:
            for c in range(3):
                if (
                    board_state[0][c] is not None
                    and board_state[0][c] == board_state[1][c] == board_state[2][c]
                ):
                    winner = board_state[0][c]
                    break

        # 检查对角线
        if winner is None:
            center_cell = board_state[1][1]
            if center_cell is not None:
                if board_state[0][0] == center_cell == board_state[2][2]:
                    winner = center_cell
                elif board_state[0][2] == center_cell == board_state[2][0]:
                    winner = center_cell

        # 如果找到获胜者
        if winner:
            print(f"找到获胜者: {winner}")
            self.game_over(winner)
            return True

        # 检查平局（无获胜者且棋盘已满）
        is_full = all(cell is not None for row in board_state for cell in row)
        if is_full:
            print("棋盘已满，游戏平局。")
            self.game_over("Draw")
            return True

        print("游戏继续。")
        return False  # 游戏继续

    def game_over(self, result):
        """
        处理游戏结束

        参数:
            result: 游戏结果（'X', 'O', 或 'Draw'）
        """
        print(f"游戏结束！结果: {result}")
        self.game_over_flag = True  # 设置标志
        self.disable_buttons()

        status_text = ""
        if result == "X":
            status_text = "人类你输了 (AI Wins 'X')"
        elif result == "O":
            status_text = "人类你赢了 (Player Wins 'O')"
        elif result == "Draw":
            status_text = "平局 (Draw)"
        else:
            status_text = f"游戏结束 - 未知: {result}"

        self.update_status(status_text)

    def disable_buttons(self):
        """禁用所有棋盘按钮"""
        print("禁用游戏按钮。")
        for btn in self.buttons:
            btn.disabled = True

    def reset_game(self):
        """重置游戏棋盘和UI元素"""
        print("重置游戏...")
        self.board.reset_board()
        self.game_over_flag = False  # 重置游戏结束标志

        for i, btn in enumerate(self.buttons):
            btn.text = ""
            btn.disabled = False  # 重新启用按钮
            # 确保内部棋盘也已清空（reset_board应该会做这个）
            row, col = i // 3, i % 3
            if self.board.board[row][col] is not None:
                print(f"警告: 重置后棋盘格子[{row}][{col}]不为None。强制清空。")
                self.board.board[row][col] = None

        # --- 设置AI为起始玩家并触发其移动 ---
        self.board.current_player = 0  # *** AI ('X') 先手 ***

        # 更新状态并触发AI移动，仅在AI启用时
        if self._ai_enabled:
            self.update_status("AI ('X') 正在思考...")
            # 再次使用Clock调度以确保安全/一致性
            Clock.schedule_once(self.trigger_initial_ai_move, 0.1)
        else:
            # AI加载失败，显示错误并禁用棋盘
            self.update_status("错误：AI未加载！")
            self.disable_buttons()

    def update_status(self, message: str):
        """
        安全地更新状态标签文本

        参数:
            message (str): 要显示的状态消息
        """
        if self.parent and hasattr(self.parent, "status_label"):
            self.parent.status_label.text = message
        else:
            print(f"状态更新（未找到标签）: {message}")


class GameControls(BoxLayout):
    """
    游戏控制面板类

    管理游戏界面布局，包括棋盘、状态标签和重置按钮
    """

    def __init__(self, **kwargs):
        """
        初始化游戏控制面板

        创建左右布局：左侧棋盘，右侧控制面板
        """
        super(GameControls, self).__init__(**kwargs)
        self.orientation = "horizontal"
        self.padding = 10
        self.spacing = 10

        # 设置背景颜色
        with self.canvas.before:
            Color(0.95, 0.95, 0.95, 1)  # 浅灰色背景
            self._background_rect = Rectangle(size=self.size, pos=self.pos)
        self.bind(size=self._update_background, pos=self._update_background)

        # --- 左侧：游戏棋盘 ---
        self.game_grid = GameGrid(size_hint=(0.7, 1))
        self.add_widget(self.game_grid)

        # --- 右侧：控制面板 ---
        control_panel = BoxLayout(
            orientation="vertical", size_hint=(0.3, 1), spacing=10, padding=10
        )

        # 标题标签
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

        # 状态标签（在棋盘初始化后根据AI状态设置）
        self.status_label = Label(
            font_size="18sp",
            font_name=FONT_NAME,
            color=[0.2, 0.2, 0.2, 1],
            size_hint=(1, 0.3),
        )

        # 绑定大小以更新text_size用于对齐/换行
        def update_text_size(instance, size):
            instance.text_size = (size[0], None)  # 宽度约束，高度自动

        self.status_label.bind(size=update_text_size)
        control_panel.add_widget(self.status_label)

        # --- 重置按钮 ---
        reset_button = Button(
            text="重置游戏",
            font_size="16sp",
            font_name=FONT_NAME,
            size_hint_y=None,  # 固定高度
            height=50,
            background_color=[0.2, 0.6, 0.8, 1],
            color=[1, 1, 1, 1],
        )
        reset_button.bind(on_press=self.reset_game_action)
        control_panel.add_widget(reset_button)

        self.add_widget(control_panel)

        # --- 设置初始状态标签文本 ---
        # 这在self.game_grid初始化并尝试AI加载后运行
        if self.game_grid._ai_enabled:
            # AI做出第一步之前的初始状态（通过Clock）
            self.status_label.text = "游戏开始\nAI ('X') 正在思考..."
        else:
            self.status_label.text = "错误：无法加载AI模型！\n请检查模型文件。"
            self.status_label.color = [0.8, 0.1, 0.1, 1]

    def _update_background(self, instance, value):
        """
        更新背景矩形的大小和位置

        参数:
            instance: 组件实例
            value: 新的大小/位置值
        """
        self._background_rect.pos = instance.pos
        self._background_rect.size = instance.size

    def reset_game_action(self, instance):
        """
        重置按钮点击事件处理

        参数:
            instance: 按钮实例
        """
        print("重置按钮被按下（GameControls）")
        # 委托给棋盘重置，棋盘会处理AI第一步等
        self.game_grid.reset_game()


class TicTacToeApp(App):
    """
    井字棋应用程序类

    Kivy应用程序的主入口点
    """

    def build(self):
        """
        构建应用程序界面

        返回:
            Widget: 根组件（GameControls实例）
        """
        self.title = "井字棋 AI 对战 (Tic Tac Toe AI) - AI First"
        return GameControls()


if __name__ == "__main__":
    TicTacToeApp().run()
