# -*- coding: utf-8 -*-
"""
增强版井字棋游戏主程序 - 支持模式/先手/难度选择

使用 gui.shared 中的共享组件。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kivy.app import App
from kivy.graphics import Color, Rectangle
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.spinner import Spinner

from src.gui.shared import GameGrid, FONT_NAME


class GameControls(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "horizontal"
        self.padding = 10
        self.spacing = 10

        with self.canvas.before:
            Color(0.95, 0.95, 0.95, 1)
            self._background_rect = Rectangle(size=self.size, pos=self.pos)
        self.bind(size=self._update_background, pos=self._update_background)

        self.game_grid = GameGrid(size_hint=(0.6, 1))
        self.add_widget(self.game_grid)

        control_panel = BoxLayout(
            orientation="vertical", size_hint=(0.4, 1), spacing=10, padding=10
        )

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

        self.status_label = Label(
            font_size="18sp",
            font_name=FONT_NAME,
            color=[0.2, 0.2, 0.2, 1],
            size_hint=(1, 0.2),
        )
        control_panel.add_widget(self.status_label)

        # 游戏模式选择
        control_panel.add_widget(
            Label(text="游戏模式:", font_size="14sp", font_name=FONT_NAME,
                  size_hint_y=None, height=30)
        )
        self.mode_spinner = Spinner(
            text="人机对战", values=["人机对战", "人人对战", "机机对战"],
            font_name=FONT_NAME, size_hint_y=None, height=40,
        )
        control_panel.add_widget(self.mode_spinner)

        # 先手选择
        control_panel.add_widget(
            Label(text="先手选择:", font_size="14sp", font_name=FONT_NAME,
                  size_hint_y=None, height=30)
        )
        self.first_spinner = Spinner(
            text="玩家先手", values=["玩家先手", "AI先手"],
            font_name=FONT_NAME, size_hint_y=None, height=40,
        )
        control_panel.add_widget(self.first_spinner)

        # 难度选择
        control_panel.add_widget(
            Label(text="AI难度:", font_size="14sp", font_name=FONT_NAME,
                  size_hint_y=None, height=30)
        )
        self.difficulty_spinner = Spinner(
            text="中等", values=["简单", "中等", "困难"],
            font_name=FONT_NAME, size_hint_y=None, height=40,
        )
        control_panel.add_widget(self.difficulty_spinner)

        # 按钮区域
        button_layout = BoxLayout(
            orientation="vertical", spacing=5, size_hint_y=None, height=150
        )

        start_btn = Button(
            text="开始游戏", font_size="16sp", font_name=FONT_NAME,
            background_color=[0.2, 0.6, 0.8, 1], color=[1, 1, 1, 1],
        )
        start_btn.bind(on_press=self.start_game)
        button_layout.add_widget(start_btn)

        load_btn = Button(
            text="加载游戏", font_size="16sp", font_name=FONT_NAME,
            background_color=[0.4, 0.7, 0.4, 1], color=[1, 1, 1, 1],
        )
        load_btn.bind(on_press=self.load_game)
        button_layout.add_widget(load_btn)

        reset_btn = Button(
            text="重置游戏", font_size="16sp", font_name=FONT_NAME,
            background_color=[0.8, 0.4, 0.4, 1], color=[1, 1, 1, 1],
        )
        reset_btn.bind(on_press=self.reset_game)
        button_layout.add_widget(reset_btn)

        control_panel.add_widget(button_layout)
        control_panel.add_widget(Label(size_hint_y=1))

        self.add_widget(control_panel)
        self.status_label.text = "选择游戏模式并开始"

    def _update_background(self, instance, value):
        self._background_rect.pos = instance.pos
        self._background_rect.size = instance.size

    def start_game(self, instance):
        mode_map = {"人机对战": "pve", "人人对战": "pvp", "机机对战": "eve"}
        difficulty_map = {"简单": "easy", "中等": "medium", "困难": "hard"}

        game_mode = mode_map.get(self.mode_spinner.text, "pve")
        ai_first = self.first_spinner.text == "AI先手"
        difficulty = difficulty_map.get(self.difficulty_spinner.text, "medium")

        self.game_grid.initialize_game(
            game_mode=game_mode, ai_first=ai_first, difficulty=difficulty
        )
        self.status_label.text = "游戏开始！"

    def load_game(self, instance):
        if self.game_grid.load_checkpoint():
            self.status_label.text = "游戏已加载"
        else:
            self.status_label.text = "没有可加载的游戏"

    def reset_game(self, instance):
        self.game_grid.reset_board()
        self.status_label.text = "游戏已重置，请点击开始游戏"


class TicTacToeApp(App):
    def build(self):
        self.title = "井字棋 AI 对战 - 增强版"
        return GameControls()


if __name__ == "__main__":
    TicTacToeApp().run()
