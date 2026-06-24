# -*- coding: utf-8 -*-
"""
井字棋游戏主程序 - 标准版 GUI

使用 gui.shared 中的共享组件。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label

from src.gui.shared import GameGrid, FONT_NAME


class GameControls(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "horizontal"
        self.padding = 10
        self.spacing = 10

        self.game_grid = GameGrid(size_hint=(0.7, 1))
        self.add_widget(self.game_grid)

        control_panel = BoxLayout(
            orientation="vertical", size_hint=(0.3, 1), spacing=10, padding=10
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
            size_hint=(1, 0.3),
        )
        control_panel.add_widget(self.status_label)

        reset_button = Button(
            text="重置游戏",
            font_size="16sp",
            font_name=FONT_NAME,
            size_hint_y=None,
            height=50,
            background_color=[0.2, 0.6, 0.8, 1],
            color=[1, 1, 1, 1],
        )
        reset_button.bind(on_press=self.reset_game_action)
        control_panel.add_widget(reset_button)

        self.add_widget(control_panel)

        if self.game_grid._ai_enabled:
            self.status_label.text = "游戏开始\nAI ('X') 正在思考..."
            self.game_grid.initialize_game(
                game_mode="pve", ai_first=True, difficulty="hard"
            )
        else:
            self.status_label.text = "错误：无法加载AI模型！"

    def reset_game_action(self, instance):
        self.game_grid.reset_board()
        self.status_label.text = "游戏已重置，请点击开始游戏"


class TicTacToeApp(App):
    def build(self):
        self.title = "井字棋 AI 对战"
        return GameControls()


if __name__ == "__main__":
    TicTacToeApp().run()
