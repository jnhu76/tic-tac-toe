# -*- coding: utf-8 -*-
"""
GUI模块

包含游戏界面实现
"""

from .main import AIManager as LegacyAIManager
from .main import GameGrid as LegacyGameGrid
from .main import TicTacToeApp as LegacyTicTacToeApp
from .main_enhanced import AIManager, GameControls, GameGrid, TicTacToeApp

__all__ = [
    "AIManager",
    "GameGrid",
    "GameControls",
    "TicTacToeApp",
    "LegacyAIManager",
    "LegacyGameGrid",
    "LegacyTicTacToeApp",
]
