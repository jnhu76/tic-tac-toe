# -*- coding: utf-8 -*-
"""
核心模块

包含游戏板定义、检查点系统等核心功能
"""

from .common import BOARD_COL, BOARD_ROW, BOARD_SIZE, GameBoard
from .game import GameEngine

try:
    from .checkpoint import (
        CheckpointManager,
        GameState,
        TrainingState,
        create_checkpoint_manager,
    )
except ImportError:
    pass

__all__ = [
    "GameBoard",
    "GameEngine",
    "BOARD_SIZE",
    "BOARD_COL",
    "BOARD_ROW",
    "CheckpointManager",
    "TrainingState",
    "GameState",
    "create_checkpoint_manager",
]
