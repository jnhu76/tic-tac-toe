# -*- coding: utf-8 -*-
"""
AI模块

包含神经网络、训练算法和对手策略
"""

from .train import (
    NeuralNetwork,
    TicTacToeGame,
    board_to_input,
    evaluate_model,
    get_computer_move,
    get_random_move,
    learn_from_game,
    load_model,
    play_random_game,
    save_model,
    train_against_random,
)
from src.core.game import GameEngine

__all__ = [
    "NeuralNetwork",
    "TicTacToeGame",
    "GameEngine",
    "board_to_input",
    "get_random_move",
    "get_computer_move",
    "learn_from_game",
    "play_random_game",
    "train_against_random",
    "evaluate_model",
    "save_model",
    "load_model",
]
