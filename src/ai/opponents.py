# -*- coding: utf-8 -*-
"""
对手策略模块

定义统一的 Opponent 协议，让训练和评估可以注入不同的对手策略。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.core.game import GameEngine


@runtime_checkable
class Opponent(Protocol):
    """对手策略协议 — 所有对手实现此接口"""

    def get_move(self, game: GameEngine) -> int:
        """根据当前游戏状态返回一步合法移动的位置索引 (0-8)"""
        ...


class RandomOpponent:
    """随机对手 — 从合法移动中随机选择"""

    def get_move(self, game: GameEngine) -> int:
        moves = game.get_legal_moves()
        import numpy as np
        return moves[np.random.randint(len(moves))]


class RuleBasedOpponent:
    """
    规则对手 — 优先赢 → 堵对手 → 占中心 → 随机
    与 train_enhanced.py 中的 rule_based_opponent_move 行为一致
    """

    def get_move(self, game: GameEngine) -> int:
        b = game.board
        lines = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),
            (0, 3, 6), (1, 4, 7), (2, 5, 8),
            (0, 4, 8), (2, 4, 6),
        ]
        my_symbol = game.get_symbol(game.current_player)
        opp_symbol = "O" if my_symbol == "X" else "X"

        # 赢
        for a, b_idx, c in lines:
            cells = [b[a], b[b_idx], b[c]]
            if cells.count(my_symbol) == 2 and cells.count(None) == 1:
                return [a, b_idx, c][cells.index(None)]

        # 堵
        for a, b_idx, c in lines:
            cells = [b[a], b[b_idx], b[c]]
            if cells.count(opp_symbol) == 2 and cells.count(None) == 1:
                return [a, b_idx, c][cells.index(None)]

        # 中心
        if b[4] is None:
            return 4

        # 随机
        import numpy as np
        moves = game.get_legal_moves()
        return moves[np.random.randint(len(moves))]


class NeuralNetOpponent:
    """
    神经网络对手 — 使用 NeuralNetwork 预测移动
    支持 epsilon-greedy 探索
    """

    def __init__(self, nn, epsilon: float = 0.0):
        from src.ai.train import NeuralNetwork, board_to_input
        self.nn = nn
        self.epsilon = epsilon
        self._board_to_input = board_to_input

    def get_move(self, game: GameEngine) -> int:
        board_2d = game.get_board_2d()
        inputs = self._board_to_input(board_2d)
        self.nn.forward(inputs)
        moves = game.get_legal_moves()

        if self.epsilon > 0.0:
            import numpy as np
            if np.random.rand() < self.epsilon:
                return moves[np.random.randint(len(moves))]

        best_move = -1
        best_prob = -1.0
        for i in moves:
            if self.nn.outputs[i] > best_prob:
                best_prob = self.nn.outputs[i]
                best_move = i
        return best_move
