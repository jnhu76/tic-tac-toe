# -*- coding: utf-8 -*-
"""
井字棋游戏引擎模块

统一的游戏逻辑实现，消除 GameBoard(2D) 和 TicTacToeGame(1D) 之间的分裂。
所有消费者（训练、GUI、测试）通过 GameEngine 访问游戏规则。

使用 1D list 表示棋盘（与 C 代码一致），但提供 2D 转换接口。
"""

import copy
from typing import Optional


BOARD_SIZE = 9
BOARD_ROWS = 3
BOARD_COLS = 3


class GameEngine:
    """
    井字棋游戏引擎 — 统一的游戏逻辑模块

    棋盘用一维列表表示 (0-8)。
    board[i] = None (空位) | 'X' | 'O'

    interface（调用者需要知道的）:
      - make_move(pos: int) -> bool
      - check_winner() -> str | None     ('X', 'O', 或 None)
      - is_game_over() -> tuple[bool, str | None]
      - get_legal_moves() -> list[int]
      - get_symbol(player: int) -> str
      - clone() -> GameEngine
      - get_board_2d() -> list[list[str|None]]
    """

    __slots__ = ("board", "current_player")

    def __init__(self) -> None:
        self.board: list[Optional[str]] = [None] * BOARD_SIZE
        self.current_player: int = 0  # 0=X先手, 1=O后手

    def reset(self) -> None:
        self.board = [None] * BOARD_SIZE
        self.current_player = 0

    def get_symbol(self, player: int) -> str:
        return "X" if player == 0 else "O"

    def make_move(self, pos: int) -> bool:
        if pos < 0 or pos >= BOARD_SIZE or self.board[pos] is not None:
            return False
        self.board[pos] = self.get_symbol(self.current_player)
        self.current_player = 1 - self.current_player
        return True

    def get_legal_moves(self) -> list[int]:
        return [i for i in range(BOARD_SIZE) if self.board[i] is None]

    def get_available_moves(self) -> list[int]:
        """向后兼容别名"""
        return self.get_legal_moves()

    _WIN_LINES = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6),
    ]

    def check_winner(self) -> Optional[str]:
        b = self.board
        for a, b_idx, c in self._WIN_LINES:
            if b[a] is not None and b[a] == b[b_idx] == b[c]:
                return b[a]
        return None

    def is_game_over(self) -> tuple[bool, Optional[str]]:
        winner = self.check_winner()
        if winner is not None:
            return True, winner
        if all(cell is not None for cell in self.board):
            return True, "T"
        return False, None

    def get_board_2d(self) -> list[list[Optional[str]]]:
        return [
            [self.board[i * BOARD_COLS + j] for j in range(BOARD_COLS)]
            for i in range(BOARD_ROWS)
        ]

    def clone(self) -> "GameEngine":
        return copy.deepcopy(self)
