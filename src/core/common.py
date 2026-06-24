"""
井字棋游戏通用模块

本模块定义了井字棋游戏的基础配置和棋盘类，提供游戏的核心数据结构。

主要组件：
1. 游戏配置常量：棋盘大小、行列数等
2. GameBoard类：棋盘状态管理

作者：AI Assistant
版本：2.0
日期：2026-04-13
"""

# ==================== 游戏配置常量 ====================
BOARD_ROW = 3  # 棋盘行数
BOARD_COL = 3  # 棋盘列数
BOARD_SIZE = BOARD_ROW * BOARD_COL  # 棋盘总大小（9个格子）


class GameBoard:
    """
    游戏棋盘类

    管理井字棋游戏的核心状态，包括：
    - 棋盘状态：3x3的二维列表，存储每个位置的玩家标记
    - 当前玩家：0表示AI（X），1表示人类（O）
    - 棋盘尺寸信息

    棋盘状态表示：
    - None：空位
    - 'X'：AI玩家的标记
    - 'O'：人类玩家的标记
    """

    board: list[list[str | None]]  # 棋盘状态矩阵
    current_player: int  # 当前玩家（0=AI, 1=人类）
    col: int  # 列数
    row: int  # 行数
    board_size: int  # 棋盘总大小

    def __init__(self, col: int = BOARD_COL, row: int = BOARD_ROW):
        """
        初始化游戏棋盘

        参数:
            col (int): 列数，默认为BOARD_COL（3）
            row (int): 行数，默认为BOARD_ROW（3）
        """
        self.board_size = BOARD_SIZE
        self.col = col
        self.row = row
        # 初始化棋盘：所有位置为None（空）
        self.board = [[None for _ in range(self.col)] for _ in range(self.row)]
        self.current_player = 0  # AI（X）先手

    def reset_board(self):
        """
        重置棋盘状态

        将所有位置清空，并将当前玩家重置为AI（玩家0）
        """
        self.current_player = 0
        # 重新初始化棋盘：所有位置为None
        self.board = [[None for _ in range(self.col)] for _ in range(self.row)]
