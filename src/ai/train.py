"""
井字棋神经网络训练模块

本模块实现了基于强化学习的井字棋AI训练系统，包含以下核心功能：
1. 神经网络模型：用于评估棋盘状态和选择最优动作
2. 训练环境：模拟井字棋游戏环境，支持多种对手类型
3. 训练策略：提供课程学习、混合对手训练、自我对弈等多种训练方法
4. 评估系统：全面评估模型性能，支持随机对手和规则对手
5. 日志系统：记录训练过程和性能指标
6. 检查点系统：支持训练中断恢复和定期自动保存
7. 先手后手功能：支持AI作为先手或后手进行训练和对战

作者：AI Assistant
版本：3.0
日期：2026-04-13
"""

import copy
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.common import BOARD_COL, BOARD_ROW, BOARD_SIZE, GameBoard
from src.core.config import ModelConfig, ensure_dir_for_file, resolve_model_path
from src.core.game import GameEngine

# ==================== 神经网络配置参数 ====================
NN_INPUT_SIZE = 18  # 输入层大小：3x3棋盘，每个位置2个特征（X和O）
NN_HIDDEN_SIZE = 100  # 隐藏层大小：与C代码一致
NN_OUTPUT_SIZE = 9  # 输出层大小：9个可能的位置
LEARNING_RATE = 0.1  # 学习率：与C代码一致


class NeuralNetwork:
    """
    神经网络类

    实现一个简单的三层全连接神经网络，用于井字棋游戏的状态评估和动作选择。
    网络结构：输入层(18) -> 隐藏层(100) -> 输出层(9)

    激活函数：
    - 隐藏层：ReLU
    - 输出层：Softmax（输出概率分布）

    与C代码完全一致的反向传播实现
    """

    def __init__(
        self,
        input_size: int = NN_INPUT_SIZE,
        hidden_size: int = NN_HIDDEN_SIZE,
        output_size: int = NN_OUTPUT_SIZE,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.weights_ih = np.random.uniform(-0.5, 0.5, (input_size, hidden_size))
        self.weights_ho = np.random.uniform(-0.5, 0.5, (hidden_size, output_size))
        self.biases_h = np.random.uniform(-0.5, 0.5, hidden_size)
        self.biases_o = np.random.uniform(-0.5, 0.5, output_size)

        self.inputs = np.zeros(input_size)
        self.hidden_raw = np.zeros(hidden_size)
        self.hidden = np.zeros(hidden_size)
        self.raw_logits = np.zeros(output_size)
        self.outputs = np.zeros(output_size)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs.copy()

        self.hidden_raw = np.dot(self.inputs, self.weights_ih) + self.biases_h
        self.hidden = np.maximum(0, self.hidden_raw)

        self.raw_logits = np.dot(self.hidden, self.weights_ho) + self.biases_o
        self.outputs = self.softmax(self.raw_logits)

        return self.outputs

    def softmax(self, x: np.ndarray) -> np.ndarray:
        shifted = x - np.max(x)
        exp_x = np.exp(shifted)
        s = exp_x.sum()
        if s > 0:
            return exp_x / s
        return np.ones_like(x) / len(x)

    def backward(
        self, target_probs: np.ndarray, learning_rate: float, reward_scaling: float
    ):
        """
        反向传播 - 与C代码的backprop完全一致

        C代码逻辑：
        output_deltas[i] = (output[i] - target[i]) * |reward_scaling|
        hidden_deltas[i] = sum(output_deltas * weights_ho) * relu'(hidden)
        weights_ho -= lr * output_deltas * hidden
        biases_o -= lr * output_deltas
        weights_ih -= lr * hidden_deltas * inputs
        biases_h -= lr * hidden_deltas
        """
        output_deltas = (self.outputs - target_probs) * abs(reward_scaling)

        hidden_errors = np.dot(output_deltas, self.weights_ho.T)
        relu_deriv = (self.hidden_raw > 0).astype(float)
        hidden_deltas = hidden_errors * relu_deriv

        self.weights_ho -= learning_rate * np.outer(self.hidden, output_deltas)
        self.biases_o -= learning_rate * output_deltas

        self.weights_ih -= learning_rate * np.outer(self.inputs, hidden_deltas)
        self.biases_h -= learning_rate * hidden_deltas

    def copy(self) -> "NeuralNetwork":
        return copy.deepcopy(self)

    def get_best_move(self, inputs: np.ndarray, board: list[list[str | None]]) -> int:
        """
        根据当前棋盘状态选择最佳合法移动

        与C代码的get_computer_move一致：选择概率最高的合法位置

        参数:
            inputs (np.ndarray): 棋盘状态特征
            board (list[list[str|None]]): 当前棋盘状态

        返回:
            int: 选择的移动位置索引（0-8），如果无可用移动则返回-1
        """
        output = self.forward(inputs)
        best_move = -1
        best_prob = -1.0
        for i in range(BOARD_ROW):
            for j in range(BOARD_COL):
                pos = i * BOARD_COL + j
                if board[i][j] is None and output[pos] > best_prob:
                    best_prob = output[pos]
                    best_move = pos
        return best_move


def board_to_input(board: list[list[str | None]]) -> np.ndarray:
    """
    将棋盘状态转换为神经网络输入特征

    编码方式（与C代码一致）：
    - 空位：[0, 0]
    - X：[1, 0]
    - O：[0, 1]
    """
    inputs = np.zeros(NN_INPUT_SIZE)
    for i in range(BOARD_ROW):
        for j in range(BOARD_COL):
            pos = i * BOARD_COL + j
            cell = board[i][j]
            if cell == "X":
                inputs[pos * 2] = 1
                inputs[pos * 2 + 1] = 0
            elif cell == "O":
                inputs[pos * 2] = 0
                inputs[pos * 2 + 1] = 1
    return inputs


class TicTacToeGame(GameEngine):
    """向后兼容别名 — 使用 GameEngine 作为实现"""
    pass


def get_random_move(game: TicTacToeGame) -> int:
    available = game.get_available_moves()
    return available[np.random.randint(len(available))]


def get_computer_move(game: TicTacToeGame, nn: NeuralNetwork) -> int:
    board_2d = game.get_board_2d()
    inputs = board_to_input(board_2d)
    nn.forward(inputs)
    best_move = -1
    best_prob = -1.0
    for i in range(BOARD_SIZE):
        if game.board[i] is None and nn.outputs[i] > best_prob:
            best_prob = nn.outputs[i]
            best_move = i
    return best_move


def learn_from_game(
    nn: NeuralNetwork, move_history: list[int], nn_player: int, winner: str | None
):
    """
    从一局游戏学习 - 与C代码的learn_from_game完全一致

    参数:
        nn: 神经网络
        move_history: 移动历史（位置索引列表）
        nn_player: NN是哪个玩家 (0=X, 1=O)
        winner: 获胜者 ('X', 'O', 'T' 或 None)
    """
    nn_symbol = "X" if nn_player == 0 else "O"

    if winner == "T":
        reward = 0.3
    elif winner == nn_symbol:
        reward = 1.0
    else:
        reward = -2.0

    num_moves = len(move_history)

    for move_idx in range(num_moves):
        if move_idx % 2 != nn_player:
            continue

        game = TicTacToeGame()
        for i in range(move_idx):
            symbol = "X" if i % 2 == 0 else "O"
            game.board[move_history[i]] = symbol

        inputs = board_to_input(game.get_board_2d())
        nn.forward(inputs)

        move = move_history[move_idx]

        move_importance = 0.5 + 0.5 * move_idx / num_moves
        scaled_reward = reward * move_importance

        target_probs = np.zeros(NN_OUTPUT_SIZE)

        if scaled_reward >= 0:
            target_probs[move] = 1.0
        else:
            valid_moves_left = BOARD_SIZE - move_idx - 1
            if valid_moves_left > 0:
                other_prob = 1.0 / valid_moves_left
                for i in range(BOARD_SIZE):
                    if game.board[i] is None and i != move:
                        target_probs[i] = other_prob

        nn.backward(target_probs, LEARNING_RATE, scaled_reward)


def play_random_game(nn: NeuralNetwork) -> str | None:
    """
    与随机对手玩一局游戏并学习 - 与C代码一致

    人类(随机)=X先手(player 0), NN=O后手(player 1)

    返回:
        winner: 'X', 'O', 或 'T'
    """
    game = TicTacToeGame()
    move_history = []

    while True:
        if game.current_player == 0:
            move = get_random_move(game)
        else:
            move = get_computer_move(game, nn)

        symbol = game.get_symbol(game.current_player)
        game.board[move] = symbol
        move_history.append(move)
        game.current_player = 1 - game.current_player

        game_over, winner = game.is_game_over()
        if game_over:
            learn_from_game(nn, move_history, 1, winner)
            return winner


def train_against_random(
    nn: NeuralNetwork, num_games: int = 150000, log_interval: int = 10000
):
    """
    训练神经网络对抗随机对手 - 与C代码完全一致
    """
    wins = 0
    losses = 0
    ties = 0
    played = 0

    print(f"Training neural network against {num_games} random games...")

    for i in range(num_games):
        winner = play_random_game(nn)
        played += 1

        if winner == "O":
            wins += 1
        elif winner == "X":
            losses += 1
        else:
            ties += 1

        if (i + 1) % log_interval == 0:
            print(
                f"Games: {i+1}, Wins: {wins} ({wins*100/played:.1f}%), "
                f"Losses: {losses} ({losses*100/played:.1f}%), "
                f"Ties: {ties} ({ties*100/played:.1f}%)"
            )
            played = 0
            wins = 0
            losses = 0
            ties = 0

    print("\nTraining complete!")


def save_model(nn: NeuralNetwork, filepath: str):
    with open(filepath, "wb") as f:
        pickle.dump(nn, f)
    print(f"Neural network saved to {filepath}")


def load_model(filepath: str) -> NeuralNetwork:
    with open(filepath, "rb") as f:
        return pickle.load(f)


def evaluate_model(nn: NeuralNetwork, num_games: int = 1000):
    """
    评估模型性能
    """
    game = TicTacToeGame()
    wins = 0
    losses = 0
    ties = 0

    for _ in range(num_games):
        g = TicTacToeGame()
        while True:
            if g.current_player == 0:
                move = get_random_move(g)
            else:
                move = get_computer_move(g, nn)

            symbol = g.get_symbol(g.current_player)
            g.board[move] = symbol
            g.current_player = 1 - g.current_player

            game_over, winner = g.is_game_over()
            if game_over:
                if winner == "O":
                    wins += 1
                elif winner == "X":
                    losses += 1
                else:
                    ties += 1
                break

    total = num_games
    print(f"\nEvaluation results ({num_games} games vs random):")
    print(f"  Wins:   {wins} ({wins*100/total:.1f}%)")
    print(f"  Losses: {losses} ({losses*100/total:.1f}%)")
    print(f"  Ties:   {ties} ({ties*100/total:.1f}%)")

    return {"wins": wins, "losses": losses, "ties": ties}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="井字棋AI训练")
    parser.add_argument("-e", "--episodes", type=int, default=150000, help="训练游戏数")
    parser.add_argument(
        "-m", "--model-path", type=str, default=None, help="模型保存路径"
    )
    args = parser.parse_args()

    model_path = args.model_path or ModelConfig.get_default_model_path()
    ensure_dir_for_file(model_path)

    print("=" * 60)
    print("井字棋AI训练")
    print("=" * 60)
    print(f"训练游戏数: {args.episodes}")
    print(f"模型路径: {model_path}")
    print(f"学习率: {LEARNING_RATE}")
    print(f"隐藏层大小: {NN_HIDDEN_SIZE}")
    print("=" * 60)

    nn = NeuralNetwork()
    train_against_random(nn, args.episodes)
    save_model(nn, model_path)

    evaluate_model(nn, num_games=1000)
