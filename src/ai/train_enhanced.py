# -*- coding: utf-8 -*-
"""
增强版训练模块 - 集成检查点和先手后手功能

本模块在原有训练功能基础上增加了：
1. 检查点系统：支持训练中断恢复和定期自动保存
2. 先手后手控制：明确指定AI作为先手(X)或后手(O)
3. 增强的评估系统：分别评估AI作为先手和后的性能

作者：AI Assistant
版本：3.0
日期：2026-04-13
"""

import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ai.train import (
    BOARD_COL,
    BOARD_ROW,
    BOARD_SIZE,
    NN_INPUT_SIZE,
    LEARNING_RATE,
    NeuralNetwork,
    TicTacToeGame,
    board_to_input,
    get_computer_move,
    get_random_move,
    learn_from_game,
)
from src.ai.opponents import Opponent, RandomOpponent, RuleBasedOpponent, NeuralNetOpponent
from src.core.checkpoint import (
    AutoSaveCallback,
    CheckpointManager,
    TrainingState,
    create_checkpoint_manager,
)
from src.core.config import (
    BEST_MODEL_PATH,
    DEFAULT_MODEL_PATH,
    ModelConfig,
    TrainingConfig,
    ensure_dir_for_file,
    resolve_model_path,
)


class TrainingLogger:
    def __init__(self, log_file: str):
        self.log_file = log_file

    def log(self, message: str):
        print(message)
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"[{self._get_timestamp()}] {message}\n")
        except Exception:
            pass

    def log_metrics(self, episode: int, metrics: dict):
        msg = (
            f"Episode {episode}: "
            f"WR_rule={metrics.get('win_rate_rule', 0):.2%}, "
            f"WR_first={metrics.get('win_rate_first', 0):.2%}, "
            f"WR_second={metrics.get('win_rate_second', 0):.2%}, "
            f"epsilon={metrics.get('epsilon', 0):.4f}"
        )
        self.log(msg)

    @staticmethod
    def _get_timestamp() -> str:
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def random_opponent_move(game: TicTacToeGame) -> int:
    available = game.get_available_moves()
    return available[np.random.randint(len(available))]


def rule_based_opponent_move(game: TicTacToeGame) -> int:
    """简单规则对手：优先赢、其次堵、再占中心、最后随机"""
    b = game.board
    lines = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6),
    ]
    my_symbol = game.get_symbol(game.current_player)
    opp_symbol = "O" if my_symbol == "X" else "X"

    for a, b_idx, c in lines:
        cells = [b[a], b[b_idx], b[c]]
        if cells.count(my_symbol) == 2 and cells.count(None) == 1:
            empty_idx = [a, b_idx, c][cells.index(None)]
            return empty_idx

    for a, b_idx, c in lines:
        cells = [b[a], b[b_idx], b[c]]
        if cells.count(opp_symbol) == 2 and cells.count(None) == 1:
            empty_idx = [a, b_idx, c][cells.index(None)]
            return empty_idx

    if b[4] is None:
        return 4

    available = game.get_available_moves()
    return available[np.random.randint(len(available))]


def self_play_move(
    opponent_nn: NeuralNetwork, game: TicTacToeGame, epsilon: float = 0.0
) -> int:
    board_2d = game.get_board_2d()
    inputs = board_to_input(board_2d)
    opponent_nn.forward(inputs)
    available = game.get_available_moves()
    if np.random.rand() < epsilon and available:
        return available[np.random.randint(len(available))]
    best_move = -1
    best_prob = -1.0
    for i in available:
        if opponent_nn.outputs[i] > best_prob:
            best_prob = opponent_nn.outputs[i]
            best_move = i
    return best_move


@dataclass
class FirstMoverConfig:
    """
    先手配置类

    用于配置训练或对战中的先手后手规则
    """

    ai_first_ratio: float = 0.5  # AI作为先手的比例（0-1）
    alternate_turns: bool = False  # 是否交替先手
    fixed_first_player: Optional[str] = None  # 固定先手玩家 ('X', 'O', 或 None随机)

    def get_first_player(self, episode: int = 0) -> str:
        """
        获取当前回合的先手玩家

        参数:
            episode (int): 当前训练轮次（用于交替模式）

        返回:
            str: 'X' 或 'O'
        """
        if self.fixed_first_player is not None:
            return self.fixed_first_player

        if self.alternate_turns:
            # 交替模式：偶数轮X先手，奇数轮O先手
            return "X" if episode % 2 == 0 else "O"

        # 随机模式：根据ai_first_ratio决定
        return "X" if np.random.random() < self.ai_first_ratio else "O"


def train_with_checkpoint_and_firstmover(
    nn: NeuralNetwork,
    total_episodes: int = 3000000,
    learning_rate: float = 0.001,
    discount_factor: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.9998,
    checkpoint_manager: Optional[CheckpointManager] = None,
    firstmover_config: Optional[FirstMoverConfig] = None,
    resume_from_checkpoint: Optional[str] = None,
    opponent_type: str = "mixed",  # 'random', 'rule', 'mixed', 'self_play'
    self_play_ratio: float = 0.3,
    rule_opponent_ratio: float = 0.5,
    model_save_path: str = "best_model.pkl",
    log_interval: int = 10000,
    eval_interval: int = 25000,
    target_win_rate_rule: float = 0.50,
    num_eval_games: int = 2000,
) -> NeuralNetwork:
    """
    增强版训练函数 - 集成检查点和先手后手控制

    参数:
        nn (NeuralNetwork): 神经网络模型
        total_episodes (int): 总训练轮次
        learning_rate (float): 学习率
        discount_factor (float): 折扣因子
        epsilon_start (float): 初始探索率
        epsilon_end (float): 最终探索率
        epsilon_decay (float): 探索率衰减系数
        checkpoint_manager (CheckpointManager, optional): 检查点管理器
        firstmover_config (FirstMoverConfig, optional): 先手配置
        resume_from_checkpoint (str, optional): 从检查点恢复的路径
        opponent_type (str): 对手类型
        self_play_ratio (float): 自我对弈比例
        rule_opponent_ratio (float): 规则对手比例
        model_save_path (str): 模型保存路径
        log_interval (int): 日志记录间隔
        eval_interval (int): 评估间隔
        target_win_rate_rule (float): 目标胜率
        num_eval_games (int): 评估游戏数

    返回:
        NeuralNetwork: 训练后的神经网络
    """
    logger = TrainingLogger("training_enhanced_log.txt")

    # 初始化检查点管理器
    if checkpoint_manager is None:
        checkpoint_manager = create_checkpoint_manager(
            checkpoint_dir="checkpoints",
            max_checkpoints=5,
            auto_save_interval=eval_interval,
        )

    # 初始化先手配置
    if firstmover_config is None:
        firstmover_config = FirstMoverConfig(ai_first_ratio=0.5)

    # 从检查点恢复（如果指定）
    start_episode = 0
    best_win_rate_rule = 0.0
    best_win_rate_random = 0.0
    epsilon = epsilon_start
    metrics_history = []

    if resume_from_checkpoint is not None:
        try:
            nn, state = checkpoint_manager.load_training_checkpoint(
                resume_from_checkpoint
            )
            start_episode = state.episode
            best_win_rate_rule = state.best_win_rate_rule
            best_win_rate_random = state.best_win_rate_random
            epsilon = state.epsilon
            metrics_history = state.metrics_history
            logger.log(f"从检查点恢复: 轮次 {start_episode}/{total_episodes}")
        except Exception as e:
            logger.log(f"无法从检查点恢复: {e}，从头开始训练")

    # 创建对手网络（用于自我对弈）
    opponent_nn = nn.copy()

    logger.log("=" * 60)
    logger.log("开始增强版训练")
    logger.log(f"总轮次: {total_episodes}")
    logger.log(f"先手配置: AI先手比例={firstmover_config.ai_first_ratio}")
    logger.log(f"对手类型: {opponent_type}")
    logger.log("=" * 60)

    for episode in range(start_episode, total_episodes):
        # 确定先手玩家
        first_player = firstmover_config.get_first_player(episode)
        ai_player = (
            firstmover_config.get_first_player(episode)
            if firstmover_config.fixed_first_player
            else ("X" if np.random.random() < 0.5 else "O")
        )

        # 创建游戏环境
        game = TicTacToeGame()
        trajectory = []

        # 选择对手（使用 Opponent 协议）
        opponent_obj: Opponent
        if opponent_type == "random":
            opponent_obj = RandomOpponent()
        elif opponent_type == "rule":
            opponent_obj = RuleBasedOpponent()
        elif opponent_type == "self_play":
            opponent_obj = NeuralNetOpponent(opponent_nn, epsilon=0.3)
        else:  # mixed
            rand_val = np.random.rand()
            if rand_val < self_play_ratio:
                opponent_obj = NeuralNetOpponent(opponent_nn, epsilon=0.3)
            elif rand_val < self_play_ratio + rule_opponent_ratio:
                opponent_obj = RuleBasedOpponent()
            else:
                opponent_obj = RandomOpponent()

        # 游戏循环
        ai_symbol = game.get_symbol(
            0 if ai_player == "X" else 1
        )
        while True:
            over, _ = game.is_game_over()
            if over:
                break

            if game.get_symbol(game.current_player) == ai_player:
                # AI移动 - epsilon-greedy
                inputs = board_to_input(game.get_board_2d())
                nn.forward(inputs)
                available = game.get_available_moves()
                if np.random.rand() < epsilon and available:
                    move_index = available[np.random.randint(len(available))]
                else:
                    best_prob = -1.0
                    move_index = -1
                    for i in available:
                        if nn.outputs[i] > best_prob:
                            best_prob = nn.outputs[i]
                            move_index = i
                trajectory.append((inputs, move_index))
            else:
                # 对手移动
                move_index = opponent_obj.get_move(game)

            game.make_move(move_index)

        # 计算奖励
        winner = game.check_winner()
        if winner == ai_symbol:
            reward = 1.0
        elif winner is not None:
            reward = -1.0
        else:
            reward = 0.0

        # 反向传播
        for idx, (inputs, move_index) in enumerate(reversed(trajectory)):
            discounted_reward = reward * (discount_factor ** idx)
            nn.forward(inputs)
            target_probs = nn.outputs.copy()
            target_probs[move_index] += discounted_reward
            nn.backward(target_probs, learning_rate, discounted_reward)

        # 更新探索率
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # 定期更新对手网络
        if episode % 1000 == 0 and opponent_type in ["self_play", "mixed"]:
            opponent_nn = nn.copy()

        # 定期记录日志
        if (episode + 1) % log_interval == 0:
            logger.log(
                f"Episode {episode + 1}/{total_episodes}, Epsilon: {epsilon:.4f}"
            )

        # 定期评估和保存检查点
        if (episode + 1) % eval_interval == 0:
            # 分别评估AI作为先手和后的性能
            metrics_first = evaluate_with_firstmover(
                nn,
                RuleBasedOpponent(),
                num_games=num_eval_games // 2,
                ai_first=True,
            )
            metrics_second = evaluate_with_firstmover(
                nn,
                RuleBasedOpponent(),
                num_games=num_eval_games // 2,
                ai_first=False,
            )

            metrics = {
                "win_rate_random": 0.0,  # 暂不评估随机对手
                "draw_rate_random": 0.0,
                "loss_rate_random": 0.0,
                "win_rate_rule": (
                    metrics_first["win_rate"] + metrics_second["win_rate"]
                )
                / 2,
                "draw_rate_rule": (
                    metrics_first["draw_rate"] + metrics_second["draw_rate"]
                )
                / 2,
                "loss_rate_rule": (
                    metrics_first["loss_rate"] + metrics_second["loss_rate"]
                )
                / 2,
                "win_rate_first": metrics_first["win_rate"],
                "win_rate_second": metrics_second["win_rate"],
                "epsilon": epsilon,
            }

            metrics_history.append({"episode": episode + 1, "metrics": metrics})

            logger.log_metrics(episode + 1, metrics)
            logger.log(f"  AI先手胜率: {metrics_first['win_rate']:.2%}")
            logger.log(f"  AI后手胜率: {metrics_second['win_rate']:.2%}")

            # 保存最佳模型
            current_best = max(best_win_rate_rule, metrics["win_rate_rule"])
            if current_best > best_win_rate_rule + 0.01:
                best_win_rate_rule = current_best
                with open(model_save_path, "wb") as f:
                    pickle.dump(nn, f)
                logger.log(f"新最佳模型已保存! 规则对手胜率: {best_win_rate_rule:.2%}")

            # 保存检查点
            state = TrainingState(
                episode=episode + 1,
                epsilon=epsilon,
                best_win_rate_rule=best_win_rate_rule,
                best_win_rate_random=best_win_rate_random,
                learning_rate=learning_rate,
                total_episodes=total_episodes,
                phase=0,
                metrics_history=metrics_history[-10:],  # 只保留最近10条
                timestamp=logger._get_timestamp(),
            )
            checkpoint_manager.save_training_checkpoint(nn, state)

            # 检查是否达到目标
            if metrics["win_rate_rule"] >= target_win_rate_rule:
                logger.log(f"目标胜率 ({target_win_rate_rule:.2%}) 已达成!")
                break

    logger.log("=" * 60)
    logger.log("训练完成!")
    logger.log(f"最佳规则对手胜率: {best_win_rate_rule:.2%}")
    logger.log("=" * 60)

    return nn


def evaluate_with_firstmover(
    nn: NeuralNetwork,
    opponent,
    num_games: int = 1000,
    ai_first: bool = True,
) -> Dict[str, float]:
    """
    评估AI在特定先手条件下的性能

    参数:
        nn (NeuralNetwork): 神经网络模型
        opponent: 对手（Opponent 对象或 callable(game) -> int）
        num_games (int): 评估游戏数
        ai_first (bool): AI是否先手
    """
    wins = 0
    draws = 0
    losses = 0

    ai_player = "X" if ai_first else "O"

    for _ in range(num_games):
        game = TicTacToeGame()

        while True:
            over, _ = game.is_game_over()
            if over:
                break

            if game.get_symbol(game.current_player) == ai_player:
                inputs = board_to_input(game.get_board_2d())
                move_index = nn.get_best_move(inputs, game.get_board_2d())
            else:
                if hasattr(opponent, "get_move"):
                    move_index = opponent.get_move(game)
                else:
                    move_index = opponent(game)

            game.make_move(move_index)

        winner = game.check_winner()
        if winner == ai_player:
            wins += 1
        elif winner is not None:
            losses += 1
        else:
            draws += 1

    total = num_games
    return {
        "win_rate": wins / total,
        "draw_rate": draws / total,
        "loss_rate": losses / total,
        "ai_first": ai_first,
    }


def comprehensive_evaluation(
    nn: NeuralNetwork, num_games: int = 2000
) -> Dict[str, Dict[str, float]]:
    """
    全面评估神经网络性能（分别评估先手和后的情况）

    参数:
        nn (NeuralNetwork): 神经网络模型
        num_games (int): 每种情况的评估游戏数

    返回:
        Dict[str, Dict[str, float]]: 详细的性能指标
    """
    results = {}

    # 评估对随机对手
    print("评估对随机对手（AI先手）...")
    results["random_first"] = evaluate_with_firstmover(
        nn, RandomOpponent(), num_games // 2, ai_first=True
    )

    print("评估对随机对手（AI后手）...")
    results["random_second"] = evaluate_with_firstmover(
        nn, RandomOpponent(), num_games // 2, ai_first=False
    )

    # 评估对规则对手
    print("评估对规则对手（AI先手）...")
    results["rule_first"] = evaluate_with_firstmover(
        nn, RuleBasedOpponent(), num_games // 2, ai_first=True
    )

    print("评估对规则对手（AI后手）...")
    results["rule_second"] = evaluate_with_firstmover(
        nn, RuleBasedOpponent(), num_games // 2, ai_first=False
    )

    # 计算综合指标
    results["summary"] = {
        "avg_win_rate_random": (
            results["random_first"]["win_rate"] + results["random_second"]["win_rate"]
        )
        / 2,
        "avg_win_rate_rule": (
            results["rule_first"]["win_rate"] + results["rule_second"]["win_rate"]
        )
        / 2,
        "win_rate_first": results["rule_first"]["win_rate"],
        "win_rate_second": results["rule_second"]["win_rate"],
    }

    return results


def print_evaluation_report(results: Dict[str, Dict[str, float]]):
    """
    打印评估报告

    参数:
        results (Dict): 评估结果
    """
    print("\n" + "=" * 60)
    print("综合评估报告")
    print("=" * 60)

    print("\n对随机对手:")
    print(
        f"  AI先手 - 胜率: {results['random_first']['win_rate']:.2%}, 平局: {results['random_first']['draw_rate']:.2%}, 失败: {results['random_first']['loss_rate']:.2%}"
    )
    print(
        f"  AI后手 - 胜率: {results['random_second']['win_rate']:.2%}, 平局: {results['random_second']['draw_rate']:.2%}, 失败: {results['random_second']['loss_rate']:.2%}"
    )
    print(f"  平均胜率: {results['summary']['avg_win_rate_random']:.2%}")

    print("\n对规则对手:")
    print(
        f"  AI先手 - 胜率: {results['rule_first']['win_rate']:.2%}, 平局: {results['rule_first']['draw_rate']:.2%}, 失败: {results['rule_first']['loss_rate']:.2%}"
    )
    print(
        f"  AI后手 - 胜率: {results['rule_second']['win_rate']:.2%}, 平局: {results['rule_second']['draw_rate']:.2%}, 失败: {results['rule_second']['loss_rate']:.2%}"
    )
    print(f"  平均胜率: {results['summary']['avg_win_rate_rule']:.2%}")

    print("\n" + "=" * 60)
    print(f"AI先手胜率: {results['summary']['win_rate_first']:.2%}")
    print(f"AI后手胜率: {results['summary']['win_rate_second']:.2%}")
    print("=" * 60 + "\n")


def parse_enhanced_training_args():
    """
    解析增强版训练命令行参数

    返回:
        argparse.Namespace: 解析后的参数
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="井字棋AI增强版训练脚本（支持检查点和先手控制）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 训练轮次参数
    parser.add_argument(
        "-e",
        "--episodes",
        type=int,
        default=TrainingConfig.DEFAULT_TOTAL_EPISODES,
        help="训练总轮次 (total_episodes)",
    )

    # 模型路径参数
    parser.add_argument(
        "-m",
        "--model-path",
        type=str,
        default=None,
        help=f"模型保存路径 (默认: {DEFAULT_MODEL_PATH})",
    )

    # AI先手比例
    parser.add_argument(
        "--ai-first-ratio", type=float, default=0.5, help="AI作为先手的比例 (0.0-1.0)"
    )

    # 对手类型
    parser.add_argument(
        "--opponent-type",
        type=str,
        choices=["random", "rule", "mixed", "selfplay"],
        default="mixed",
        help="对手类型: random(随机), rule(规则), mixed(混合), selfplay(自我对弈)",
    )

    # 学习率参数
    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        default=TrainingConfig.DEFAULT_LEARNING_RATE,
        help="学习率",
    )

    # 评估间隔参数
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=TrainingConfig.DEFAULT_EVAL_INTERVAL,
        help="评估间隔（轮次）",
    )

    # 日志间隔参数
    parser.add_argument(
        "--log-interval",
        type=int,
        default=TrainingConfig.DEFAULT_LOG_INTERVAL,
        help="日志记录间隔（轮次）",
    )

    # 目标胜率参数
    parser.add_argument(
        "--target-win-rate",
        type=float,
        default=TrainingConfig.DEFAULT_TARGET_WIN_RATE,
        help="目标胜率（对规则对手）",
    )

    # 评估游戏数参数
    parser.add_argument(
        "--num-eval-games",
        type=int,
        default=TrainingConfig.DEFAULT_NUM_EVAL_GAMES,
        help="每次评估的游戏数",
    )

    # 检查点目录
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints", help="检查点保存目录"
    )

    # 从检查点恢复
    parser.add_argument("--resume", type=str, default=None, help="从指定检查点恢复训练")

    return parser.parse_args()


if __name__ == "__main__":
    # 确保模型目录存在
    ModelConfig.ensure_models_dir_exists()

    # 解析命令行参数
    args = parse_enhanced_training_args()

    # 解析模型路径
    model_path = resolve_model_path(args.model_path)
    ensure_dir_for_file(model_path)

    print("=" * 60)
    print("增强版训练（带检查点和先手控制）")
    print("=" * 60)
    print(f"训练轮次: {args.episodes}")
    print(f"模型路径: {model_path}")
    print(f"AI先手比例: {args.ai_first_ratio:.0%}")
    print(f"对手类型: {args.opponent_type}")
    print(f"学习率: {args.learning_rate}")
    print(f"评估间隔: {args.eval_interval}")
    print(f"目标胜率: {args.target_win_rate:.0%}")
    print(f"检查点目录: {args.checkpoint_dir}")
    print("=" * 60)

    # 创建神经网络
    nn = NeuralNetwork()

    # 配置先手策略
    firstmover_config = FirstMoverConfig(ai_first_ratio=args.ai_first_ratio)

    # 创建检查点管理器
    cm = create_checkpoint_manager(
        checkpoint_dir=args.checkpoint_dir,
        max_checkpoints=5,
        auto_save_interval=args.eval_interval,
    )

    # 开始训练
    print("\n开始训练...")
    nn = train_with_checkpoint_and_firstmover(
        nn,
        total_episodes=args.episodes,
        learning_rate=args.learning_rate,
        checkpoint_manager=cm,
        firstmover_config=firstmover_config,
        opponent_type=args.opponent_type,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        target_win_rate_rule=args.target_win_rate,
        num_eval_games=args.num_eval_games,
        model_save_path=model_path,
        resume_from_checkpoint=args.resume,
    )

    # 全面评估
    print("\n进行全面评估...")
    results = comprehensive_evaluation(nn, num_games=args.num_eval_games)
    print_evaluation_report(results)
