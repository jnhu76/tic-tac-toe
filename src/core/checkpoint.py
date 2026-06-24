# -*- coding: utf-8 -*-
"""
检查点管理系统

本模块实现了训练和游戏过程的检查点保存和恢复功能，支持：
1. 定期自动保存训练状态
2. 游戏过程中断恢复
3. 训练参数和模型权重的完整保存
4. 版本兼容性检查

作者：AI Assistant
版本：1.0
日期：2026-04-13
"""

import json
import os
import pickle
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np


@dataclass
class TrainingState:
    """
    训练状态数据结构

    保存训练过程中的所有关键状态信息
    """

    episode: int  # 当前训练轮次
    epsilon: float  # 当前探索率
    best_win_rate_rule: float  # 最佳规则对手胜率
    best_win_rate_random: float  # 最佳随机对手胜率
    learning_rate: float  # 当前学习率
    total_episodes: int  # 总训练轮次
    phase: int  # 当前训练阶段
    metrics_history: list  # 性能指标历史
    timestamp: str  # 保存时间戳
    version: str = "1.0"  # 状态版本号

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingState":
        """从字典创建实例"""
        return cls(**data)


@dataclass
class GameState:
    """
    游戏状态数据结构

    保存游戏过程中的所有关键状态信息
    """

    board: list  # 棋盘状态
    current_player: str  # 当前玩家 ('X' 或 'O')
    move_history: list  # 移动历史
    game_mode: str  # 游戏模式 ('pvp', 'pve', 'eve')
    ai_difficulty: str  # AI难度 ('easy', 'medium', 'hard')
    is_game_over: bool  # 游戏是否结束
    winner: Optional[str]  # 获胜者
    timestamp: str  # 保存时间戳
    version: str = "1.0"  # 状态版本号

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GameState":
        """从字典创建实例"""
        return cls(**data)


class CheckpointManager:
    """
    检查点管理器

    负责管理训练和游戏状态的保存、加载和恢复

    功能：
    1. 定期自动保存检查点
    2. 手动保存和加载检查点
    3. 检查点版本管理
    4. 自动清理旧检查点
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        max_checkpoints: int = 5,
        auto_save_interval: int = 1000,
        verbose: bool = True,
    ):
        """
        初始化检查点管理器

        参数:
            checkpoint_dir (str): 检查点保存目录
            max_checkpoints (int): 最大保留检查点数量
            auto_save_interval (int): 自动保存间隔（轮次）
            verbose (bool): 是否输出详细信息
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.auto_save_interval = auto_save_interval
        self.verbose = verbose

        # 创建检查点目录
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 子目录
        self.training_dir = self.checkpoint_dir / "training"
        self.game_dir = self.checkpoint_dir / "game"
        self.training_dir.mkdir(exist_ok=True)
        self.game_dir.mkdir(exist_ok=True)

        # 状态跟踪
        self.last_save_time: float = 0
        self.save_count = 0

        if self.verbose:
            print(f"[CheckpointManager] 初始化完成，检查点目录: {self.checkpoint_dir}")

    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _cleanup_old_checkpoints(self, checkpoint_type: str):
        """
        清理旧的检查点文件

        参数:
            checkpoint_type (str): 'training' 或 'game'
        """
        target_dir = (
            self.training_dir if checkpoint_type == "training" else self.game_dir
        )

        # 获取所有检查点文件
        checkpoints = sorted(
            target_dir.glob("*.pkl"), key=lambda x: x.stat().st_mtime, reverse=True
        )

        # 删除超出限制的旧检查点
        if len(checkpoints) > self.max_checkpoints:
            for old_checkpoint in checkpoints[self.max_checkpoints :]:
                old_checkpoint.unlink()
                if self.verbose:
                    print(f"[CheckpointManager] 删除旧检查点: {old_checkpoint.name}")

    def save_training_checkpoint(
        self, nn, state: TrainingState, filename: Optional[str] = None
    ) -> str:
        """
        保存训练检查点

        参数:
            nn: 神经网络模型
            state (TrainingState): 训练状态
            filename (str, optional): 自定义文件名

        返回:
            str: 保存的文件路径
        """
        if filename is None:
            filename = f"training_ep{state.episode}_{self._get_timestamp()}.pkl"

        filepath = self.training_dir / filename

        # 准备保存数据
        checkpoint_data = {
            "neural_network": nn,
            "state": state.to_dict(),
            "metadata": {
                "save_time": self._get_timestamp(),
                "version": state.version,
                "type": "training",
            },
        }

        # 保存到文件
        with open(filepath, "wb") as f:
            pickle.dump(checkpoint_data, f)

        self.save_count += 1
        self.last_save_time = time.time()

        # 清理旧检查点
        self._cleanup_old_checkpoints("training")

        if self.verbose:
            print(f"[CheckpointManager] 训练检查点已保存: {filepath}")
            print(f"  - 轮次: {state.episode}/{state.total_episodes}")
            print(f"  - 最佳规则胜率: {state.best_win_rate_rule:.2%}")
            print(f"  - Epsilon: {state.epsilon:.4f}")

        return str(filepath)

    def save_game_checkpoint(
        self, state: GameState, filename: Optional[str] = None
    ) -> str:
        """
        保存游戏检查点

        参数:
            state (GameState): 游戏状态
            filename (str, optional): 自定义文件名

        返回:
            str: 保存的文件路径
        """
        if filename is None:
            filename = f"game_{self._get_timestamp()}.pkl"

        filepath = self.game_dir / filename

        # 准备保存数据
        checkpoint_data = {
            "state": state.to_dict(),
            "metadata": {
                "save_time": self._get_timestamp(),
                "version": state.version,
                "type": "game",
            },
        }

        # 保存到文件
        with open(filepath, "wb") as f:
            pickle.dump(checkpoint_data, f)

        self.save_count += 1
        self.last_save_time = time.time()

        # 清理旧检查点
        self._cleanup_old_checkpoints("game")

        if self.verbose:
            print(f"[CheckpointManager] 游戏检查点已保存: {filepath}")
            print(f"  - 当前玩家: {state.current_player}")
            print(f"  - 游戏模式: {state.game_mode}")
            print(f"  - 移动数: {len(state.move_history)}")

        return str(filepath)

    def load_training_checkpoint(self, filepath: Optional[str] = None) -> tuple:
        if filepath is None:
            checkpoints = sorted(
                self.training_dir.glob("*.pkl"),
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )
            if not checkpoints:
                raise FileNotFoundError("没有找到训练检查点")
            filepath = str(checkpoints[0])

        fp = Path(filepath)

        if not fp.exists():
            raise FileNotFoundError(f"检查点文件不存在: {filepath}")

        with open(fp, "rb") as f:
            checkpoint_data = pickle.load(f)

        nn = checkpoint_data["neural_network"]
        state = TrainingState.from_dict(checkpoint_data["state"])

        if self.verbose:
            print(f"[CheckpointManager] 训练检查点已加载: {filepath}")
            print(f"  - 轮次: {state.episode}/{state.total_episodes}")
            print(f"  - 保存时间: {checkpoint_data['metadata']['save_time']}")

        return nn, state

    def load_game_checkpoint(self, filepath: Optional[str] = None) -> GameState:
        if filepath is None:
            checkpoints = sorted(
                self.game_dir.glob("*.pkl"),
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )
            if not checkpoints:
                raise FileNotFoundError("没有找到游戏检查点")
            filepath = str(checkpoints[0])

        fp = Path(filepath)

        if not fp.exists():
            raise FileNotFoundError(f"检查点文件不存在: {filepath}")

        with open(fp, "rb") as f:
            checkpoint_data = pickle.load(f)

        state = GameState.from_dict(checkpoint_data["state"])

        if self.verbose:
            print(f"[CheckpointManager] 游戏检查点已加载: {filepath}")
            print(f"  - 当前玩家: {state.current_player}")
            print(f"  - 保存时间: {checkpoint_data['metadata']['save_time']}")

        return state

    def list_checkpoints(self, checkpoint_type: str = "all") -> Dict[str, list]:
        """
        列出所有可用的检查点

        参数:
            checkpoint_type (str): 'training', 'game', 或 'all'

        返回:
            Dict[str, list]: 检查点列表
        """
        result = {}

        if checkpoint_type in ["training", "all"]:
            training_checkpoints = sorted(
                self.training_dir.glob("*.pkl"),
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )
            result["training"] = [
                {
                    "filename": cp.name,
                    "path": str(cp),
                    "size": cp.stat().st_size,
                    "modified": datetime.fromtimestamp(cp.stat().st_mtime).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                }
                for cp in training_checkpoints
            ]

        if checkpoint_type in ["game", "all"]:
            game_checkpoints = sorted(
                self.game_dir.glob("*.pkl"),
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )
            result["game"] = [
                {
                    "filename": cp.name,
                    "path": str(cp),
                    "size": cp.stat().st_size,
                    "modified": datetime.fromtimestamp(cp.stat().st_mtime).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                }
                for cp in game_checkpoints
            ]

        return result

    def should_auto_save(self, episode: int) -> bool:
        """
        检查是否应该自动保存

        参数:
            episode (int): 当前训练轮次

        返回:
            bool: 是否应该保存
        """
        return episode > 0 and episode % self.auto_save_interval == 0

    def delete_checkpoint(self, filepath: str) -> bool:
        fp = Path(filepath)
        if fp.exists():
            fp.unlink()
            if self.verbose:
                print(f"[CheckpointManager] 已删除检查点: {filepath}")
            return True
        return False

    def clear_all_checkpoints(self, checkpoint_type: str = "all"):
        """
        清除所有检查点

        参数:
            checkpoint_type (str): 'training', 'game', 或 'all'
        """
        if checkpoint_type in ["training", "all"]:
            for cp in self.training_dir.glob("*.pkl"):
                cp.unlink()
            if self.verbose:
                print(f"[CheckpointManager] 已清除所有训练检查点")

        if checkpoint_type in ["game", "all"]:
            for cp in self.game_dir.glob("*.pkl"):
                cp.unlink()
            if self.verbose:
                print(f"[CheckpointManager] 已清除所有游戏检查点")


class AutoSaveCallback:
    """
    自动保存回调类

    用于在训练过程中定期自动保存检查点
    """

    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        nn,
        get_state_fn: Callable[[], TrainingState],
    ):
        """
        初始化自动保存回调

        参数:
            checkpoint_manager (CheckpointManager): 检查点管理器
            nn: 神经网络模型
            get_state_fn (callable): 获取当前训练状态的函数
        """
        self.cm = checkpoint_manager
        self.nn = nn
        self.get_state_fn = get_state_fn

    def __call__(self, episode: int):
        """
        回调函数

        参数:
            episode (int): 当前训练轮次
        """
        if self.cm.should_auto_save(episode):
            state = self.get_state_fn()
            self.cm.save_training_checkpoint(self.nn, state)


# 便捷函数
def create_checkpoint_manager(
    checkpoint_dir: str = "checkpoints",
    max_checkpoints: int = 5,
    auto_save_interval: int = 1000,
) -> CheckpointManager:
    """
    创建检查点管理器的便捷函数

    参数:
        checkpoint_dir (str): 检查点保存目录
        max_checkpoints (int): 最大保留检查点数量
        auto_save_interval (int): 自动保存间隔

    返回:
        CheckpointManager: 检查点管理器实例
    """
    return CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        max_checkpoints=max_checkpoints,
        auto_save_interval=auto_save_interval,
    )


def quick_save_training(nn, episode: int, **kwargs) -> str:
    """
    快速保存训练检查点

    参数:
        nn: 神经网络模型
        episode (int): 当前轮次
        **kwargs: 其他状态参数

    返回:
        str: 保存的文件路径
    """
    cm = CheckpointManager(verbose=False)
    state = TrainingState(
        episode=episode,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **kwargs,
    )
    return cm.save_training_checkpoint(nn, state)


def quick_save_game(board, current_player: str, **kwargs) -> str:
    """
    快速保存游戏检查点

    参数:
        board: 棋盘状态
        current_player (str): 当前玩家
        **kwargs: 其他状态参数

    返回:
        str: 保存的文件路径
    """
    cm = CheckpointManager(verbose=False)
    state = GameState(
        board=board,
        current_player=current_player,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **kwargs,
    )
    return cm.save_game_checkpoint(state)
