"""
井字棋项目统一配置模块

本模块定义了项目中的全局配置，包括：
1. 模型文件路径配置
2. 训练参数默认配置
3. 路径解析工具函数

作者：AI Assistant
版本：1.0
日期：2026-04-13
"""

import os
import sys
from pathlib import Path
from typing import Optional


# ==================== 项目根目录配置 ====================
def get_project_root() -> Path:
    """
    获取项目根目录

    返回:
        Path: 项目根目录路径
    """
    # 从当前文件位置向上回溯到项目根目录
    current_file = Path(__file__).resolve()
    # src/core/config.py -> src/core -> src -> project_root
    return current_file.parent.parent.parent


# ==================== 模型路径配置 ====================
class ModelConfig:
    """
    模型配置类

    统一管理所有模型文件的保存和加载路径
    """

    # 项目根目录
    PROJECT_ROOT: Path = get_project_root()

    # 模型存储目录
    MODELS_DIR: Path = PROJECT_ROOT / "models"

    # 默认模型文件名
    DEFAULT_MODEL_NAME: str = "tic_tac_toe_ai_model.pkl"
    BEST_MODEL_NAME: str = "best_model.pkl"

    # 完整模型路径（供训练保存和推理加载使用）
    @classmethod
    def get_default_model_path(cls) -> str:
        """获取默认模型路径"""
        return str(cls.MODELS_DIR / cls.DEFAULT_MODEL_NAME)

    @classmethod
    def get_best_model_path(cls) -> str:
        """获取最佳模型路径"""
        return str(cls.MODELS_DIR / cls.BEST_MODEL_NAME)

    @classmethod
    def ensure_models_dir_exists(cls) -> None:
        """确保模型目录存在"""
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ==================== 训练参数配置 ====================
class TrainingConfig:
    """
    训练参数配置类

    定义训练过程的默认参数，可通过命令行参数覆盖
    """

    # 默认训练轮次
    DEFAULT_TOTAL_EPISODES: int = 100000

    # 学习率
    DEFAULT_LEARNING_RATE: float = 0.001

    # 折扣因子
    DEFAULT_DISCOUNT_FACTOR: float = 0.95

    # 探索率参数
    DEFAULT_EPSILON_START: float = 1.0
    DEFAULT_EPSILON_END: float = 0.01
    DEFAULT_EPSILON_DECAY: float = 0.9998

    # 评估和日志间隔
    DEFAULT_LOG_INTERVAL: int = 10000
    DEFAULT_EVAL_INTERVAL: int = 25000

    # 目标胜率
    DEFAULT_TARGET_WIN_RATE: float = 0.50

    # 评估游戏数
    DEFAULT_NUM_EVAL_GAMES: int = 2000

    # 检查点配置
    DEFAULT_CHECKPOINT_DIR: str = "checkpoints"
    DEFAULT_MAX_CHECKPOINTS: int = 5


# ==================== 便捷导入路径 ====================
# 供其他模块直接导入使用
DEFAULT_MODEL_PATH = ModelConfig.get_default_model_path()
BEST_MODEL_PATH = ModelConfig.get_best_model_path()


# ==================== 路径解析函数 ====================
def resolve_model_path(model_path: Optional[str] = None) -> str:
    """
    解析模型路径

    如果未提供路径，返回默认路径；
    如果提供的是相对路径，基于项目根目录解析；
    如果提供的是绝对路径，直接返回。

    参数:
        model_path (Optional[str]): 用户提供的模型路径

    返回:
        str: 解析后的完整路径
    """
    if model_path is None:
        return DEFAULT_MODEL_PATH

    path = Path(model_path)

    # 如果是绝对路径，直接返回
    if path.is_absolute():
        return str(path)

    # 如果是相对路径，基于项目根目录解析
    # 检查是否以 models/ 开头
    if path.parts[0] == "models":
        return str(ModelConfig.PROJECT_ROOT / path)

    # 默认放入 models 目录
    return str(ModelConfig.MODELS_DIR / path)


def ensure_dir_for_file(file_path: str) -> None:
    """
    确保文件所在目录存在

    参数:
        file_path (str): 文件路径
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
