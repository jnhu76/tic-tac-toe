# 井字棋AI项目 - Claude开发指南

## 项目概述

本项目实现了一个基于强化学习的井字棋AI系统，包含神经网络训练模块和图形用户界面。AI通过自我对弈和对抗不同类型的对手来学习最优策略。

### 项目结构

```
tic_tac_toe/
├── src/                          # 源代码目录
│   ├── __init__.py               # 包初始化文件
│   ├── core/                     # 核心模块
│   │   ├── __init__.py
│   │   ├── common.py             # 共享模块（游戏板定义）
│   │   └── checkpoint.py         # 检查点系统核心模块
│   ├── ai/                       # AI相关模块
│   │   ├── __init__.py
│   │   ├── train.py              # 神经网络训练模块
│   │   └── train_enhanced.py     # 增强版训练模块（含检查点和先手控制）
│   └── gui/                      # 界面模块
│       ├── __init__.py
│       ├── main.py               # 游戏主程序（GUI）
│       └── main_enhanced.py      # 增强版游戏主程序
├── tests/                        # 测试目录
│   ├── __init__.py
│   └── test_checkpoint.py        # 测试套件
├── scripts/                      # 脚本目录
│   └── benchmark_checkpoint.py   # 性能评估脚本
├── docs/                         # 文档目录
│   ├── CHECKPOINT_DESIGN.md      # 检查点系统设计文档
│   └── IMPLEMENTATION_SUMMARY.md # 实现总结
├── assets/
│   └── fonts/
│       └── SourceHanSansSC-Normal-2.otf  # 中文字体
├── best_model.pkl                # 训练好的AI模型
├── training_log.txt              # 训练日志
├── requirements.txt              # Python依赖
├── setup_env.sh                  # 环境配置脚本
└── readme.md                     # 项目说明文档
```

## 快速开始

### 环境配置

```bash
# 方式1：使用环境配置脚本
source setup_env.sh

# 方式2：手动配置
uv venv .venv
source ./.venv/bin/activate
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
uv pip install -r requirements.txt
```

### 运行程序

```bash
# 运行标准版游戏
python -m src.gui.main

# 运行增强版游戏（推荐）
python -m src.gui.main_enhanced

# 训练AI
python -m src.ai.train

# 运行测试
python -m tests.test_checkpoint

# 性能评估
python scripts/benchmark_checkpoint.py
```

## 核心模块详解

### 1. src/core/common.py - 游戏基础模块

#### GameBoard类
```python
from src.core.common import GameBoard

# 创建游戏板
board = GameBoard()

# 属性
board.board          # 3x3棋盘状态矩阵
board.current_player # 当前玩家（0=AI, 1=人类）
board.col            # 列数
board.row            # 行数
board.board_size     # 棋盘总大小
```

### 2. src/core/config.py - 统一配置系统

#### ModelConfig 类
```python
from src.core.config import ModelConfig, resolve_model_path

# 获取默认模型路径（models/tic_tac_toe_ai_model.pkl）
model_path = ModelConfig.get_default_model_path()

# 获取最佳模型路径（models/best_model.pkl）
best_path = ModelConfig.get_best_model_path()

# 确保模型目录存在
ModelConfig.ensure_models_dir_exists()

# 解析模型路径（支持相对路径和绝对路径）
path = resolve_model_path("my_model.pkl")  # 解析为 models/my_model.pkl
path = resolve_model_path("/absolute/path/model.pkl")  # 保持绝对路径
```

#### TrainingConfig 类
```python
from src.core.config import TrainingConfig

# 默认训练参数
episodes = TrainingConfig.DEFAULT_TOTAL_EPISODES  # 100000
learning_rate = TrainingConfig.DEFAULT_LEARNING_RATE  # 0.001
eval_interval = TrainingConfig.DEFAULT_EVAL_INTERVAL  # 25000
```

### 3. src/core/checkpoint.py - 检查点系统

#### CheckpointManager类
```python
from src.core.checkpoint import CheckpointManager

# 创建检查点管理器
cm = CheckpointManager(
    checkpoint_dir="checkpoints",
    max_checkpoints=5,
    auto_save_interval=1000
)

# 保存训练检查点
cm.save_training_checkpoint(nn, episode=1000, epsilon=0.5, ...)

# 加载检查点
state, nn = cm.load_training_checkpoint(filepath)

# 保存游戏检查点
cm.save_game_checkpoint(board, current_player, ...)

# 列出所有检查点
checkpoints = cm.list_checkpoints()
```

#### TrainingState和GameState
```python
from src.core.checkpoint import TrainingState, GameState

# 训练状态
training_state = TrainingState(
    episode=1000,
    epsilon=0.5,
    best_win_rate_rule=0.6,
    ...
)

# 游戏状态
game_state = GameState(
    board=[['X', None, 'O'], ...],
    current_player='X',
    ...
)
```

### 4. src/ai/train.py - 训练模块

#### 命令行参数支持
```bash
# 查看所有可用参数
python -m src.ai.train --help

# 常用参数示例
python -m src.ai.train -e 200000                    # 训练20万轮次
python -m src.ai.train -m models/my_model.pkl       # 指定模型路径
python -m src.ai.train -lr 0.002                    # 设置学习率
python -m src.ai.train --mode selfplay              # 使用自我对弈模式
python -m src.ai.train --resume checkpoints/ep10000.pkl  # 从检查点恢复

# 完整参数
# -e, --episodes: 训练总轮次 (默认: 100000)
# -m, --model-path: 模型保存路径
# -lr, --learning-rate: 学习率 (默认: 0.001)
# --eval-interval: 评估间隔 (默认: 25000)
# --log-interval: 日志间隔 (默认: 10000)
# --target-win-rate: 目标胜率 (默认: 0.5)
# --num-eval-games: 评估游戏数 (默认: 2000)
# --mode: 训练模式 [curriculum|selfplay|mixed]
# --resume: 从检查点恢复训练
```

#### NeuralNetwork类
```python
from src.ai.train import NeuralNetwork

# 创建神经网络
nn = NeuralNetwork(
    input_size=18,      # 输入层大小
    hidden_size=128,    # 隐藏层大小
    output_size=9       # 输出层大小
)

# 前向传播
output = nn.forward(inputs)

# 获取最佳移动
move_index = nn.get_best_move(inputs, board, strategy='greedy', epsilon=0.1)
```

#### 训练函数
```python
from src.ai.train import train_with_self_play_and_curriculum

# 增强版训练（推荐）
nn = train_with_self_play_and_curriculum(
    nn,
    total_episodes=3000000,
    learning_rate=0.001,
    discount_factor=0.95,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.9998,
    self_play_ratio=0.3,
    rule_opponent_ratio=0.5,
    target_win_rate_rule=0.50
)
```

#### 评估函数
```python
from src.ai.train import evaluate_neural_network, load_model, save_model

# 加载模型
nn = load_model('best_model.pkl')

# 全面评估
metrics = evaluate_neural_network(nn, num_games=2000)
print(f"对随机对手胜率: {metrics['win_rate_random']:.2%}")
print(f"对规则对手胜率: {metrics['win_rate_rule']:.2%}")

# 保存模型
save_model(nn, 'best_model.pkl')
```

### 5. src/ai/train_enhanced.py - 增强版训练模块

#### 命令行参数支持
```bash
# 查看所有可用参数
python -m src.ai.train_enhanced --help

# 常用参数示例
python -m src.ai.train_enhanced -e 150000                              # 训练15万轮次
python -m src.ai.train_enhanced --ai-first-ratio 0.7                   # AI 70%先手
python -m src.ai.train_enhanced --opponent-type rule                   # 使用规则对手
python -m src.ai.train_enhanced --checkpoint-dir my_checkpoints        # 自定义检查点目录

# 完整参数（除train.py参数外）
# --ai-first-ratio: AI先手比例 0.0-1.0 (默认: 0.5)
# --opponent-type: 对手类型 [random|rule|mixed|selfplay] (默认: mixed)
# --checkpoint-dir: 检查点保存目录 (默认: checkpoints)
```

#### FirstMoverConfig类
```python
from src.ai.train_enhanced import FirstMoverConfig

# AI 50%概率先手
config = FirstMoverConfig(ai_first_ratio=0.5)

# AI始终先手
config = FirstMoverConfig(fixed_first_player='X')

# AI始终后手
config = FirstMoverConfig(fixed_first_player='O')

# 交替先手
config = FirstMoverConfig(alternate_turns=True)

# 获取先手玩家
first_player = config.get_first_player(episode=100)
```

#### 增强版训练函数
```python
from src.ai.train_enhanced import train_with_checkpoint_and_firstmover
from src.core.checkpoint import CheckpointManager

# 创建检查点管理器
cm = CheckpointManager()

# 增强版训练
nn = train_with_checkpoint_and_firstmover(
    nn,
    total_episodes=3000000,
    checkpoint_manager=cm,
    firstmover_config=FirstMoverConfig(ai_first_ratio=0.5),
    resume_from_checkpoint=None,  # 或指定检查点路径恢复训练
    opponent_type='mixed',
    self_play_ratio=0.3,
    rule_opponent_ratio=0.5
)
```

### 6. src/gui/main.py - GUI应用程序

#### 运行方式
```bash
python -m src.gui.main
```

#### 主要组件
- **AIManager**: AI管理器，加载模型并预测移动
- **GameGrid**: 游戏网格UI，处理点击事件
- **TicTacToeApp**: 主应用程序类

### 7. src/gui/main_enhanced.py - 增强版GUI

#### 运行方式
```bash
python -m src.gui.main_enhanced
```

#### 新增功能
- 游戏模式选择（人机/人人/机机）
- 先手选择（玩家/AI）
- AI难度选择（简单/中等/困难）
- 游戏状态自动保存和加载
- 检查点功能支持

## 训练流程指南

### 基础训练

```python
from src.ai.train import NeuralNetwork, train_with_self_play_and_curriculum

# 创建神经网络
nn = NeuralNetwork()

# 开始训练
nn = train_with_self_play_and_curriculum(
    nn,
    total_episodes=3000000,
    learning_rate=0.001,
    discount_factor=0.95,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.9998,
    self_play_ratio=0.3,
    rule_opponent_ratio=0.5,
    target_win_rate_rule=0.50,
    model_save_path='best_model.pkl'
)
```

### 带检查点的训练

```python
from src.ai.train import NeuralNetwork
from src.ai.train_enhanced import train_with_checkpoint_and_firstmover, FirstMoverConfig
from src.core.checkpoint import CheckpointManager

# 创建组件
nn = NeuralNetwork()
cm = CheckpointManager(checkpoint_dir="checkpoints")
config = FirstMoverConfig(ai_first_ratio=0.5)

# 训练（支持中断恢复）
nn = train_with_checkpoint_and_firstmover(
    nn,
    total_episodes=3000000,
    checkpoint_manager=cm,
    firstmover_config=config,
    resume_from_checkpoint=None  # 首次训练设为None
)
```

### 从检查点恢复训练

```python
# 从特定检查点恢复
nn = train_with_checkpoint_and_firstmover(
    nn,
    checkpoint_manager=cm,
    resume_from_checkpoint="checkpoints/training/training_ep10000_xxx.pkl"
)

# 或自动加载最新检查点
latest = cm.get_latest_checkpoint('training')
if latest:
    nn = train_with_checkpoint_and_firstmover(
        nn,
        checkpoint_manager=cm,
        resume_from_checkpoint=latest['filepath']
    )
```

## 高级训练技巧

### 1. 调整超参数

**学习率调整**:
- 初始学习率: 0.001（推荐）
- 如果训练不稳定，降低到0.0005
- 如果收敛太慢，提高到0.002

**折扣因子**:
- 推荐值: 0.95
- 较高值（0.97-0.99）：更重视长期奖励
- 较低值（0.90-0.94）：更重视短期奖励

**探索率衰减**:
- 推荐值: 0.9995-0.9998
- 较慢衰减：更多探索，训练时间更长
- 较快衰减：更快收敛，可能错过最优策略

### 2. 选择训练策略

**课程学习**:
- 适合：从零开始训练
- 优点：逐步增加难度，稳定收敛
- 缺点：训练时间较长

**混合对手**:
- 适合：快速提升对规则对手的性能
- 优点：平衡不同对手类型
- 缺点：可能对随机对手性能下降

**自我对弈**:
- 适合：追求最优策略
- 优点：发现新策略，不依赖对手质量
- 缺点：训练不稳定，需要更多轮次

### 3. 监控训练过程

**关键指标**:
- `win_rate_random`: 对随机对手胜率（目标>80%）
- `win_rate_rule`: 对规则对手胜率（目标>50%）
- `epsilon`: 当前探索率

**训练日志示例**:
```
[2026-04-13 14:30:15] Episode 25000: WR_random=75.00%, WR_rule=45.00%, epsilon=0.0821
```

## 检查点功能使用

### 训练检查点

```python
from src.core.checkpoint import CheckpointManager, create_checkpoint_manager

# 创建管理器
cm = create_checkpoint_manager(
    checkpoint_dir="checkpoints",
    max_checkpoints=5,
    auto_save_interval=1000
)

# 手动保存
cm.save_training_checkpoint(nn, episode=1000, epsilon=0.5, ...)

# 列出检查点
checkpoints = cm.list_checkpoints(checkpoint_type='training')

# 加载检查点
state, nn = cm.load_training_checkpoint(filepath)
```

### 游戏检查点

```python
# 保存游戏状态
cm.save_game_checkpoint(
    board=game_board,
    current_player='X',
    move_history=[0, 4, 8],
    game_mode='pve',
    ai_difficulty='medium'
)

# 加载游戏状态
state = cm.load_game_checkpoint(filepath)
```

## 测试

### 运行所有测试

```bash
python -m tests.test_checkpoint
```

### 详细输出

```bash
python -m tests.test_checkpoint -v
```

### 测试覆盖

- CheckpointManager初始化
- 训练检查点保存/加载
- 游戏检查点保存/加载
- 检查点自动清理
- 先手后手配置
- 集成测试

## 性能评估

```bash
python scripts/benchmark_checkpoint.py
```

评估内容：
- 检查点保存/加载性能
- 不同先手策略对训练的影响
- 系统资源占用情况

## 常见问题解决

### 1. 训练问题

**问题**: 训练过程中胜率波动很大
**解决**:
- 降低学习率
- 增加评估间隔
- 检查奖励设计

**问题**: 对规则对手胜率始终为0
**解决**:
- 增加对规则对手的训练比例
- 调整奖励函数，增加对获胜的奖励
- 延长训练时间

### 2. 导入错误

**问题**: `ModuleNotFoundError: No module named 'src'`
**解决**:
- 确保在项目根目录运行
- 使用 `python -m` 方式运行模块
- 检查 `src/__init__.py` 是否存在

### 3. GUI问题

**问题**: AI移动卡顿
**解决**:
- 使用Clock.schedule_once异步执行
- 优化模型推理速度
- 减少UI更新频率

## API参考

### 核心模块 (src/core/)

#### common.py
- `GameBoard`: 游戏板类
- `BOARD_SIZE`, `BOARD_COL`, `BOARD_ROW`: 常量

#### config.py
- `ModelConfig`: 模型配置类
  - `get_default_model_path()`: 获取默认模型路径
  - `get_best_model_path()`: 获取最佳模型路径
  - `ensure_models_dir_exists()`: 确保模型目录存在
- `TrainingConfig`: 训练参数配置类
  - `DEFAULT_TOTAL_EPISODES`: 默认训练轮次 (100000)
  - `DEFAULT_LEARNING_RATE`: 默认学习率 (0.001)
  - `DEFAULT_EVAL_INTERVAL`: 默认评估间隔 (25000)
- `resolve_model_path()`: 解析模型路径
- `ensure_dir_for_file()`: 确保文件目录存在

#### checkpoint.py
- `CheckpointManager`: 检查点管理器
- `TrainingState`: 训练状态数据类
- `GameState`: 游戏状态数据类
- `create_checkpoint_manager()`: 工厂函数

### AI模块 (src/ai/)

#### train.py
- `NeuralNetwork`: 神经网络类
- `train_with_self_play_and_curriculum()`: 训练函数
- `evaluate_neural_network()`: 评估函数
- `save_model()`, `load_model()`: 模型保存/加载

#### train_enhanced.py
- `FirstMoverConfig`: 先手配置类
- `train_with_checkpoint_and_firstmover()`: 增强版训练
- `evaluate_with_firstmover()`: 增强版评估

### GUI模块 (src/gui/)

#### main.py / main_enhanced.py
- `AIManager`: AI管理器
- `GameGrid`: 游戏网格
- `TicTacToeApp`: 主应用

## 开发最佳实践

### 1. 代码规范

- 使用类型注解
- 添加详细的中文注释
- 遵循PEP 8编码规范
- 保持函数单一职责

### 2. 版本控制

**提交信息格式**:
```
[类型] 简短描述

详细描述（可选）

影响范围：src/ai/train.py, src/core/checkpoint.py
```

**类型**:
- [新增] 新功能
- [修复] Bug修复
- [优化] 性能优化
- [文档] 文档更新
- [重构] 代码重构

## 部署指南

### 系统要求

**最低要求**:
- Python 3.8+
- NumPy 1.20+
- Kivy 2.0+（仅GUI）
- 内存: 512MB+
- 存储: 100MB+

**推荐配置**:
- Python 3.10+
- NumPy 1.24+
- Kivy 2.3+
- 内存: 1GB+
- 存储: 500MB+

### 运行方式

```bash
# 开发模式
python -m src.gui.main_enhanced

# 或使用旧版入口
python main_enhanced.py
```

## 未来改进方向

### 1. 算法改进

- [ ] 实现Deep Q-Network (DQN)
- [ ] 添加Actor-Critic算法
- [ ] 实现蒙特卡洛树搜索(MCTS)
- [ ] 添加注意力机制

### 2. 功能扩展

- [ ] 支持不同棋盘大小
- [ ] 添加难度级别选择
- [ ] 实现在线对战功能
- [ ] 添加游戏回放功能

### 3. 性能优化

- [ ] GPU加速训练
- [ ] 模型量化压缩
- [ ] 推理速度优化
- [ ] 内存使用优化
