# 井字棋 AI 对战系统

基于强化学习的井字棋（Tic-Tac-Toe）人工智能对战系统，支持人机对战、人人对战和机机对战模式。

参考：[图解神经网络和强化学习：400 行 C 代码训练一个井字棋高手](https://arthurchiao.art/blog/reinforcement-learning-400-lines-of-c-code-zh/)

---

## 功能特性

- **AI 对战**：基于神经网络训练的 AI 对手（100,000 局训练，vs 随机对手 89% 胜率）
- **多种模式**：人机对战、人人对战、机机对战
- **难度选择**：简单、中等、困难（控制 AI 探索率）
- **先手选择**：可配置 AI 先手或后手
- **检查点功能**：训练中断恢复 + 游戏状态保存
- **Kivy GUI**：跨平台图形界面，支持自定义中文字体

---

## 项目结构

```
tic_tac_toe/
├── src/
│   ├── core/
│   │   ├── game.py          # 统一游戏引擎 (GameEngine)
│   │   ├── common.py        # 棋盘定义 (GameBoard)
│   │   ├── checkpoint.py    # 检查点系统
│   │   └── config.py        # 统一配置
│   ├── ai/
│   │   ├── train.py         # 神经网络 + 基础训练
│   │   ├── train_enhanced.py # 增强训练（检查点 + 先手控制）
│   │   └── opponents.py     # Opponent 协议 + 策略类
│   └── gui/
│       ├── shared.py        # 共享 GUI 组件 (AIManager, GameGrid)
│       ├── main.py          # 标准版入口
│       └── main_enhanced.py # 增强版入口（模式/难度选择）
├── tests/
│   ├── test_game_engine.py  # GameEngine + Opponent + 集成测试
│   ├── test_game_logic.py   # 神经网络 + 训练逻辑测试
│   └── test_checkpoint.py   # 检查点系统测试
├── scripts/
│   └── benchmark_checkpoint.py
├── models/                  # 训练好的模型 (.pkl)
├── requirements.txt
└── setup_env.sh
```

---

## 快速开始

### 环境配置

```bash
# 安装 uv（如未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建虚拟环境并安装依赖
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 运行游戏

```bash
# 标准版
python -m src.gui.main

# 增强版（推荐，支持模式/难度/先手选择）
python -m src.gui.main_enhanced
```

### 训练 AI

```bash
# 基础训练（100,000 局，~30 秒）
python -m src.ai.train -e 100000

# 增强版训练（含检查点 + 先手控制）
python -m src.ai.train_enhanced -e 100000 --opponent-type mixed

# 查看所有参数
python -m src.ai.train --help
python -m src.ai.train_enhanced --help
```

### 运行测试

```bash
python -m pytest tests/ -v
```

---

## 架构设计

### 核心模块

| 模块 | 职责 |
|------|------|
| `GameEngine` | 统一游戏逻辑：落子、胜负判定、合法移动、克隆 |
| `NeuralNetwork` | 三层全连接网络 (18→100→9)，ReLU + Softmax |
| `Opponent` 协议 | 统一对手接口，支持 Random / RuleBased / NeuralNet |
| `CheckpointManager` | 训练和游戏状态的保存、加载、自动清理 |

### 训练流程

```
GameEngine ←→ NeuralNetwork (epsilon-greedy)
      ↑
Opponent (Random / RuleBased / NeuralNet)
      ↓
learn_from_game() → backward()
```

AI 通过自我对弈和对抗不同对手学习最优策略，支持课程学习、混合对手训练。

### Opponent 策略

- `RandomOpponent`：随机选择合法移动
- `RuleBasedOpponent`：优先赢 → 堵对手 → 占中心 → 随机
- `NeuralNetOpponent`：使用神经网络预测，支持 epsilon-greedy 探索

---

## 命令行参数

### train.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-e, --episodes` | 100000 | 训练局数 |
| `-m, --model-path` | models/tic_tac_toe_ai_model.pkl | 模型保存路径 |

### train_enhanced.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-e, --episodes` | 100000 | 训练轮次 |
| `--ai-first-ratio` | 0.5 | AI 先手比例 |
| `--opponent-type` | mixed | 对手类型: random/rule/mixed/selfplay |
| `--learning-rate` | 0.001 | 学习率 |
| `--eval-interval` | 25000 | 评估间隔 |
| `--checkpoint-dir` | checkpoints | 检查点目录 |
| `--resume` | None | 从检查点恢复 |

---

## 测试覆盖

107 个测试，覆盖：

- **GameEngine**：初始化、落子、胜负判定、克隆、游戏结束后拒绝落子
- **Opponent 协议**：RandomOpponent / RuleBasedOpponent / NeuralNetOpponent 各自逻辑
- **NeuralNetwork**：前向传播、反向传播、softmax、模型保存/加载
- **训练集成**：增强训练烟雾测试、评估函数、综合评估
- **CheckpointManager**：保存/加载训练和游戏检查点、自动清理、先手配置

```bash
# 运行全部测试
python -m pytest tests/ -v

# 只运行 GameEngine 测试
python -m pytest tests/test_game_engine.py -v
```

---

## 系统要求

- Python 3.10+
- 依赖：numpy, kivy
- 内存：至少 512MB RAM

---

## 许可证

MIT License
