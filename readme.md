# 井字棋 AI 对战系统 🤖✨

基于深度强化学习的井字棋（Tic-Tac-Toe）人工智能对战系统，支持人机对战、人人对战和机机对战模式。

参考：[图解神经网络和强化学习：400 行 C 代码训练一个井字棋高手（2025）](https://arthurchiao.art/blog/reinforcement-learning-400-lines-of-c-code-zh/)

---

## 功能特性 🚀

- 🤖 **AI 对战**：基于神经网络训练的 AI 对手
- 🎮 **多种模式**：支持人机对战、人人对战、机机对战
- 💾 **检查点功能**：支持训练中断恢复和游戏状态保存
- 🎯 **先手选择**：可配置 AI 先手或后手
- 📊 **难度选择**：简单、中等、困难三种 AI 难度
- 📝 **中文界面**：完整的中文界面和文档
- ✨ **基于 Kivy 的图形用户界面 (GUI)**
- 🔤 **自定义字体支持**：从 `assets/fonts/` 加载并使用自定义字体
- 📦 **跨平台支持**：包含 Android (`.apk`) 和桌面 (Windows/Linux) 打包说明

---

## 项目结构 📁

```
tic_tac_toe/
├── src/                       # 源代码目录
│   ├── __init__.py            # 包初始化文件
│   ├── core/                  # 核心模块
│   │   ├── __init__.py
│   │   ├── common.py          # 共享模块（游戏板定义）
│   │   └── checkpoint.py      # 检查点系统核心模块
│   ├── ai/                    # AI相关模块
│   │   ├── __init__.py
│   │   ├── train.py           # 神经网络训练模块
│   │   └── train_enhanced.py  # 增强版训练模块（含检查点和先手控制）
│   └── gui/                   # 界面模块
│       ├── __init__.py
│       ├── main.py            # 游戏主程序（GUI）
│       └── main_enhanced.py   # 增强版游戏主程序
├── tests/                     # 测试目录
│   ├── __init__.py
│   └── test_checkpoint.py     # 测试套件
├── scripts/                   # 脚本目录
│   └── benchmark_checkpoint.py # 性能评估脚本
├── docs/                      # 文档目录
│   ├── CHECKPOINT_DESIGN.md   # 检查点系统设计文档
│   └── IMPLEMENTATION_SUMMARY.md  # 实现总结
├── requirements.txt           # 项目依赖
├── setup_env.sh               # 环境配置脚本
└── .gitignore                 # Git 忽略规则
```

---

## 环境配置与依赖管理 🛠️

本项目使用 [uv](https://github.com/astral-sh/uv) 工具进行 Python 环境管理，uv 是一个极速的 Python 包管理器和环境管理工具。

### 1. 安装 uv 工具

如果尚未安装 uv，请执行以下命令：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

或参考官方文档：https://docs.astral.sh/uv/getting-started/installation/

### 2. 创建虚拟环境

在项目根目录下执行：

```bash
uv venv .venv
```

此命令将创建一个名为 `.venv` 的虚拟环境。

### 3. 激活虚拟环境

**Linux/macOS:**
```bash
source ./.venv/bin/activate
```

**Windows:**
```bash
.venv\Scripts\activate
```

### 4. 配置 PyPI 镜像源（推荐）

为提升依赖安装速度，建议配置清华大学 PyPI 镜像源：

```bash
# 配置 uv 使用清华大学镜像源
uv pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

或使用环境变量方式（临时生效）：
```bash
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
```

### 5. 安装依赖

激活虚拟环境后，安装项目依赖：

```bash
uv pip install -r requirements.txt
```

### 6. 导出依赖列表

当添加新依赖后，更新 requirements.txt：

```bash
uv pip freeze > requirements.txt
```

---

## 快速开始 ⚡

### 运行游戏

#### 标准版游戏界面
```bash
python -m src.gui.main
```

#### 增强版游戏界面（推荐）
```bash
python -m src.gui.main_enhanced
```

增强版支持：
- 游戏模式选择（人机/人人/机机）
- 先手选择（玩家/AI）
- AI 难度选择（简单/中等/困难）
- 游戏状态自动保存和加载

### 训练 AI

#### 标准训练
```bash
python -m src.ai.train
```

#### 增强版训练（含检查点和先手控制）
```bash
python -m src.ai.train_enhanced
```

#### 命令行参数

训练脚本支持丰富的命令行参数配置：

**基础参数：**
```bash
# 指定训练轮次（默认 100000）
python -m src.ai.train -e 200000

# 指定模型保存路径
python -m src.ai.train -m models/my_model.pkl

# 指定学习率
python -m src.ai.train -lr 0.002
```

**增强版训练参数：**
```bash
# AI 70% 先手，使用混合对手训练
python -m src.ai.train_enhanced -e 150000 --ai-first-ratio 0.7 --opponent-type mixed

# 从检查点恢复训练
python -m src.ai.train_enhanced --resume checkpoints/training_ep50000.pkl

# 自定义检查点目录
python -m src.ai.train_enhanced --checkpoint-dir my_checkpoints
```

**查看所有可用参数：**
```bash
python -m src.ai.train --help
python -m src.ai.train_enhanced --help
```

### 运行测试

```bash
python -m tests.test_checkpoint
```

或使用旧版入口（根目录下的副本）：
```bash
python test_checkpoint.py
```

### 性能评估

```bash
python scripts/benchmark_checkpoint.py
```

---

## 使用指南 📖

### 人机对战

1. 启动游戏：`python -m src.gui.main_enhanced`（或 `python main_enhanced.py`）
2. 选择游戏模式："人机对战"
3. 选择先手："玩家先手"或"AI先手"
4. 选择难度："简单"、"中等"或"困难"
5. 点击"开始游戏"

### 检查点功能 💾

#### 训练检查点

训练过程中会自动保存检查点，支持从中断处恢复：

```python
from checkpoint import CheckpointManager
from train_enhanced import train_with_checkpoint_and_firstmover

# 创建检查点管理器
cm = CheckpointManager(checkpoint_dir="checkpoints")

# 从检查点恢复训练
nn = train_with_checkpoint_and_firstmover(
    nn,
    checkpoint_manager=cm,
    resume_from_checkpoint="checkpoints/training/training_ep10000_xxx.pkl"
)
```

#### 游戏检查点

游戏状态会自动保存，支持加载继续游戏：
- 点击"加载游戏"按钮恢复上次游戏
- 每次移动后自动保存

### 先手后手配置 🎯

```python
from train_enhanced import FirstMoverConfig

# AI 50% 概率先手
config = FirstMoverConfig(ai_first_ratio=0.5)

# AI 始终先手
config = FirstMoverConfig(fixed_first_player='X')

# AI 始终后手
config = FirstMoverConfig(fixed_first_player='O')

# 交替先手
config = FirstMoverConfig(alternate_turns=True)
```

### 统一配置系统 ⚙️

项目使用统一的配置模块管理路径和参数：

```python
from src.core.config import ModelConfig, TrainingConfig, resolve_model_path

# 获取默认模型路径
model_path = ModelConfig.get_default_model_path()
# 输出: /path/to/project/models/tic_tac_toe_ai_model.pkl

# 获取最佳模型路径
best_path = ModelConfig.get_best_model_path()
# 输出: /path/to/project/models/best_model.pkl

# 解析自定义路径
path = resolve_model_path("my_model.pkl")  # 相对路径
path = resolve_model_path("/absolute/path/model.pkl")  # 绝对路径
```

**模型存储规范：**
- 所有模型默认保存在 `models/` 目录下
- 训练脚本自动创建 models 目录
- GUI 程序优先从 models 目录加载模型

---

## 依赖说明 📦

项目主要依赖：

- `numpy`: 数值计算
- `kivy`: GUI 框架
- `pandas`: 数据处理（可选）

完整依赖列表见 `requirements.txt`。

---

## 系统要求 💻

- Python 3.8+
- 操作系统：Linux / macOS / Windows
- 内存：至少 2GB RAM
- 磁盘空间：至少 100MB

---

## 开发指南 🛠️

### 代码规范

- 遵循 PEP 8 代码风格
- 使用类型注解
- 添加中文文档字符串

### 测试

运行所有测试：
```bash
python test_checkpoint.py
```

### 文档

详细设计文档位于 `docs/` 目录：
- `CHECKPOINT_DESIGN.md`: 检查点系统设计
- `IMPLEMENTATION_SUMMARY.md`: 实现总结

---

## 许可证 📄

MIT License

---

## 作者 ✍️

AI Assistant

---

## 更新日志 📝

### v3.0 (2026-04-13)
- ✅ 新增检查点系统，支持训练中断恢复
- ✅ 新增先手后手功能
- ✅ 新增增强版游戏界面
- ✅ 完善文档和测试
- ✅ 优化项目结构和环境配置

### v2.0 (2026-04-12)
- 优化训练算法
- 修复规则对手逻辑
- 添加中文注释

### v1.0 (2026-04-11)
- 初始版本
- 基础人机对战功能
