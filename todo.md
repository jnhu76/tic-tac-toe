# 井字棋 AI 项目待办事项

## 待解决问题

暂无高优先级问题。

## 已完成事项

- [x] 项目结构优化 - 创建模块化目录结构
- [x] 游戏逻辑验证测试设计
- [x] 统一配置系统实现 (`src/core/config.py`)
- [x] 命令行参数支持 (`-e`, `-m`, `--ai-first-ratio` 等)
- [x] 模型路径统一配置
- [x] 更新文档 (`readme.md`, `claude.md`)
- [x] **游戏训练逻辑修复** (2026-06-01)
  - 根因：Python版训练逻辑与C代码不一致，具体问题：
    1. 反向传播(backward)未按C代码的 `output - target` + `|reward_scaling|` 实现
    2. 训练方法错误：对softmax输出直接加减reward而非构建0/1目标分布
    3. 缺少负奖励处理（C代码中负奖励分散给其他合法动作）
    4. 缺少时间重要性衰减（`move_importance = 0.5 + 0.5 * move_idx/num_moves`）
    5. 学习率差异（C=0.1 vs Python=0.001）、隐藏层大小差异（C=100 vs Python=128）
    6. `main_enhanced.py`导入了不存在的 `quick_load_game` 函数
  - 修复：重写 `src/ai/train.py` 使训练逻辑与C代码完全一致
  - 结果：150000局训练后 vs 随机对手 胜率83.3%，败率0.8%
  - 相关文件：`src/ai/train.py`, `src/ai/__init__.py`, `src/gui/main.py`, `src/gui/main_enhanced.py`

## 功能增强（未来规划）

- [ ] 训练可视化（TensorBoard 集成）
- [ ] 模型性能对比工具
- [ ] 更复杂的神经网络架构（CNN、ResNet）
- [ ] 支持更大的棋盘（4x4、5x5）
- [ ] 在线对战模式

## 技术债务

- [x] 完善单元测试覆盖率 (2026-06-01) - 59个pytest测试覆盖核心模块
- [x] 添加类型检查（mypy）(2026-06-01) - `train.py`, `common.py`, `checkpoint.py` 全部通过
- [x] 代码风格统一（black、isort）(2026-06-01) - 所有源文件和测试已格式化
- [ ] CI/CD 流程搭建

---

**最后更新**: 2026-06-01
