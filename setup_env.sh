#!/bin/bash
# -*- coding: utf-8 -*-
"""
环境配置脚本

本脚本用于配置井字棋AI项目的Python环境
包括：
1. 创建虚拟环境
2. 激活虚拟环境
3. 配置PyPI镜像源
4. 安装依赖

使用方法:
    source setup_env.sh

作者：AI Assistant
版本：1.0
日期：2026-04-13
"""

echo "=============================================="
echo "井字棋AI项目 - 环境配置脚本"
echo "=============================================="
echo ""

# 检查uv是否安装
if ! command -v uv &> /dev/null; then
    echo "错误：未找到uv工具"
    echo "请先安装uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
    return 1
fi

echo "✓ uv工具已安装"

# 配置清华大学PyPI镜像源
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
echo "✓ 已配置清华大学PyPI镜像源"

# 检查虚拟环境是否存在
if [ ! -d ".venv" ]; then
    echo ""
    echo "步骤1: 创建虚拟环境..."
    uv venv .venv
    if [ $? -ne 0 ]; then
        echo "错误：创建虚拟环境失败"
        return 1
    fi
    echo "✓ 虚拟环境创建成功"
else
    echo "✓ 虚拟环境已存在"
fi

# 激活虚拟环境
echo ""
echo "步骤2: 激活虚拟环境..."
source ./.venv/bin/activate
if [ $? -ne 0 ]; then
    echo "错误：激活虚拟环境失败"
    return 1
fi
echo "✓ 虚拟环境已激活"

# 检查requirements.txt是否存在
if [ ! -f "requirements.txt" ]; then
    echo ""
    echo "警告：未找到requirements.txt文件"
    echo "将创建空的依赖文件"
    touch requirements.txt
fi

# 安装依赖
echo ""
echo "步骤3: 安装依赖..."
echo "使用镜像源: $UV_INDEX_URL"
uv pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "错误：安装依赖失败"
    return 1
fi
echo "✓ 依赖安装成功"

echo ""
echo "=============================================="
echo "环境配置完成！"
echo "=============================================="
echo ""
echo "当前环境信息:"
echo "  Python版本: $(python --version)"
echo "  虚拟环境: $VIRTUAL_ENV"
echo "  PyPI镜像: $UV_INDEX_URL"
echo ""
echo "可用命令:"
echo "  python main.py              # 运行标准版游戏"
echo "  python main_enhanced.py     # 运行增强版游戏"
echo "  python train.py             # 训练AI"
echo "  python test_checkpoint.py   # 运行测试"
echo ""
