# -*- coding: utf-8 -*-
"""
检查点系统和先手后手功能性能评估

本脚本用于评估：
1. 检查点保存/加载性能
2. 不同先手策略对训练的影响
3. 系统资源占用情况

运行方式:
    python benchmark_checkpoint.py

作者：AI Assistant
版本：1.0
日期：2026-04-13
"""

import time
import os
import tempfile
import shutil
import numpy as np
from typing import Dict, List, Tuple
import json
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.checkpoint import CheckpointManager, TrainingState, GameState
from src.ai.train_enhanced import FirstMoverConfig, evaluate_with_firstmover
from src.ai.train import NeuralNetwork, random_opponent_move, rule_based_opponent_move


class PerformanceBenchmark:
    """性能评估类"""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_checkpoint_save_load(self, num_iterations: int = 10) -> Dict:
        """
        评估检查点保存和加载性能
        
        参数:
            num_iterations (int): 测试迭代次数
            
        返回:
            Dict: 性能指标
        """
        print("=" * 60)
        print("检查点保存/加载性能评估")
        print("=" * 60)
        
        test_dir = tempfile.mkdtemp()
        cm = CheckpointManager(checkpoint_dir=test_dir, verbose=False)
        nn = NeuralNetwork()
        
        save_times = []
        load_times = []
        file_sizes = []
        memory_usages = []
        
        for i in range(num_iterations):
            # 准备状态
            state = TrainingState(
                episode=i*1000,
                epsilon=0.5,
                best_win_rate_rule=0.6,
                best_win_rate_random=0.7,
                learning_rate=0.001,
                total_episodes=10000,
                phase=0,
                metrics_history=[],
                timestamp="2026-04-13 12:00:00"
            )
            
            # 测量保存时间
            start_time = time.time()
            filepath = cm.save_training_checkpoint(nn, state)
            save_time = time.time() - start_time
            
            save_times.append(save_time)
            file_sizes.append(os.path.getsize(filepath))
            
            # 测量加载时间
            start_time = time.time()
            nn_loaded, state_loaded = cm.load_training_checkpoint(filepath)
            load_time = time.time() - start_time
            
            load_times.append(load_time)
        
        # 计算统计指标
        results = {
            'save_time_mean': np.mean(save_times),
            'save_time_std': np.std(save_times),
            'load_time_mean': np.mean(load_times),
            'load_time_std': np.std(load_times),
            'file_size_mean': np.mean(file_sizes),
            'file_size_std': np.std(file_sizes),
            'save_times': save_times,
            'load_times': load_times,
            'file_sizes': file_sizes
        }
        
        # 打印结果
        print(f"\n保存性能:")
        print(f"  平均时间: {results['save_time_mean']*1000:.2f} ms")
        print(f"  标准差: {results['save_time_std']*1000:.2f} ms")
        print(f"  最小: {min(save_times)*1000:.2f} ms")
        print(f"  最大: {max(save_times)*1000:.2f} ms")
        
        print(f"\n加载性能:")
        print(f"  平均时间: {results['load_time_mean']*1000:.2f} ms")
        print(f"  标准差: {results['load_time_std']*1000:.2f} ms")
        print(f"  最小: {min(load_times)*1000:.2f} ms")
        print(f"  最大: {max(load_times)*1000:.2f} ms")
        
        print(f"\n文件大小:")
        print(f"  平均: {results['file_size_mean']/1024/1024:.2f} MB")
        print(f"  标准差: {results['file_size_std']/1024/1024:.2f} MB")
        
        # 清理
        shutil.rmtree(test_dir, ignore_errors=True)
        
        self.results['checkpoint_save_load'] = results
        return results
    
    def benchmark_game_checkpoint(self, num_iterations: int = 100) -> Dict:
        """
        评估游戏检查点性能
        
        参数:
            num_iterations (int): 测试迭代次数
            
        返回:
            Dict: 性能指标
        """
        print("\n" + "=" * 60)
        print("游戏检查点性能评估")
        print("=" * 60)
        
        test_dir = tempfile.mkdtemp()
        cm = CheckpointManager(checkpoint_dir=test_dir, verbose=False)
        
        save_times = []
        load_times = []
        file_sizes = []
        
        for i in range(num_iterations):
            # 创建游戏状态
            state = GameState(
                board=[['X' if j == 0 else 'O' if j == 1 else None 
                       for j in range(3)] for _ in range(3)],
                current_player='X',
                move_history=[
                    {'player': 0, 'symbol': 'X', 'position': (0, 0)},
                    {'player': 1, 'symbol': 'O', 'position': (0, 1)},
                ],
                game_mode='pve',
                ai_difficulty='medium',
                is_game_over=False,
                winner=None,
                timestamp="2026-04-13 12:00:00"
            )
            
            # 测量保存时间
            start_time = time.time()
            filepath = cm.save_game_checkpoint(state)
            save_time = time.time() - start_time
            
            save_times.append(save_time)
            file_sizes.append(os.path.getsize(filepath))
            
            # 测量加载时间
            start_time = time.time()
            loaded_state = cm.load_game_checkpoint(filepath)
            load_time = time.time() - start_time
            
            load_times.append(load_time)
        
        # 计算统计指标
        results = {
            'save_time_mean': np.mean(save_times),
            'save_time_std': np.std(save_times),
            'load_time_mean': np.mean(load_times),
            'load_time_std': np.std(load_times),
            'file_size_mean': np.mean(file_sizes),
            'file_size_std': np.std(file_sizes)
        }
        
        # 打印结果
        print(f"\n保存性能:")
        print(f"  平均时间: {results['save_time_mean']*1000:.2f} ms")
        print(f"  标准差: {results['save_time_std']*1000:.2f} ms")
        
        print(f"\n加载性能:")
        print(f"  平均时间: {results['load_time_mean']*1000:.2f} ms")
        print(f"  标准差: {results['load_time_std']*1000:.2f} ms")
        
        print(f"\n文件大小:")
        print(f"  平均: {results['file_size_mean']:.2f} bytes")
        
        # 清理
        shutil.rmtree(test_dir, ignore_errors=True)
        
        self.results['game_checkpoint'] = results
        return results
    
    def benchmark_firstmover_strategies(self, num_games: int = 200) -> Dict:
        """
        评估不同先手策略的性能
        
        参数:
            num_games (int): 每种策略的评估游戏数
            
        返回:
            Dict: 性能指标
        """
        print("\n" + "=" * 60)
        print("先手策略性能评估")
        print("=" * 60)
        
        nn = NeuralNetwork()
        strategies = {
            'AI先手': FirstMoverConfig(fixed_first_player='X'),
            'AI后手': FirstMoverConfig(fixed_first_player='O'),
            '随机50%': FirstMoverConfig(ai_first_ratio=0.5),
            '交替先手': FirstMoverConfig(alternate_turns=True)
        }
        
        results = {}
        
        for name, config in strategies.items():
            print(f"\n测试策略: {name}")
            
            # 统计100次分配
            first_players = [config.get_first_player(i) for i in range(100)]
            x_count = first_players.count('X')
            o_count = first_players.count('O')
            
            print(f"  X先手次数: {x_count}")
            print(f"  O先手次数: {o_count}")
            print(f"  X比例: {x_count/100:.2%}")
            
            # 评估AI作为先手和后的性能
            if name == 'AI先手':
                metrics_first = evaluate_with_firstmover(nn, rule_based_opponent_move, num_games, ai_first=True)
                metrics_second = {'win_rate': 0, 'draw_rate': 0, 'loss_rate': 0}
            elif name == 'AI后手':
                metrics_first = {'win_rate': 0, 'draw_rate': 0, 'loss_rate': 0}
                metrics_second = evaluate_with_firstmover(nn, rule_based_opponent_move, num_games, ai_first=False)
            else:
                metrics_first = evaluate_with_firstmover(nn, rule_based_opponent_move, num_games//2, ai_first=True)
                metrics_second = evaluate_with_firstmover(nn, rule_based_opponent_move, num_games//2, ai_first=False)
            
            print(f"  AI先手胜率: {metrics_first['win_rate']:.2%}")
            print(f"  AI后手胜率: {metrics_second['win_rate']:.2%}")
            print(f"  平均胜率: {(metrics_first['win_rate'] + metrics_second['win_rate'])/2:.2%}")
            
            results[name] = {
                'x_ratio': x_count / 100,
                'win_rate_first': metrics_first['win_rate'],
                'win_rate_second': metrics_second['win_rate'],
                'avg_win_rate': (metrics_first['win_rate'] + metrics_second['win_rate']) / 2
            }
        
        self.results['firstmover_strategies'] = results
        return results
    
    def benchmark_training_overhead(self, num_episodes: int = 1000) -> Dict:
        """
        评估检查点对训练性能的影响
        
        参数:
            num_episodes (int): 训练轮次
            
        返回:
            Dict: 性能指标
        """
        print("\n" + "=" * 60)
        print("检查点对训练性能的影响")
        print("=" * 60)
        
        test_dir = tempfile.mkdtemp()
        
        # 无检查点训练
        print("\n测试1: 无检查点训练")
        nn1 = NeuralNetwork()
        start_time = time.time()
        
        for episode in range(num_episodes):
            # 模拟训练步骤
            inputs = np.random.randn(18)
            targets = np.random.randn(9)
            nn1.backward(targets, 0.001)
        
        time_no_checkpoint = time.time() - start_time
        print(f"  耗时: {time_no_checkpoint:.2f} 秒")
        
        # 有检查点训练（每100轮保存）
        print("\n测试2: 每100轮保存检查点")
        nn2 = NeuralNetwork()
        cm = CheckpointManager(checkpoint_dir=test_dir, auto_save_interval=100, verbose=False)
        start_time = time.time()
        
        for episode in range(num_episodes):
            # 模拟训练步骤
            inputs = np.random.randn(18)
            targets = np.random.randn(9)
            nn2.backward(targets, 0.001)
            
            # 检查是否保存
            if cm.should_auto_save(episode):
                state = TrainingState(
                    episode=episode,
                    epsilon=0.5,
                    best_win_rate_rule=0.6,
                    best_win_rate_random=0.7,
                    learning_rate=0.001,
                    total_episodes=num_episodes,
                    phase=0,
                    metrics_history=[],
                    timestamp="2026-04-13 12:00:00"
                )
                cm.save_training_checkpoint(nn2, state)
        
        time_with_checkpoint = time.time() - start_time
        print(f"  耗时: {time_with_checkpoint:.2f} 秒")
        
        # 计算开销
        overhead = (time_with_checkpoint - time_no_checkpoint) / time_no_checkpoint * 100
        print(f"\n检查点开销: {overhead:.2f}%")
        
        results = {
            'time_no_checkpoint': time_no_checkpoint,
            'time_with_checkpoint': time_with_checkpoint,
            'overhead_percent': overhead,
            'episodes': num_episodes,
            'save_interval': 100
        }
        
        # 清理
        shutil.rmtree(test_dir, ignore_errors=True)
        
        self.results['training_overhead'] = results
        return results
    
    def generate_report(self, output_file: str = 'benchmark_report.json'):
        """
        生成性能评估报告
        
        参数:
            output_file (str): 输出文件路径
        """
        report = {
            'summary': {
                'checkpoint_save_time_ms': self.results.get('checkpoint_save_load', {}).get('save_time_mean', 0) * 1000,
                'checkpoint_load_time_ms': self.results.get('checkpoint_save_load', {}).get('load_time_mean', 0) * 1000,
                'checkpoint_file_size_mb': self.results.get('checkpoint_save_load', {}).get('file_size_mean', 0) / 1024 / 1024,
                'game_checkpoint_save_time_ms': self.results.get('game_checkpoint', {}).get('save_time_mean', 0) * 1000,
                'training_overhead_percent': self.results.get('training_overhead', {}).get('overhead_percent', 0)
            },
            'details': self.results
        }
        
        # 保存JSON报告
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print("\n" + "=" * 60)
        print("性能评估报告")
        print("=" * 60)
        print(f"\n报告已保存到: {output_file}")
        print("\n关键指标:")
        print(f"  训练检查点保存时间: {report['summary']['checkpoint_save_time_ms']:.2f} ms")
        print(f"  训练检查点加载时间: {report['summary']['checkpoint_load_time_ms']:.2f} ms")
        print(f"  训练检查点文件大小: {report['summary']['checkpoint_file_size_mb']:.2f} MB")
        print(f"  游戏检查点保存时间: {report['summary']['game_checkpoint_save_time_ms']:.2f} ms")
        print(f"  检查点对训练的开销: {report['summary']['training_overhead_percent']:.2f}%")
        
        return report


def main():
    """主函数"""
    print("=" * 60)
    print("井字棋AI - 检查点系统性能评估")
    print("=" * 60)
    print()
    
    benchmark = PerformanceBenchmark()
    
    # 运行各项评估
    benchmark.benchmark_checkpoint_save_load(num_iterations=10)
    benchmark.benchmark_game_checkpoint(num_iterations=100)
    benchmark.benchmark_firstmover_strategies(num_games=200)
    benchmark.benchmark_training_overhead(num_episodes=1000)
    
    # 生成报告
    report = benchmark.generate_report('benchmark_report.json')
    
    print("\n" + "=" * 60)
    print("性能评估完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
