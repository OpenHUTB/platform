#!/usr/bin/env python
"""性能分析脚本"""
import os
import sys
import argparse
import logging
from pathlib import Path
import time
import torch
import psutil
import GPUtil
from typing import Dict

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="性能分析")
    
    # 基础参数
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--model-path", type=str, required=True, help="模型路径")
    parser.add_argument("--output-dir", type=str, required=True, help="输出目录")
    
    # 分析参数
    parser.add_argument("--num-episodes", type=int, default=10, help="分析回合数")
    parser.add_argument("--profile-cuda", action="store_true", help="分析CUDA性能")
    parser.add_argument("--profile-memory", action="store_true", help="分析内存使用")
    
    return parser.parse_args()

class Profiler:
    """性能分析器"""
    def __init__(self, config: Dict):
        self.config = config
        self.stats = {
            'fps': [],
            'latency': [],
            'cpu_usage': [],
            'memory_usage': [],
            'gpu_usage': [],
            'gpu_memory': []
        }
        
    def start_episode(self):
        """开始回合"""
        self.episode_start = time.time()
        self.step_times = []
        
    def step(self):
        """记录步骤"""
        # 记录时间
        step_time = time.time()
        if self.step_times:
            self.stats['latency'].append(step_time - self.step_times[-1])
        self.step_times.append(step_time)
        
        # 记录系统资源
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        self.stats['cpu_usage'].append(cpu_percent)
        self.stats['memory_usage'].append(memory_percent)
        
        # 记录GPU资源
        if torch.cuda.is_available():
            gpu = GPUtil.getGPUs()[0]
            self.stats['gpu_usage'].append(gpu.load * 100)
            self.stats['gpu_memory'].append(gpu.memoryUtil * 100)
            
    def end_episode(self):
        """结束回合"""
        episode_time = time.time() - self.episode_start
        fps = len(self.step_times) / episode_time
        self.stats['fps'].append(fps)
        
    def generate_report(self, output_path: Path):
        """生成报告"""
        import pandas as pd
        import matplotlib.pyplot as plt
        
        # 创建统计表格
        stats_df = pd.DataFrame({
            'Metric': ['FPS', 'Latency (ms)', 'CPU Usage (%)', 'Memory Usage (%)', 'GPU Usage (%)', 'GPU Memory (%)'],
            'Mean': [
                np.mean(self.stats['fps']),
                np.mean(self.stats['latency']) * 1000,
                np.mean(self.stats['cpu_usage']),
                np.mean(self.stats['memory_usage']),
                np.mean(self.stats['gpu_usage']),
                np.mean(self.stats['gpu_memory'])
            ],
            'Std': [
                np.std(self.stats['fps']),
                np.std(self.stats['latency']) * 1000,
                np.std(self.stats['cpu_usage']),
                np.std(self.stats['memory_usage']),
                np.std(self.stats['gpu_usage']),
                np.std(self.stats['gpu_memory'])
            ]
        })
        
        # 保存统计数据
        stats_df.to_csv(output_path / 'stats.csv', index=False)
        
        # 绘制图表
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        metrics = list(self.stats.keys())
        for i, (metric, ax) in enumerate(zip(metrics, axes.flat)):
            ax.plot(self.stats[metric])
            ax.set_title(metric)
            ax.grid(True)
        plt.tight_layout()
        plt.savefig(output_path / 'metrics.png')
        
def main():
    """主函数"""
    args = parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型和环境
    model = torch.load(args.model_path)
    env = CarlaEnv(load_config(args.config)["environment"])
    
    # 创建分析器
    profiler = Profiler({
        'profile_cuda': args.profile_cuda,
        'profile_memory': args.profile_memory
    })
    
    # 运行分析
    for episode in range(args.num_episodes):
        profiler.start_episode()
        obs = env.reset()
        done = False
        
        while not done:
            # 预测动作
            with torch.no_grad():
                action = model(obs)
                
            # 执行动作
            obs, reward, done, info = env.step(action)
            
            # 记录性能
            profiler.step()
            
        profiler.end_episode()
        logging.info(f"完成回合 {episode+1}/{args.num_episodes}")
        
    # 生成报告
    profiler.generate_report(output_dir)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 