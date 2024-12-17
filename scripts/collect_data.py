#!/usr/bin/env python
"""数据收集脚本"""
import os
import sys
import argparse
import logging
from pathlib import Path
import yaml
import time

from src.environments import CarlaEnv
from src.data import DataCollector
from src.utils.config import load_config
from src.utils.logger import setup_logger

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="收集训练数据")
    
    # 基础参数
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--output-dir", type=str, required=True, help="输出目录")
    
    # 收集参数
    parser.add_argument("--num-episodes", type=int, default=100, help="收集回合数")
    parser.add_argument("--scenario", type=str, help="场景类型")
    parser.add_argument("--save-video", action="store_true", help="保存视频")
    
    # 其他参数
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--log-level", type=str, default="INFO", help="日志级别")
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 更新配置
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.scenario:
        config["environment"]["scenario"] = args.scenario
        
    # 设置日志
    setup_logger(
        log_file=output_dir / "collect.log",
        level=args.log_level
    )
    
    # 创建环境
    env = CarlaEnv(config["environment"])
    
    # 创建数据收集器
    collector = DataCollector({
        "save_dir": str(output_dir),
        "save_video": args.save_video
    })
    
    # 收集数据
    for episode in range(args.num_episodes):
        obs = env.reset()
        done = False
        episode_steps = 0
        
        while not done:
            # 使用预定义的专家策略
            action = env.get_expert_action()
            
            # 执行动作
            next_obs, reward, done, info = env.step(action)
            
            # 收集数据
            collector.collect(obs, action, reward, next_obs, done, info)
            
            obs = next_obs
            episode_steps += 1
            
        # 结束回合
        collector.end_episode()
        
        logging.info(f"完成回合 {episode+1}/{args.num_episodes}, 步数: {episode_steps}")
        
    return 0

if __name__ == "__main__":
    sys.exit(main()) 