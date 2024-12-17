#!/usr/bin/env python
"""训练启动脚本"""
import os
import sys
import argparse
import logging
from pathlib import Path
import yaml

from src.training import TrainManager
from src.utils.config import load_config, merge_configs
from src.utils.logger import setup_logger

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练自动驾驶智能体")
    
    # 基础参数
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--exp-name", type=str, required=True, help="实验名称")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    # 训练参数
    parser.add_argument("--resume", type=str, help="恢复训练的检查点路径")
    parser.add_argument("--eval-only", action="store_true", help="仅进行评估")
    
    # 环境参数
    parser.add_argument("--num-workers", type=int, help="并行环境数量")
    parser.add_argument("--scenario", type=str, help="场景类型")
    
    # 日志参数
    parser.add_argument("--log-level", type=str, default="INFO", help="日志级别")
    parser.add_argument("--no-wandb", action="store_true", help="禁用Wandb")
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 更新配置
    exp_dir = Path("experiments") / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    config.update({
        "exp_name": args.exp_name,
        "exp_dir": str(exp_dir),
        "seed": args.seed
    })
    
    if args.num_workers:
        config["environment"]["num_workers"] = args.num_workers
    if args.scenario:
        config["environment"]["scenario"] = args.scenario
        
    # 设置日志
    setup_logger(
        log_file=exp_dir / "train.log",
        level=args.log_level
    )
    
    # 保存配置
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
        
    # 创建训练管理器
    trainer = TrainManager(config)
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
        
    if args.eval_only:
        trainer.evaluate()
    else:
        trainer.train()
        
    return 0

if __name__ == "__main__":
    sys.exit(main()) 