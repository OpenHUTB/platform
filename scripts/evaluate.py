#!/usr/bin/env python
"""评估启动脚本"""
import os
import sys
import argparse
import logging
from pathlib import Path
import yaml

from src.evaluation import ModelEvaluator
from src.utils.config import load_config
from src.utils.logger import setup_logger

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="评估自动驾驶智能体")
    
    # 基础参数
    parser.add_argument("--model-path", type=str, required=True, help="模型路径")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--output-dir", type=str, required=True, help="输出目录")
    
    # 评估参数
    parser.add_argument("--num-episodes", type=int, help="评估回合数")
    parser.add_argument("--scenario", type=str, help="评估场景")
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
    
    if args.num_episodes:
        config["evaluation"]["num_episodes"] = args.num_episodes
    if args.scenario:
        config["evaluation"]["scenario"] = args.scenario
    if args.save_video:
        config["evaluation"]["save_video"] = True
        
    # 设置日志
    setup_logger(
        log_file=output_dir / "eval.log",
        level=args.log_level
    )
    
    # 保存配置
    with open(output_dir / "eval_config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
        
    # 创建评估器
    evaluator = ModelEvaluator(config)
    
    # 加载模型
    evaluator.load_model(args.model_path)
    
    # 运行评估
    results = evaluator.evaluate()
    
    # 保存结果
    evaluator.save_results(output_dir / "results.json")
    
    # 生成报告
    evaluator.generate_report(output_dir / "report.html")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 