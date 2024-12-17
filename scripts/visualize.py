#!/usr/bin/env python
"""可视化启动脚本"""
import os
import sys
import argparse
import logging
from pathlib import Path
import yaml
import streamlit as st

from src.visualization import Dashboard
from src.utils.config import load_config
from src.utils.logger import setup_logger

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="启动可视化界面")
    
    # 基础参数
    parser.add_argument("--config", type=str, help="可视化配置文件路径")
    parser.add_argument("--port", type=int, default=8501, help="服务端口")
    
    # 数据参数
    parser.add_argument("--exp-dir", type=str, help="实验目录")
    parser.add_argument("--log-dir", type=str, help="日志目录")
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    if args.config:
        config = load_config(args.config)
    else:
        config = {}
        
    # 更新配置
    if args.exp_dir:
        config['exp_dir'] = args.exp_dir
    if args.log_dir:
        config['log_dir'] = args.log_dir
        
    # 创建仪表盘
    dashboard = Dashboard(config)
    
    # 启动界面
    dashboard.run(port=args.port)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 