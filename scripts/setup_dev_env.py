#!/usr/bin/env python
"""开发环境设置脚本"""
import os
import sys
import subprocess
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 7):
        raise RuntimeError("需要Python 3.7或更高版本")
    logger.info(f"Python版本: {sys.version}")

def check_cuda():
    """检查CUDA环境"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_version = torch.version.cuda
            logger.info(f"CUDA可用，版本: {cuda_version}")
        else:
            logger.warning("CUDA不可用")
    except ImportError:
        logger.warning("未安装PyTorch")

def setup_carla():
    """设置CARLA环境"""
    carla_root = os.getenv("CARLA_ROOT")
    if not carla_root:
        logger.error("未设置CARLA_ROOT环境变量")
        return False
        
    carla_path = Path(carla_root)
    if not carla_path.exists():
        logger.error(f"CARLA路径不存在: {carla_path}")
        return False
        
    # 添加CARLA Python API到PYTHONPATH
    python_path = carla_path / "PythonAPI" / "carla" / "dist"
    egg_files = list(python_path.glob("carla-*%d.%d-%s.egg" % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'
    )))
    
    if not egg_files:
        logger.error("未找到CARLA Python API")
        return False
        
    os.environ["PYTHONPATH"] = str(egg_files[0])
    logger.info(f"已设置CARLA Python API: {egg_files[0]}")
    return True

def create_directories():
    """创建必要的目录"""
    dirs = [
        "data",
        "logs",
        "checkpoints",
        "recordings",
        "experiments",
        "configs/custom"
    ]
    
    for dir_name in dirs:
        path = Path(dir_name)
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"创建目录: {path}")

def install_dependencies():
    """安装项目依赖"""
    try:
        subprocess.run(["pip", "install", "-e", "."], check=True)
        logger.info("安装基础依赖完成")
        
        subprocess.run(["pip", "install", "-e", ".[dev]"], check=True)
        logger.info("安装开发依赖完成")
    except subprocess.CalledProcessError as e:
        logger.error(f"安装依赖失败: {e}")
        return False
    return True

def setup_git_hooks():
    """设置Git钩子"""
    try:
        subprocess.run(["pre-commit", "install"], check=True)
        logger.info("安装pre-commit钩子完成")
    except subprocess.CalledProcessError as e:
        logger.error(f"安装pre-commit钩子失败: {e}")
        return False
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="设置开发环境")
    parser.add_argument("--skip-carla", action="store_true", help="跳过CARLA设置")
    parser.add_argument("--skip-deps", action="store_true", help="跳过依赖安装")
    args = parser.parse_args()
    
    logger.info("开始设置开发环境...")
    
    # 检查Python版本
    check_python_version()
    
    # 检查CUDA
    check_cuda()
    
    # 设置CARLA
    if not args.skip_carla:
        if not setup_carla():
            return 1
            
    # 创建目录
    create_directories()
    
    # 安装依赖
    if not args.skip_deps:
        if not install_dependencies():
            return 1
            
    # 设置Git钩子
    if not setup_git_hooks():
        return 1
        
    logger.info("开发环境设置完成!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 