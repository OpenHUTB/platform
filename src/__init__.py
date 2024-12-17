"""
CARLA自动驾驶算法测试平台
"""

__version__ = "0.1.0"
__author__ = "Your Organization"
__license__ = "MIT"

from typing import Dict, Any
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 导出主要接口
from .environments import CarlaEnv
from .algorithms import BaseAlgorithm
from .training import TrainManager
from .evaluation import ModelEvaluator
from .visualization import Dashboard

__all__ = [
    'CarlaEnv',
    'BaseAlgorithm',
    'TrainManager',
    'ModelEvaluator',
    'Dashboard'
] 