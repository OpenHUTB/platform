"""算法注册系统"""
from typing import Dict, Type
from src.algorithms.base.algorithm import BaseAlgorithm

class AlgorithmRegistry:
    """算法注册器"""
    def __init__(self):
        self._algorithms = {}
        
    def register(self, name: str, algorithm_class: Type[BaseAlgorithm]):
        """注册算法"""
        if name in self._algorithms:
            raise ValueError(f"Algorithm {name} already registered")
        self._algorithms[name] = algorithm_class
        
    def get(self, name: str) -> Type[BaseAlgorithm]:
        """获取算法类"""
        if name not in self._algorithms:
            raise ValueError(f"Algorithm {name} not found")
        return self._algorithms[name]
        
    def list_algorithms(self) -> list:
        """列出所有已注册算法"""
        return list(self._algorithms.keys())
        
# 全局算法注册器
ALGORITHM_REGISTRY = AlgorithmRegistry()

# 注册装饰器
def register_algorithm(name: str):
    """算法注册装饰器"""
    def decorator(cls):
        ALGORITHM_REGISTRY.register(name, cls)
        return cls
    return decorator 