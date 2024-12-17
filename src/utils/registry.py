from typing import Dict, Type, Any, Callable
from functools import wraps

class Registry:
    """组件注册器"""
    def __init__(self):
        self._registry = {}
        
    def register(self, name: str, obj: Any) -> None:
        """注册组件"""
        if name in self._registry:
            raise KeyError(f"Name {name} already registered!")
        self._registry[name] = obj
        
    def get(self, name: str) -> Any:
        """获取组件"""
        if name not in self._registry:
            raise KeyError(f"Name {name} not registered!")
        return self._registry[name]
        
    def list(self) -> list:
        """列出所有已注册组件"""
        return list(self._registry.keys())

# 创建全局注册器
ALGORITHM_REGISTRY = Registry()
NETWORK_REGISTRY = Registry()
REWARD_REGISTRY = Registry()
ENV_REGISTRY = Registry()

def register_algorithm(name: str):
    """算法注册装饰器"""
    def decorator(cls):
        ALGORITHM_REGISTRY.register(name, cls)
        return cls
    return decorator

def register_network(name: str):
    """网络注册装饰��"""
    def decorator(cls):
        NETWORK_REGISTRY.register(name, cls)
        return cls
    return decorator

def register_reward(name: str):
    """奖励函数注册装饰器"""
    def decorator(cls):
        REWARD_REGISTRY.register(name, cls)
        return cls
    return decorator

def register_env(name: str):
    """环境注册装饰器"""
    def decorator(cls):
        ENV_REGISTRY.register(name, cls)
        return cls
    return decorator

def create_algorithm(name: str, config: Dict) -> Any:
    """创建算法实例"""
    algorithm_cls = ALGORITHM_REGISTRY.get(name)
    return algorithm_cls(config)

def create_network(name: str, config: Dict) -> Any:
    """创建网络实例"""
    network_cls = NETWORK_REGISTRY.get(name)
    return network_cls(config)

def create_reward(name: str, config: Dict) -> Any:
    """创建奖励函数实例"""
    reward_cls = REWARD_REGISTRY.get(name)
    return reward_cls(config)

def create_env(name: str, config: Dict) -> Any:
    """创建环境实例"""
    env_cls = ENV_REGISTRY.get(name)
    return env_cls(config) 