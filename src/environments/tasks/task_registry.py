"""任务注册系统"""
from typing import Dict, Type
from src.environments.carla_env import CarlaEnv


class TaskRegistry:
    """任务注册器"""
    def __init__(self):
        self._tasks = {}
        
    def register(self, name: str, task_class: Type[CarlaEnv]):
        """注册任务"""
        if name in self._tasks:
            raise ValueError(f"Task {name} already registered")
        self._tasks[name] = task_class
        
    def get(self, name: str) -> Type[CarlaEnv]:
        """获取任务类"""
        if name not in self._tasks:
            raise ValueError(f"Task {name} not found")
        return self._tasks[name]
        
    def list_tasks(self) -> list:
        """列出所有任务"""
        return list(self._tasks.keys())

# 全局任务注册器
TASK_REGISTRY = TaskRegistry()

# 注册装饰器
def register_task(name: str):
    """任务注册装饰器"""
    def decorator(cls):
        TASK_REGISTRY.register(name, cls)
        return cls
    return decorator 