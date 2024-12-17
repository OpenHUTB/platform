"""环境模块"""
from .carla_env import CarlaEnv
from src.environments.sensors.sensor_manager import SensorManager
from src.environments.tasks.task_registry import TaskRegistry, register_task

__all__ = [
    'CarlaEnv',
    'SensorManager',
    'TaskRegistry',
    'register_task'
] 