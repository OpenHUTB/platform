from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import gym
import numpy as np

class BaseEnv(gym.Env, ABC):
    """环境基类，定义了环境需要实现的基本接口"""
    
    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """重置环境"""
        raise NotImplementedError
        
    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """环境步进"""
        raise NotImplementedError
        
    @abstractmethod
    def get_observation_space(self) -> Dict[str, gym.Space]:
        """获取观察空间"""
        raise NotImplementedError
        
    @abstractmethod
    def get_action_space(self) -> gym.Space:
        """获取动作空间"""
        raise NotImplementedError
        
    @abstractmethod
    def get_reward_info(self) -> Dict[str, float]:
        """获取奖励分解信息"""
        raise NotImplementedError
        
    @abstractmethod
    def render(self, mode: str = 'human') -> Any:
        """渲染环境"""
        raise NotImplementedError 