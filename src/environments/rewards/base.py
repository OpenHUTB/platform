from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np

class BaseReward(ABC):
    """奖励函数基类"""
    
    @abstractmethod
    def __call__(self, state: Dict[str, Any], action: np.ndarray, 
                 next_state: Dict[str, Any], info: Dict[str, Any]) -> float:
        """计算奖励"""
        raise NotImplementedError
        
    @abstractmethod
    def get_info(self) -> Dict[str, float]:
        """获取奖励分解信息"""
        raise NotImplementedError

class RewardWrapper:
    """奖励函数包装器，用于组合多个奖励函数"""
    
    def __init__(self, rewards: Dict[str, BaseReward], weights: Dict[str, float]):
        self.rewards = rewards
        self.weights = weights
        self.reward_info = {}
        
    def __call__(self, state: Dict[str, Any], action: np.ndarray,
                 next_state: Dict[str, Any], info: Dict[str, Any]) -> float:
        """计算总奖励"""
        total_reward = 0.0
        self.reward_info.clear()
        
        for name, reward_func in self.rewards.items():
            reward = reward_func(state, action, next_state, info)
            weighted_reward = reward * self.weights[name]
            total_reward += weighted_reward
            
            # 记录分解奖励
            self.reward_info[name] = reward
            self.reward_info[f'{name}_weighted'] = weighted_reward
            
        return total_reward
        
    def get_info(self) -> Dict[str, float]:
        """获取奖励分解信息"""
        return self.reward_info.copy() 