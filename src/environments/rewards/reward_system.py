from typing import Dict, List
import numpy as np
from abc import ABC, abstractmethod

class BaseReward(ABC):
    """奖励函数基类"""
    @abstractmethod
    def __call__(self, state: Dict, action: np.ndarray, next_state: Dict, info: Dict) -> float:
        """计算奖励"""
        pass

class SafetyReward(BaseReward):
    """安全奖励"""
    def __init__(self, config: Dict):
        self.collision_penalty = config.get('collision_penalty', -100)
        self.min_distance_threshold = config.get('min_distance_threshold', 5.0)
        self.distance_reward_factor = config.get('distance_reward_factor', 0.1)
        
    def __call__(self, state: Dict, action: np.ndarray, next_state: Dict, info: Dict) -> float:
        reward = 0.0
        
        # 碰撞惩罚
        if info.get('collision', False):
            reward += self.collision_penalty
            
        # 安全距离奖励
        min_distance = info.get('distance_to_others', float('inf'))
        if min_distance < self.min_distance_threshold:
            reward += (min_distance - self.min_distance_threshold) * self.distance_reward_factor
            
        return reward

class EfficiencyReward(BaseReward):
    """效率奖励"""
    def __init__(self, config: Dict):
        self.target_speed = config.get('target_speed', 30.0)  # km/h
        self.speed_reward_factor = config.get('speed_reward_factor', 0.1)
        self.progress_reward_factor = config.get('progress_reward_factor', 1.0)
        
    def __call__(self, state: Dict, action: np.ndarray, next_state: Dict, info: Dict) -> float:
        reward = 0.0
        
        # 速度奖励
        current_speed = info.get('speed', 0)  # km/h
        speed_reward = -abs(current_speed - self.target_speed) * self.speed_reward_factor
        reward += speed_reward
        
        # 进度奖励
        progress = info.get('progress', 0)  # 路径完成度 [0,1]
        reward += progress * self.progress_reward_factor
        
        return reward

class ComfortReward(BaseReward):
    """舒适度奖励"""
    def __init__(self, config: Dict):
        self.jerk_penalty_factor = config.get('jerk_penalty_factor', 0.1)
        self.acceleration_penalty_factor = config.get('acceleration_penalty_factor', 0.1)
        
    def __call__(self, state: Dict, action: np.ndarray, next_state: Dict, info: Dict) -> float:
        reward = 0.0
        
        # 加速度惩罚
        acceleration = info.get('acceleration', 0)  # m/s^2
        reward -= abs(acceleration) * self.acceleration_penalty_factor
        
        # 抖动惩罚
        jerk = info.get('jerk', 0)  # m/s^3
        reward -= abs(jerk) * self.jerk_penalty_factor
        
        return reward

class RewardSystem:
    """奖励系统"""
    def __init__(self, config: Dict):
        self.config = config
        self.rewards = {
            'safety': SafetyReward(config.get('safety_reward', {})),
            'efficiency': EfficiencyReward(config.get('efficiency_reward', {})),
            'comfort': ComfortReward(config.get('comfort_reward', {}))
        }
        
        # 奖励权重
        self.weights = config.get('reward_weights', {
            'safety': 1.0,
            'efficiency': 0.5,
            'comfort': 0.3
        })
        
    def compute_reward(self, state: Dict, action: np.ndarray, next_state: Dict, info: Dict) -> float:
        """计算总奖励"""
        total_reward = 0.0
        reward_info = {}
        
        for name, reward_func in self.rewards.items():
            reward = reward_func(state, action, next_state, info)
            weighted_reward = reward * self.weights[name]
            total_reward += weighted_reward
            reward_info[f'{name}_reward'] = reward
            
        return total_reward, reward_info 