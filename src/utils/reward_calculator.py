import numpy as np
from typing import Dict
import yaml


class RewardCalculator:
    """奖励计算器"""
    def __init__(self, config_path: str):
        # 加载奖励配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # 初始化奖励组件
        self.reward_components = {
            'safety': self._safety_reward,
            'efficiency': self._efficiency_reward,
            'comfort': self._comfort_reward,
            'rule': self._rule_following_reward
        }
        
    def calculate(self, obs: Dict, action: np.ndarray, info: Dict) -> float:
        """计算总奖励"""
        total_reward = 0
        reward_info = {}
        
        # 计算各个组件的奖励
        for name, reward_func in self.reward_components.items():
            if self.config[f'{name}_rewards']['enabled']:
                reward = reward_func(obs, action, info)
                total_reward += reward
                reward_info[f'{name}_reward'] = reward
                
        return total_reward, reward_info
        
    def _safety_reward(self, obs: Dict, action: np.ndarray, info: Dict) -> float:
        """安全奖励"""
        reward = 0
        config = self.config['safety_rewards']
        
        # 碰撞惩罚
        if info.get('collision', False):
            reward += config['collision']['weight']
            
        # 安全距离奖励
        min_distance = info.get('distance_to_others', float('inf'))
        if min_distance < config['safe_distance']['threshold']:
            reward += config['safe_distance']['weight'] * (
                min_distance - config['safe_distance']['threshold']
            )
            
        # 车道偏离惩罚
        lane_deviation = info.get('lane_deviation', 0)
        if abs(lane_deviation) > config['lane_deviation']['threshold']:
            reward += config['lane_deviation']['weight'] * abs(lane_deviation)
            
        return reward
        
    def _efficiency_reward(self, obs: Dict, action: np.ndarray, info: Dict) -> float:
        """效率奖励"""
        reward = 0
        config = self.config['efficiency_rewards']
        
        # 速度奖励
        current_speed = info.get('speed', 0)
        target_speed = config['speed']['target_speed']
        speed_diff = abs(current_speed - target_speed)
        if speed_diff > config['speed']['tolerance']:
            reward += config['speed']['weight'] * -speed_diff
            
        # 进度奖励
        progress = info.get('progress', 0)
        reward += config['progress']['weight'] * progress
        
        # 完成奖励
        if info.get('success', False):
            reward += config['completion']['weight']
            
        return reward
        
    def _comfort_reward(self, obs: Dict, action: np.ndarray, info: Dict) -> float:
        """舒适度奖励"""
        reward = 0
        config = self.config['comfort_rewards']
        
        # 加速度惩罚
        acceleration = info.get('acceleration', 0)
        if abs(acceleration) > config['acceleration']['threshold']:
            reward += config['acceleration']['weight'] * (
                abs(acceleration) - config['acceleration']['threshold']
            )
            
        # 抖动惩罚
        jerk = info.get('jerk', 0)
        if abs(jerk) > config['jerk']['threshold']:
            reward += config['jerk']['weight'] * (
                abs(jerk) - config['jerk']['threshold']
            )
            
        # 转向惩罚
        steering = abs(action[0])  # 转向动作
        if steering > config['steering']['threshold']:
            reward += config['steering']['weight'] * (
                steering - config['steering']['threshold']
            )
            
        return reward
        
    def _rule_following_reward(self, obs: Dict, action: np.ndarray, info: Dict) -> float:
        """规则遵守奖励"""
        reward = 0
        config = self.config['rule_rewards']
        
        # 红绿灯
        if info.get('traffic_light_violation', False):
            reward += config['traffic_light']['weight']
            
        # 限速
        speed_limit = info.get('speed_limit', float('inf'))
        current_speed = info.get('speed', 0)
        if current_speed > speed_limit + config['speed_limit']['tolerance']:
            reward += config['speed_limit']['weight'] * (
                current_speed - speed_limit
            )
            
        # 停车标志
        if info.get('stop_sign_violation', False):
            reward += config['stop_sign']['weight']
            
        return reward 