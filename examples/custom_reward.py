from typing import Dict, Any
import numpy as np

from src.environments.rewards.base import BaseReward
from src.utils.registry import register_reward


@register_reward("custom")
class CustomReward(BaseReward):
    """自定义奖励函数"""
    def __init__(self, config: Dict):
        self.config = config
        self.info = {}
        
    def __call__(self, state: Dict[str, Any], action: np.ndarray,
                 next_state: Dict[str, Any], info: Dict[str, Any]) -> float:
        """计算奖励"""
        # 计算距离奖励
        distance_reward = self._compute_distance_reward(state, next_state)
        
        # 计算速度奖励
        speed_reward = self._compute_speed_reward(next_state)
        
        # 计算舒适度奖励
        comfort_reward = self._compute_comfort_reward(action)
        
        # 记录分解奖励
        self.info = {
            'distance_reward': distance_reward,
            'speed_reward': speed_reward,
            'comfort_reward': comfort_reward
        }
        
        # 返回总奖励
        return distance_reward + speed_reward + comfort_reward
        
    def get_info(self) -> Dict[str, float]:
        """获取奖励分解信息"""
        return self.info.copy()
        
    def _compute_distance_reward(self, state: Dict[str, Any], 
                               next_state: Dict[str, Any]) -> float:
        """计算距离奖励"""
        current_dist = np.linalg.norm(state['target_info']['relative_pos'])
        next_dist = np.linalg.norm(next_state['target_info']['relative_pos'])
        
        progress = current_dist - next_dist
        return np.exp(-next_dist / 10.0) + progress
        
    def _compute_speed_reward(self, state: Dict[str, Any]) -> float:
        """计算速度奖励"""
        speed = np.linalg.norm(state['velocity'])
        target_speed = self.config.get('target_speed', 30.0)
        
        speed_diff = abs(speed - target_speed)
        return -speed_diff / target_speed
        
    def _compute_comfort_reward(self, action: np.ndarray) -> float:
        """计算舒适度奖励"""
        # 惩罚剧烈动作
        action_penalty = -np.square(action).mean()
        return action_penalty 