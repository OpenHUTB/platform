from typing import Dict, Tuple
import numpy as np

class NavigationRewardGenerator:
    """导航任务奖励生成器"""
    def __init__(self, config: Dict):
        self.config = config
        
        # 加载奖励权重
        self.weights = config.get('reward_weights', {
            'distance': 1.0,
            'speed': 0.5,
            'collision': 1.0,
            'lane': 0.3,
            'success': 1.0,
            'comfort': 0.2
        })
        
    def generate(self, obs: Dict, action: np.ndarray, info: Dict) -> Tuple[float, Dict]:
        """生成奖励"""
        rewards = {}
        
        # 1. 距离奖励
        rewards['distance'] = self._distance_reward(info['target_distance'])
        
        # 2. 速度奖励
        rewards['speed'] = self._speed_reward(info['speed'])
        
        # 3. 碰撞惩罚
        rewards['collision'] = self._collision_penalty(info['collision'])
        
        # 4. 车道偏离惩罚
        rewards['lane'] = self._lane_penalty(info['lane_invasion'])
        
        # 5. 成功奖励
        rewards['success'] = self._success_reward(info['success'])
        
        # 6. 舒适度奖励
        rewards['comfort'] = self._comfort_reward(action)
        
        # 计算加权总奖励
        total_reward = sum(
            self.weights[k] * v for k, v in rewards.items()
        )
        
        return total_reward, rewards
        
    def _distance_reward(self, distance: float) -> float:
        """距离奖励"""
        return np.exp(-distance / 10.0)
        
    def _speed_reward(self, speed: float) -> float:
        """速度奖励"""
        target_speed = self.config['target_speed']
        speed_diff = abs(speed - target_speed)
        
        if speed_diff < 5.0:
            return 1.0
        else:
            return -speed_diff / target_speed
            
    def _collision_penalty(self, collision: bool) -> float:
        """碰撞惩罚"""
        return -100.0 if collision else 0.0
        
    def _lane_penalty(self, lane_invasion: bool) -> float:
        """车道偏离惩罚"""
        return -10.0 if lane_invasion else 0.0
        
    def _success_reward(self, success: bool) -> float:
        """成功奖励"""
        return 100.0 if success else 0.0
        
    def _comfort_reward(self, action: np.ndarray) -> float:
        """舒适度奖励"""
        # 惩罚剧烈转向
        steer_penalty = -abs(action[0])
        
        # 惩罚剧烈加速和刹车
        acc_penalty = -abs(action[1] - action[2])
        
        return steer_penalty + acc_penalty 