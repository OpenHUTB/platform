from typing import Dict, Any, Tuple
import numpy as np
import gym

from src.environments.base import BaseEnv
from src.utils.registry import register_env

@register_env("custom_navigation")
class CustomNavigationEnv(BaseEnv):
    """自定义导航环境"""
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # 创建观察空间
        self.observation_spaces = {
            'rgb': gym.spaces.Box(
                low=0, high=255,
                shape=(3, 224, 224),
                dtype=np.uint8
            ),
            'lidar': gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(32, 1024, 3),
                dtype=np.float32
            ),
            'state': gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(10,),
                dtype=np.float32
            )
        }
        
        # 创建动作空间
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, 0.0]),  # [steer, throttle, brake]
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # 创建奖励函数
        self.reward_func = create_reward(
            config['reward_type'],
            config['reward_config']
        )
        
    def reset(self) -> Dict[str, Any]:
        """重置环境"""
        # 重置CARLA环境
        super().reset()
        
        # 生成随机目标点
        self.target_location = self._generate_target()
        
        # 获取初始观察
        obs = self._get_obs()
        
        return obs
        
    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """环境步进"""
        # 执行动作
        self._apply_action(action)
        
        # 更新环境
        self.world.tick()
        
        # 获取新的观察
        next_obs = self._get_obs()
        
        # 计算奖励
        reward = self.reward_func(
            self._get_state(),
            action,
            self._get_next_state(),
            self._get_info()
        )
        
        # 检查是否结束
        done = self._is_done()
        
        # 获取额外信息
        info = self._get_info()
        
        return next_obs, reward, done, info
        
    def get_observation_space(self) -> Dict[str, gym.Space]:
        """获取观察空间"""
        return self.observation_spaces
        
    def get_action_space(self) -> gym.Space:
        """获取动作空间"""
        return self.action_space
        
    def get_reward_info(self) -> Dict[str, float]:
        """获取奖励分解信息"""
        return self.reward_func.get_info()
        
    def _get_obs(self) -> Dict[str, Any]:
        """获取观察"""
        return {
            'rgb': self._get_camera_data(),
            'lidar': self._get_lidar_data(),
            'state': self._get_state_vector()
        }
        
    def _get_state_vector(self) -> np.ndarray:
        """获取状态向量"""
        # 获取车辆状态
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        
        # 计算相对目标位置
        target_location = self.target_location
        relative_pos = np.array([
            target_location.x - transform.location.x,
            target_location.y - transform.location.y
        ])
        
        # 组合状态向量
        state = np.concatenate([
            [transform.location.x, transform.location.y],  # 位置
            [velocity.x, velocity.y],  # 速度
            [transform.rotation.yaw],  # 朝向
            relative_pos,  # 相对目标位置
            [np.linalg.norm(relative_pos)]  # 距离目标距离
        ])
        
        return state 