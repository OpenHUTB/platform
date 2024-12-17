import gym
import numpy as np
from typing import Dict, Tuple, Any
from gym import spaces

from src.environments.carla_env import CarlaEnv
from src.utils.observation_processor import ObservationProcessor
from src.utils.reward_calculator import RewardCalculator

class RLEnv(gym.Env):
    """强化学习环境包装器"""
    def __init__(self, config: Dict):
        super().__init__()
        
        # 创建CARLA环境
        self.env = CarlaEnv(config)
        
        # 创建观测和奖励处理器
        self.obs_processor = ObservationProcessor(config.get('observation', {}))
        self.reward_calculator = RewardCalculator(config.get('reward', {}))
        
        # 定义动作空间
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0]),  # [steer, throttle, brake]
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # 定义观测空间
        self.observation_space = spaces.Dict({
            'camera': spaces.Box(0, 255, (3, 84, 84), dtype=np.uint8),
            'lidar': spaces.Box(-np.inf, np.inf, (32, 1000, 4), dtype=np.float32),
            'state': spaces.Box(-np.inf, np.inf, (10,), dtype=np.float32)
        })
        
        # 环境参数
        self.frame_skip = config.get('frame_skip', 4)
        self.time_limit = config.get('time_limit', 1000)
        self.current_step = 0
        
    def reset(self) -> Dict[str, np.ndarray]:
        """重置环境"""
        raw_obs = self.env.reset()
        self.current_step = 0
        return self.obs_processor.process(raw_obs)
        
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """环境步进"""
        total_reward = 0
        done = False
        info = {}
        
        # 执行frame_skip次动作
        for _ in range(self.frame_skip):
            raw_obs, env_reward, env_done, env_info = self.env.step(action)
            
            # 处理观测和奖励
            obs = self.obs_processor.process(raw_obs)
            reward = self.reward_calculator.calculate(raw_obs, action, env_info)
            
            total_reward += reward
            info.update(env_info)
            
            self.current_step += 1
            done = env_done or self.current_step >= self.time_limit
            
            if done:
                break
                
        return obs, total_reward, done, info
        
    def render(self, mode='human'):
        """渲染环境"""
        return self.env.render(mode)
        
    def close(self):
        """关闭环境"""
        self.env.close() 