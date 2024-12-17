from typing import Dict, Tuple
import numpy as np
import gym
from gym import spaces

from src.environments.carla_env import CarlaEnv
from src.utils.observation_processor import ObservationProcessor

class NavigationEnv(CarlaEnv):
    """导航任务环境"""
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # 创建观察空间
        self.observation_space = self._create_observation_space()
        
        # 创建动作空间
        self.action_space = self._create_action_space()
        
        # 创建观测处理器
        self.obs_processor = ObservationProcessor(config.get('observation', {}))
        
        # 任务相关参数
        self.target_location = None
        self.start_location = None
        self.min_distance = config.get('min_distance', 5.0)
        self.time_limit = config.get('time_limit', 1000)
        self.steps = 0
        
    def reset(self) -> Dict:
        """重置环境"""
        # 调用父类的reset
        obs = super().reset()
        
        # 重置任务状态
        self.steps = 0
        self.start_location = self.vehicle.get_location()
        self.target_location = self._get_random_target()
        
        # 处理观测
        processed_obs = self.obs_processor.process(obs)
        
        return processed_obs
        
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """环境步进"""
        # 调用父类的step
        obs, _, done, info = super().step(action)
        
        # 处理观测
        processed_obs = self.obs_processor.process(obs)
        
        # 计算奖励
        reward = self._compute_reward(processed_obs, action, info)
        
        # 更新状态
        self.steps += 1
        if self.steps >= self.time_limit:
            done = True
            
        # 更新信息
        info.update({
            'target_distance': self._get_target_distance(),
            'success': self._check_success(),
            'timeout': self.steps >= self.time_limit
        })
        
        return processed_obs, reward, done, info
        
    def _create_observation_space(self) -> gym.Space:
        """创建观察空间"""
        camera_shape = self.config['sensors']['camera_rgb']['shape']
        lidar_shape = self.config['sensors']['lidar']['shape']
        
        return spaces.Dict({
            # 相机观测
            'camera': spaces.Box(
                low=0, high=255,
                shape=camera_shape,
                dtype=np.uint8
            ),
            
            # 激光雷达观测
            'lidar': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=lidar_shape,
                dtype=np.float32
            ),
            
            # 车辆状态
            'vehicle_state': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(10,),  # [x, y, z, roll, pitch, yaw, vx, vy, vz, speed]
                dtype=np.float32
            ),
            
            # 导航信息
            'navigation': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(3,),  # [distance, angle, progress]
                dtype=np.float32
            )
        })
        
    def _create_action_space(self) -> gym.Space:
        """创建动作空间"""
        # 连续动作空间: [steer, throttle, brake]
        return spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
    def _compute_reward(self, obs: Dict, action: np.ndarray, info: Dict) -> float:
        """计算奖励"""
        reward = 0.0
        
        # 1. 目标导向奖励
        target_distance = self._get_target_distance()
        reward += self._get_distance_reward(target_distance)
        
        # 2. 速度奖励
        current_speed = info['speed']
        reward += self._get_speed_reward(current_speed)
        
        # 3. 碰撞惩罚
        if info['collision']:
            reward += self.config['rewards']['collision_penalty']
            
        # 4. 车道偏离惩罚
        if info['lane_invasion']:
            reward += self.config['rewards']['lane_invasion_penalty']
            
        # 5. 完成奖励
        if self._check_success():
            reward += self.config['rewards']['success_reward']
            
        # 6. 舒适度奖励
        reward += self._get_comfort_reward(action, info)
        
        return reward
        
    def _get_distance_reward(self, distance: float) -> float:
        """计算距离奖励"""
        # 使用负指数奖励，距离越近奖励越大
        return self.config['rewards']['distance_factor'] * np.exp(-distance / 10.0)
        
    def _get_speed_reward(self, speed: float) -> float:
        """计算速度奖励"""
        target_speed = self.config['rewards']['target_speed']
        speed_diff = abs(speed - target_speed)
        
        # 速度接近目标速度时给予奖励
        if speed_diff < 5.0:
            return self.config['rewards']['speed_factor']
        else:
            return -self.config['rewards']['speed_factor'] * (speed_diff / target_speed)
            
    def _get_comfort_reward(self, action: np.ndarray, info: Dict) -> float:
        """计算舒适度奖励"""
        reward = 0.0
        
        # 惩罚剧烈转向
        reward += -self.config['rewards']['steer_factor'] * abs(action[0])
        
        # 惩罚剧烈加速和刹车
        acceleration = abs(action[1] - action[2])
        reward += -self.config['rewards']['acceleration_factor'] * acceleration
        
        return reward
        
    def _get_target_distance(self) -> float:
        """计算到目标的距离"""
        if self.target_location is None or self.vehicle is None:
            return float('inf')
            
        current_location = self.vehicle.get_location()
        return current_location.distance(self.target_location)
        
    def _check_success(self) -> bool:
        """检查是否完成任务"""
        return self._get_target_distance() < self.min_distance
        
    def _get_random_target(self) -> carla.Location:
        """获取随机目标点"""
        spawn_points = self.world.get_map().get_spawn_points()
        target_transform = np.random.choice(spawn_points)
        return target_transform.location 