@register_task("custom_navigation")
class CustomNavigationTask(CarlaEnv):
    """自定义导航任务"""
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # 任务特定配置
        self.target_speed = config.get('target_speed', 30.0)  # km/h
        self.min_distance = config.get('min_distance', 5.0)  # meters
        self.time_limit = config.get('time_limit', 1000)  # steps
        
        # 路径规划
        self.route_planner = RoutePlanner(self.world.get_map())
        
        # 任务状态
        self.target_location = None
        self.route = None
        self.steps = 0
        
    def reset(self) -> Dict:
        """重置环境"""
        obs = super().reset()
        
        # 设置目标点
        self.target_location = self._get_random_target()
        
        # 规划路径
        self.route = self.route_planner.plan_route(
            self.vehicle.get_location(),
            self.target_location
        )
        
        # 重置计数器
        self.steps = 0
        
        return obs
        
    def _get_reward(self) -> float:
        """计算奖励"""
        reward = 0.0
        
        # 距离奖励
        distance = self._get_target_distance()
        reward += self.reward_weights['distance'] * np.exp(-distance / 10.0)
        
        # 速度奖励
        velocity = self.vehicle.get_velocity()
        speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2)  # km/h
        speed_diff = abs(speed - self.target_speed)
        if speed_diff < 5.0:
            reward += self.reward_weights['speed']
        else:
            reward -= self.reward_weights['speed'] * (speed_diff / self.target_speed)
            
        # 碰撞惩罚
        if self._check_collision():
            reward += self.reward_weights['collision']
            
        # 车道偏离惩罚
        if self._check_lane_invasion():
            reward += self.reward_weights['lane']
            
        return reward
        
    def _is_done(self) -> bool:
        """检查是否结束"""
        # 到达目标
        if self._get_target_distance() < self.min_distance:
            return True
            
        # 碰撞
        if self._check_collision():
            return True
            
        # 超时
        if self.steps >= self.time_limit:
            return True
            
        return False 