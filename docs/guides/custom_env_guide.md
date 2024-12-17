# 自定义环境开发指南

本指南介绍如何在CARLA自动驾驶算法测试平台中自定义观察空间、动作空间和奖励函数。

## 目录
- [观察空间设计](#观察空间设计)
- [动作空间设计](#动作空间设计)
- [奖励函数设计](#奖励函数设计)
- [完整示例](#完整示例)

## 观察空间设计

### 1. 定义观察空间结构
使用gym.spaces定义观察空间的结构：

```python
def _create_observation_space(self) -> gym.Space:
    return spaces.Dict({
        # 相机观测
        'camera': spaces.Box(
            low=0, high=255,
            shape=(3, 84, 84),  # (C, H, W)
            dtype=np.uint8
        ),
        
        # 激光雷达观测
        'lidar': spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(32, 1000, 4),  # (channels, points, features)
            dtype=np.float32
        ),
        
        # 车辆状态
        'vehicle_state': spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(10,),  # [x,y,z, roll,pitch,yaw, vx,vy,vz, speed]
            dtype=np.float32
        )
    })
```

### 2. 实现观测处理器
创建专门的观测处理器来处理原始数据：

```python
class CustomObservationProcessor(ObservationProcessor):
    def __init__(self, config: Dict):
        self.image_size = config.get('image_size', (84, 84))
        self.use_grayscale = config.get('use_grayscale', True)
        
    def process(self, obs: Dict) -> Dict:
        processed_obs = {}
        
        # 处理相机图像
        if 'camera_rgb' in obs:
            processed_obs['camera'] = self._process_image(obs['camera_rgb'])
            
        # 处理其他传感器数据...
        
        return processed_obs
        
    def _process_image(self, image: np.ndarray) -> np.ndarray:
        # 调整大小
        image = cv2.resize(image, self.image_size)
        
        # 灰度转换
        if self.use_grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = np.expand_dims(image, axis=0)
            
        # 归一化
        image = image.astype(np.float32) / 255.0
        return image
```

## 动作空间设计

### 1. 定义动作空间
根据控制需求定义动作空间：

```python
def _create_action_space(self) -> gym.Space:
    # 连续动作空间: [steer, throttle, brake]
    return spaces.Box(
        low=np.array([-1.0, 0.0, 0.0]),
        high=np.array([1.0, 1.0, 1.0]),
        dtype=np.float32
    )
    
    # 或者离散动作空间
    # return spaces.Discrete(9)  # 9个预定义动作
```

### 2. 实现动作转换
将算法输出的动作转换为车辆控制：

```python
def _apply_action(self, action: np.ndarray):
    if isinstance(self.action_space, spaces.Box):
        # 连续动作处理
        steer = float(action[0])  # 转向 [-1, 1]
        throttle = float(action[1])  # 油门 [0, 1]
        brake = float(action[2])  # 刹车 [0, 1]
    else:
        # 离散动作处理
        steer, throttle, brake = self._discrete_to_continuous(action)
        
    # 应用到车辆
    control = carla.VehicleControl(
        throttle=max(0, min(1, throttle)),
        steer=max(-1, min(1, steer)),
        brake=max(0, min(1, brake))
    )
    self.vehicle.apply_control(control)
```

## 奖励函数设计

### 1. 创建奖励生成器
将奖励函数模块化：

```python
class CustomRewardGenerator:
    def __init__(self, config: Dict):
        self.weights = config.get('reward_weights', {
            'distance': 1.0,
            'speed': 0.5,
            'collision': 1.0,
            'comfort': 0.2
        })
        
    def generate(self, obs: Dict, action: np.ndarray, info: Dict) -> Tuple[float, Dict]:
        rewards = {}
        
        # 计算各项奖励
        rewards['distance'] = self._distance_reward(info)
        rewards['speed'] = self._speed_reward(info)
        rewards['collision'] = self._collision_penalty(info)
        rewards['comfort'] = self._comfort_reward(action)
        
        # 加权求和
        total_reward = sum(
            self.weights[k] * v for k, v in rewards.items()
        )
        
        return total_reward, rewards
```

### 2. 实现具体奖励函数
根据任务目标设计具体的奖励函数：

```python
def _distance_reward(self, info: Dict) -> float:
    """目标导向奖励"""
    distance = info['target_distance']
    return np.exp(-distance / 10.0)  # 距离越近奖励越大

def _speed_reward(self, info: Dict) -> float:
    """速度奖励"""
    speed = info['speed']
    target_speed = self.config['target_speed']
    speed_diff = abs(speed - target_speed)
    
    if speed_diff < 5.0:
        return 1.0
    else:
        return -speed_diff / target_speed

def _collision_penalty(self, info: Dict) -> float:
    """碰撞惩罚"""
    return -100.0 if info['collision'] else 0.0

def _comfort_reward(self, action: np.ndarray) -> float:
    """舒适度奖励"""
    # 惩罚剧烈转向
    steer_penalty = -abs(action[0])
    # 惩罚剧烈加速和刹车
    acc_penalty = -abs(action[1] - action[2])
    return steer_penalty + acc_penalty
```

## 完整示例

### 1. 配置文件
创建任务配置文件：

```yaml
# configs/tasks/custom_task.yaml
observation:
  image_size: [84, 84]
  use_grayscale: true
  lidar_range: 50.0
  num_lidar_bins: 64

action:
  type: "continuous"  # or "discrete"
  discrete_actions: 9

rewards:
  weights:
    distance: 1.0
    speed: 0.5
    collision: 1.0
    comfort: 0.2
  target_speed: 30.0
  collision_penalty: -100.0
  success_reward: 100.0
```

### 2. 环境实现
创建自定义环境：

```python
class CustomEnv(CarlaEnv):
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # 创建观察和动作空间
        self.observation_space = self._create_observation_space()
        self.action_space = self._create_action_space()
        
        # 创建观测处理器和奖励生成器
        self.obs_processor = CustomObservationProcessor(config['observation'])
        self.reward_generator = CustomRewardGenerator(config['rewards'])
        
    def step(self, action):
        # 执行动作
        obs, _, done, info = super().step(action)
        
        # 处理观测
        processed_obs = self.obs_processor.process(obs)
        
        # 计算奖励
        reward, reward_info = self.reward_generator.generate(
            processed_obs, action, info
        )
        
        # 更新信息
        info.update(reward_info)
        
        return processed_obs, reward, done, info
```

### 3. 使用示例
使用自定义环境：

```python
# 加载配置
with open('configs/tasks/custom_task.yaml', 'r') as f:
    config = yaml.safe_load(f)
    
# 创建环境
env = CustomEnv(config)

# 测试环境
obs = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    
    if done:
        obs = env.reset()
```

## 最佳实践

1. **观察空间设计**
   - 保持观测维度合理，避免过大
   - 进行适当的预处理和归一化
   - 考虑数据的时序性

2. **动作空间设计**
   - 根据任务选择合适的动作类型
   - 考虑动作的物理约束
   - 实现平滑的动作转换

3. **奖励函数设计**
   - 奖励应该稀疏且明确
   - 使用分层的奖励结构
   - 避免奖励冲突
   - 合理设置权重

4. **调试建议**
   - 使用可视化工具监控训练
   - 记录详细的奖励分解
   - 进行充分的单元测试 

## 可用接口说明

### 传感器接口

#### 1. 相机传感器
```python
# RGB相机数据
obs['camera_rgb']  # shape: (H, W, 3), dtype: uint8, range: [0, 255]

# 深度相机数据
obs['camera_depth']  # shape: (H, W), dtype: float32, unit: meters

# 语义分割相机数据
obs['camera_semantic']  # shape: (H, W), dtype: uint8, range: [0, 13]
# 语义标签: {
#   0: 'None',
#   1: 'Buildings',
#   2: 'Fences',
#   3: 'Other',
#   4: 'Pedestrians',
#   5: 'Poles',
#   6: 'RoadLines',
#   7: 'Roads',
#   8: 'Sidewalks',
#   9: 'Vegetation',
#   10: 'Vehicles',
#   11: 'Walls',
#   12: 'TrafficSigns'
# }
```

#### 2. 激光雷达数据
```python
# 点云数据
obs['lidar']  # shape: (N, 4), dtype: float32
# 每个点包含: [x, y, z, intensity]
# - x, y, z: 点的3D坐标(meters)
# - intensity: 反射强度 [0, 1]
```

#### 3. 其他传感器
```python
# GNSS数据
obs['gnss']  # Dict包含:
# - latitude: 纬度(degrees)
# - longitude: 经度(degrees)
# - altitude: 海拔(meters)

# IMU数据
obs['imu']  # Dict包含:
# - accelerometer: [ax, ay, az] (m/s^2)
# - gyroscope: [gx, gy, gz] (rad/s)
# - compass: 指南针角度(degrees)
```

### 车辆状态接口

#### 1. 基础状态
```python
info['vehicle_state']  # Dict包含:
# 位置信息
- 'position': [x, y, z]  # 全局坐标(meters)
- 'rotation': [roll, pitch, yaw]  # 欧拉角(degrees)

# 运动信息
- 'velocity': [vx, vy, vz]  # 速度(m/s)
- 'angular_velocity': [wx, wy, wz]  # 角速度(rad/s)
- 'acceleration': [ax, ay, az]  # 加速度(m/s^2)

# 控制信息
- 'control': {
    'throttle': float,  # [0, 1]
    'steer': float,  # [-1, 1]
    'brake': float,  # [0, 1]
    'hand_brake': bool,
    'reverse': bool,
    'manual_gear_shift': bool,
    'gear': int
}
```

#### 2. 交通信息
```python